#!/usr/bin/env python
"""
Push-to-talk dictation daemon for nvidia/nemotron-3.5-asr-streaming-0.6b on Apple Silicon.

HOLD the hotkey -> speak -> RELEASE -> the model transcribes and the text is typed
into whatever app is focused (via clipboard + Cmd-V, which preserves accents).

Default hotkey: hold RIGHT-OPTION (right alt) key. Configurable with --hotkey.

=========================  MACHINE-SAFETY (read this)  =========================
The 2026-06-08 audio-daemon disaster was caused by FORCE-KILLING processes mid
CoreAudio init -- NOT by the mic itself. This script is built so you NEVER have
to force-kill it:

  * The mic input stream is OPENED on hotkey-down and CLOSED on hotkey-up. The
    device is NOT held open while idle.
  * Every stream open is wrapped so it is ALWAYS stopped+closed (try/finally),
    even if transcription throws.
  * SIGINT (Ctrl-C) and SIGTERM are trapped: they stop any active stream cleanly,
    release the keyboard listener, and exit 0. So Ctrl-C is always a clean quit.
  * The model is loaded ONCE and kept warm (one model at a time).

If you ever see it misbehave: just press Ctrl-C. It will tear down cleanly.
===============================================================================
"""
import os
import sys
import time
import signal
import threading
import argparse

# MPS fallback for the few RNNT ops not yet on Metal.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np
import sounddevice as sd
import torch

MODEL = "nvidia/nemotron-3.5-asr-streaming-0.6b"
MODEL_SR = 16000  # the model wants 16kHz mono


# --------------------------------------------------------------------------- #
#  Audio recorder: opens the mic ONLY while recording, closes it cleanly.     #
# --------------------------------------------------------------------------- #
class PushToTalkRecorder:
    def __init__(self, device_sr=None):
        info = sd.query_devices(kind="input")
        self.device_sr = int(device_sr or info["default_samplerate"])
        self._frames = []
        self._stream = None
        self._lock = threading.Lock()

    def _callback(self, indata, frames, time_info, status):
        # status can warn about overflows; we just keep the audio.
        self._frames.append(indata.copy())

    def start(self):
        """Open the input stream. Safe to call on hotkey-down."""
        with self._lock:
            if self._stream is not None:
                return
            self._frames = []
            self._stream = sd.InputStream(
                samplerate=self.device_sr,
                channels=1,
                dtype="float32",
                callback=self._callback,
                blocksize=0,  # let PortAudio pick a small sane buffer
            )
            self._stream.start()

    def stop(self):
        """Stop + close the stream cleanly. ALWAYS safe to call (idempotent)."""
        with self._lock:
            if self._stream is None:
                return None
            try:
                self._stream.stop()
            finally:
                self._stream.close()
                self._stream = None
        if not self._frames:
            return None
        audio = np.concatenate(self._frames, axis=0).reshape(-1).astype(np.float32)
        self._frames = []
        # Resample device_sr -> 16kHz (linear; speech is fine with this).
        if self.device_sr != MODEL_SR:
            audio = _resample_linear(audio, self.device_sr, MODEL_SR)
        return audio


def _resample_linear(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out or x.size == 0:
        return x
    n_out = int(round(x.size * sr_out / sr_in))
    xp = np.linspace(0.0, 1.0, num=x.size, endpoint=False)
    fp = x
    xq = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(xq, xp, fp).astype(np.float32)


# --------------------------------------------------------------------------- #
#  Model wrapper: loaded once, kept warm.                                     #
# --------------------------------------------------------------------------- #
class Transcriber:
    def __init__(self, device="mps", lang="auto"):
        self.device = device
        self.lang = lang
        print(f"[dictate] loading {MODEL} on {device} (one-time, ~20s)...", flush=True)
        import nemo.collections.asr as nemo_asr
        self.model = nemo_asr.models.ASRModel.from_pretrained(MODEL).eval()
        self.model = self.model.to(torch.device(device))
        # Warm the Metal kernels with 1s of silence so the first real dictation is fast.
        self._warm()
        print("[dictate] model ready.", flush=True)

    def _warm(self):
        import tempfile, json, soundfile as sf
        silence = np.zeros(MODEL_SR, dtype=np.float32)
        wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(wav.name, silence, MODEL_SR)
        try:
            self._run(wav.name)
        except Exception:
            pass
        finally:
            os.unlink(wav.name)

    def _run(self, wav_path):
        import tempfile, json
        lang = self.lang
        man = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8")
        with man as fp:
            fp.write(json.dumps({
                "audio_filepath": os.path.abspath(wav_path),
                "duration": 100000, "text": "", "lang": lang,
            }) + "\n")
        try:
            with torch.inference_mode():
                out = self.model.transcribe([man.name], batch_size=1,
                                            target_lang=lang, verbose=False)
        finally:
            os.unlink(man.name)
        text = ""
        if out:
            h = out[0]
            text = h.text if hasattr(h, "text") else (
                h[0].text if isinstance(h, (list, tuple)) and h and hasattr(h[0], "text") else str(h))
        return text

    def transcribe_audio(self, audio: np.ndarray) -> str:
        import tempfile, soundfile as sf
        if audio is None or audio.size < MODEL_SR * 0.2:  # <0.2s -> ignore
            return ""
        wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(wav.name, audio, MODEL_SR)
        try:
            text = self._run(wav.name)
        finally:
            os.unlink(wav.name)
        # Strip the model's <en-US>/<fr-FR> language tags and tidy.
        import re
        text = re.sub(r"\s*<[a-z]{2}-[A-Z]{2}>\s*", " ", text).strip()
        return text


# --------------------------------------------------------------------------- #
#  Text injection: clipboard + Cmd-V (preserves accents). Needs Accessibility.#
# --------------------------------------------------------------------------- #
def inject_text(text: str):
    if not text:
        return
    # 1) put text on the clipboard via pbcopy
    import subprocess
    p = subprocess.run(["pbcopy"], input=text.encode("utf-8"))
    # 2) send Cmd-V to the focused app
    from pynput.keyboard import Controller, Key
    kb = Controller()
    with kb.pressed(Key.cmd):
        kb.press("v")
        kb.release("v")


# --------------------------------------------------------------------------- #
#  Hotkey loop                                                                #
# --------------------------------------------------------------------------- #
HOTKEYS = {
    "right_option": "alt_r",
    "right_cmd": "cmd_r",
    "right_ctrl": "ctrl_r",
    "fn": "fn",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps", choices=["mps", "cpu"])
    ap.add_argument("--lang", default="auto",
                    help="auto | en-US | fr-FR | ... (auto detects per utterance)")
    ap.add_argument("--hotkey", default="right_option", choices=list(HOTKEYS.keys()))
    args = ap.parse_args()

    from pynput import keyboard

    rec = PushToTalkRecorder()
    trx = Transcriber(device=args.device, lang=args.lang)

    target_key_name = HOTKEYS[args.hotkey]
    state = {"recording": False, "stop": False}

    def cleanup_and_exit(*_):
        # Trapped by SIGINT/SIGTERM -> guarantees clean teardown, no force-kill.
        state["stop"] = True
        try:
            rec.stop()  # idempotent; closes mic if open
        except Exception:
            pass
        print("\n[dictate] clean shutdown -- mic released, exiting.", flush=True)
        os._exit(0)

    signal.signal(signal.SIGINT, cleanup_and_exit)
    signal.signal(signal.SIGTERM, cleanup_and_exit)

    def key_matches(key):
        name = getattr(key, "name", None)
        return name == target_key_name

    def on_press(key):
        if key_matches(key) and not state["recording"]:
            state["recording"] = True
            try:
                rec.start()
                print("[dictate] 🎙  recording... (release hotkey to transcribe)", flush=True)
            except Exception as e:
                state["recording"] = False
                print(f"[dictate] mic open failed: {e!r}", flush=True)

    def on_release(key):
        if key_matches(key) and state["recording"]:
            state["recording"] = False
            audio = None
            try:
                audio = rec.stop()  # closes mic cleanly
            except Exception as e:
                print(f"[dictate] mic close issue: {e!r}", flush=True)
            if audio is not None:
                t0 = time.time()
                text = trx.transcribe_audio(audio)
                dt = time.time() - t0
                if text:
                    print(f"[dictate] -> \"{text}\"  ({dt:.2f}s)", flush=True)
                    inject_text(text)
                else:
                    print("[dictate] (nothing recognized)", flush=True)

    print(f"[dictate] READY. Hold [{args.hotkey}] to dictate. Ctrl-C to quit cleanly.", flush=True)
    print(f"[dictate] language={args.lang}  device={args.device}", flush=True)

    # suppress=False: we do NOT swallow the key, so the modifier still works normally.
    listener = keyboard.Listener(on_press=on_press, on_release=on_release, suppress=False)
    listener.start()
    try:
        while not state["stop"]:
            time.sleep(0.1)
    finally:
        listener.stop()
        rec.stop()


if __name__ == "__main__":
    main()
