#!/usr/bin/env python
"""
Nemotron Dictate — a local, private dictation menu-bar app for macOS.

Replaces cloud dictation with nvidia/nemotron-3.5-asr-streaming-0.6b running fully
on-device via ONNX Runtime (CPU). No audio ever leaves your Mac.

UX:
    Double-tap Right-Command  -> 🔴 recording
    Double-tap again          -> ✍️ transcribe -> text typed into the focused app -> 🎤 idle

This is the PACKAGED app entry point. It is ONNX-only: it imports onnx_engine and
NEVER imports torch / NeMo / stream_engine. Weights live in Application Support and
are downloaded on first run.

=========================  MACHINE-SAFETY  ====================================
Mic stream is opened on record-start and ALWAYS closed (finally) on record-stop;
never held open while idle. SIGINT/SIGTERM and the Quit item tear the mic down
cleanly then exit. Model loaded once, kept warm. CPU only. One model. Never needs
a force-kill.
==============================================================================
"""
import os
import sys
import time
import signal
import threading

# ---- keep the bundle lean / quiet; never touch torch or NeMo here ----
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")

# Make onnx_engine importable whether run from source (app/ subdir) or bundled.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))           # app/ (bundled location)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root (dev)

import numpy as np
import rumps

APP_NAME = "Nemotron Dictate"
HF_REPO = "altunenes/parakeet-rs"
HF_SUBDIR = "nemotron-3.5-asr-streaming-0.6b-onnx"
MODEL_FILES = ["config.json", "encoder.onnx", "encoder.onnx.data",
               "decoder_joint.onnx", "tokenizer.model"]

ICON = {"loading": "⏳", "downloading": "⬇️", "idle": "🎤",
        "recording": "🔴", "transcribing": "✍️", "error": "⚠️"}

MAX_RECORD_SECONDS = 120
DOUBLE_TAP_WINDOW = 0.40  # seconds: two taps within this = trigger


# --------------------------------------------------------------------------- #
#  Paths                                                                       #
# --------------------------------------------------------------------------- #
def support_dir():
    p = os.path.expanduser(f"~/Library/Application Support/{APP_NAME}")
    os.makedirs(p, exist_ok=True)
    return p


def model_dir():
    return os.path.join(support_dir(), "onnx_weights")


def log_path():
    return os.path.join(support_dir(), "dictate.log")


def _log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with open(log_path(), "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def model_present():
    d = model_dir()
    return all(os.path.exists(os.path.join(d, f)) and os.path.getsize(os.path.join(d, f)) > 0
               for f in MODEL_FILES)


# --------------------------------------------------------------------------- #
#  Self-contained mic recorder (no torch). Opens device only while recording.  #
# --------------------------------------------------------------------------- #
def _resource_dir():
    """Contents/Resources when bundled (…/MacOS/.. -> Resources), else app/ dir."""
    here = os.path.dirname(os.path.abspath(__file__))
    res = os.path.normpath(os.path.join(here, "..", "Resources"))
    return res if os.path.isdir(res) else here


def _bundled_portaudio():
    """Return the bundled libportaudio.dylib path if present (packaged app)."""
    cand = os.path.join(_resource_dir(), "portaudio-binaries", "libportaudio.dylib")
    return cand if os.path.exists(cand) else None


def _ensure_portaudio_loadable():
    """Make `import sounddevice` find the bundled (unzipped) libportaudio.dylib.

    py2app zips package data and dylibs cannot be dlopen'd from a zip, so the normal
    `_sounddevice_data/.../libportaudio.dylib` path fails. sounddevice's final
    fallback resolves the lib via `_sounddevice_data.__path__`. We pre-install a
    `_sounddevice_data` module whose __path__ points at our real Resources dir, so
    that fallback picks up the bundled dylib. Only acts when a bundled dylib exists
    (i.e. in the packaged app); in dev it's a no-op and the venv copy is used.
    """
    pa = _bundled_portaudio()
    if not pa or "_sounddevice_data" in sys.modules:
        return
    import types
    res = _resource_dir()
    mod = types.ModuleType("_sounddevice_data")
    mod.__path__ = [res]  # so __path__/portaudio-binaries/libportaudio.dylib resolves
    sys.modules["_sounddevice_data"] = mod


class Recorder:
    MODEL_SR = 16000

    def __init__(self):
        _ensure_portaudio_loadable()  # make sounddevice find the bundled dylib
        import sounddevice as sd
        self._sd = sd
        info = sd.query_devices(kind="input")
        self.device_sr = int(info["default_samplerate"])
        self._frames = []
        self._stream = None
        self._lock = threading.Lock()

    def _cb(self, indata, frames, t, status):
        self._frames.append(indata.copy())

    def start(self):
        with self._lock:
            if self._stream is not None:
                return
            self._frames = []
            self._stream = self._sd.InputStream(
                samplerate=self.device_sr, channels=1, dtype="float32", callback=self._cb)
            self._stream.start()

    def stop(self):
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
        if self.device_sr != self.MODEL_SR:
            n = int(round(audio.size * self.MODEL_SR / self.device_sr))
            audio = np.interp(np.linspace(0, 1, n, endpoint=False),
                              np.linspace(0, 1, audio.size, endpoint=False), audio).astype(np.float32)
        return audio


# --------------------------------------------------------------------------- #
#  Text injection: clipboard + Cmd-V (preserves accents). Needs Accessibility. #
# --------------------------------------------------------------------------- #
def inject_text(text):
    if not text:
        return
    import subprocess
    subprocess.run(["pbcopy"], input=text.encode("utf-8"))
    from pynput.keyboard import Controller, Key
    kb = Controller()
    with kb.pressed(Key.cmd):
        kb.press("v")
        kb.release("v")


# --------------------------------------------------------------------------- #
#  Permissions (macOS) — best-effort checks + open the right Settings pane.     #
# --------------------------------------------------------------------------- #
SETTINGS_PANES = {
    "microphone": "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone",
    "accessibility": "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility",
    "input": "x-apple.systempreferences:com.apple.preference.security?Privacy_ListenEvent",
}


def open_settings(pane):
    import subprocess
    url = SETTINGS_PANES.get(pane)
    if url:
        subprocess.run(["open", url])


def check_microphone():
    """Returns True/False/None (unknown). Uses AVFoundation auth status if available."""
    try:
        import AVFoundation
        st = AVFoundation.AVCaptureDevice.authorizationStatusForMediaType_(
            AVFoundation.AVMediaTypeAudio)
        # 3 = authorized, 2 = denied, 1 = restricted, 0 = not determined
        if st == 3:
            return True
        if st in (1, 2):
            return False
        return None
    except Exception:
        return None


def check_accessibility():
    try:
        from ApplicationServices import AXIsProcessTrusted
        return bool(AXIsProcessTrusted())
    except Exception:
        return None


# --------------------------------------------------------------------------- #
#  Model download (first run)                                                  #
# --------------------------------------------------------------------------- #
def download_model(progress_cb=None):
    """Download the ONNX export into model_dir(). Returns True on success."""
    from huggingface_hub import hf_hub_download
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    d = model_dir()
    os.makedirs(d, exist_ok=True)
    for i, fn in enumerate(MODEL_FILES):
        if progress_cb:
            progress_cb(f"Downloading {fn} ({i+1}/{len(MODEL_FILES)})…")
        _log(f"download {fn}")
        path = hf_hub_download(repo_id=HF_REPO, filename=f"{HF_SUBDIR}/{fn}",
                               local_dir=os.path.join(d, "_dl"))
        # move into flat model_dir
        dst = os.path.join(d, fn)
        if os.path.abspath(path) != os.path.abspath(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            import shutil
            shutil.copy2(path, dst)
    return model_present()


# --------------------------------------------------------------------------- #
#  The app                                                                     #
# --------------------------------------------------------------------------- #
class DictateApp(rumps.App):
    def __init__(self, lang="auto"):
        super().__init__(ICON["loading"], quit_button=None)
        self.lang = lang
        self._status = "loading"
        self._status_detail = "Starting…"
        self._last_text = ""
        self._lock = threading.Lock()
        self._rec_started_at = None
        self._last_tap = 0.0
        self.rec = None
        self.trx = None
        self._needs_main_load = False

        self.status_item = rumps.MenuItem("Starting…")
        self.lang_item = rumps.MenuItem(f"Language: {self.lang}")
        self.last_item = rumps.MenuItem("Last: —")
        self.menu = [
            self.status_item,
            None,
            rumps.MenuItem("Start / Stop (manual)", callback=self._manual_toggle),
            self._lang_menu(),
            self.last_item,
            None,
            rumps.MenuItem("Open permissions…", callback=self._open_perms),
            rumps.MenuItem("Show log", callback=self._show_log),
            rumps.MenuItem("Quit", callback=self._quit),
        ]

        self._listener = None
        signal.signal(signal.SIGINT, lambda *_: self._quit(None))
        signal.signal(signal.SIGTERM, lambda *_: self._quit(None))
        self._ui_timer = rumps.Timer(self._refresh_ui, 0.15)
        self._ui_timer.start()

        # Heavy startup (model load) runs on the MAIN thread via a one-shot timer.
        # Importing/initializing native libs (onnxruntime, sentencepiece) off the
        # main thread can deadlock with dyld/ObjC during macOS app init; the main
        # thread is the safe place. The menu bar is already visible by the time this
        # fires (~0.5s), so the app stays responsive while it loads.
        self._startup_done = False
        self._startup_timer = rumps.Timer(self._startup_tick, 0.5)
        self._startup_timer.start()

    def _startup_tick(self, _):
        if self._startup_done:
            return
        self._startup_done = True
        self._startup_timer.stop()
        self._startup()  # runs on the main thread

    # --------------------------------------------------------------- startup --
    def _startup(self):
        try:
            self._startup_inner()
        except Exception as e:
            import traceback
            self._set("error", f"Startup failed: {e}")
            _log(f"STARTUP CRASH: {e!r}\n{traceback.format_exc()}")

    def _startup_inner(self):
        _log(f"startup begin (model_present={model_present()}, model_dir={model_dir()})")
        try:
            self.rec = Recorder()
            _log("recorder ready")
        except Exception as e:
            self._set("error", f"Mic init failed: {e}")
            _log(f"recorder init failed: {e!r}")

        if model_present():
            # already downloaded -> load now on this (main) thread
            self._load_model_main()
        else:
            # first run: download on a WORKER thread (keeps UI responsive), then
            # hand back to the main thread to load the model.
            self._set("downloading", "First run: downloading model (~2.4 GB)…")
            threading.Thread(target=self._download_then_load, daemon=True).start()

    def _download_then_load(self):
        try:
            ok = download_model(progress_cb=lambda m: self._set("downloading", m))
            if not ok:
                self._set("error", "Model download incomplete — check network, reopen app.")
                return
        except Exception as e:
            self._set("error", f"Download failed: {e}")
            _log(f"download failed: {e!r}")
            return
        # model present now; trigger a main-thread load via the flag the UI timer watches
        self._set("loading", "Loading model…")
        self._needs_main_load = True  # picked up by _refresh_ui on the main thread

    def _load_model_main(self):
        """Load the ONNX engine. MUST run on the main thread (native init safety)."""
        self._set("loading", "Loading model…")
        try:
            from onnx_engine import StreamingTranscriber
            self.trx = StreamingTranscriber(device="cpu", lang=self.lang, onnx_dir=model_dir())
            _log("model ready")
        except Exception as e:
            self._set("error", f"Model load failed: {e}")
            _log(f"model load failed: {e!r}")
            return
        self._start_listener()
        self._set("idle", "Ready")

    def _start_listener(self):
        from pynput import keyboard
        self._keyboard = keyboard

        def on_press(key):
            # double-tap Right-Command = trigger
            if key == keyboard.Key.cmd_r:
                now = time.time()
                if now - self._last_tap < DOUBLE_TAP_WINDOW:
                    self._last_tap = 0.0
                    self._toggle()
                else:
                    self._last_tap = now
            # optional advanced trigger: F18 (Karabiner from mic key)
            elif key == getattr(keyboard.Key, "f18", None):
                self._toggle()

        self._listener = keyboard.Listener(on_press=on_press, suppress=False)
        self._listener.start()

    # ----------------------------------------------------------------- menu --
    def _lang_menu(self):
        m = rumps.MenuItem("Set language")
        for code in ["auto", "en-US", "fr-FR"]:
            m.add(rumps.MenuItem(code, callback=self._set_lang))
        return m

    def _set_lang(self, sender):
        self.lang = sender.title
        if self.trx is not None:
            self.trx.lang = self.lang
        self.lang_item.title = f"Language: {self.lang}"

    def _open_perms(self, _):
        open_settings("microphone")
        time.sleep(0.3)
        open_settings("accessibility")

    def _show_log(self, _):
        import subprocess
        subprocess.run(["open", log_path()])

    # ------------------------------------------------------------- helpers ---
    def _set(self, status, detail=None):
        with self._lock:
            self._status = status
            if detail is not None:
                self._status_detail = detail

    # ------------------------------------------------------------ UI timer --
    def _refresh_ui(self, _):
        if not getattr(self, "_sig_reasserted", False):
            signal.signal(signal.SIGINT, lambda *_: self._quit(None))
            signal.signal(signal.SIGTERM, lambda *_: self._quit(None))
            self._sig_reasserted = True
        # post-download model load happens HERE, on the main thread (native-init safe)
        if getattr(self, "_needs_main_load", False):
            self._needs_main_load = False
            self._load_model_main()
        with self._lock:
            st, detail = self._status, self._status_detail
        self.title = ICON.get(st, "🎤")
        if st == "idle":
            self.status_item.title = "Ready — double-tap Right ⌘ to dictate"
        else:
            self.status_item.title = detail
        if self._last_text:
            preview = (self._last_text[:40] + "…") if len(self._last_text) > 40 else self._last_text
            self.last_item.title = f"Last: {preview}"
        if st == "recording" and self._rec_started_at:
            if time.time() - self._rec_started_at > MAX_RECORD_SECONDS:
                threading.Thread(target=self._stop_and_transcribe, daemon=True).start()

    # -------------------------------------------------------------- hotkey --
    def _manual_toggle(self, _):
        self._toggle()

    def _toggle(self):
        with self._lock:
            st = self._status
        if st == "idle":
            self._start()
        elif st == "recording":
            threading.Thread(target=self._stop_and_transcribe, daemon=True).start()

    def _start(self):
        if self.rec is None:
            return
        try:
            self.rec.start()
        except Exception as e:
            _log(f"mic open failed: {e!r}")
            return
        with self._lock:
            self._status = "recording"
            self._rec_started_at = time.time()

    def _stop_and_transcribe(self):
        audio = None
        try:
            audio = self.rec.stop()
        except Exception as e:
            _log(f"mic close issue: {e!r}")
        with self._lock:
            self._status = "transcribing"
            self._rec_started_at = None
        text = ""
        if audio is not None and self.trx is not None:
            try:
                self.trx.reset()
                self.trx.feed(audio)
                text = self.trx.finalize()
            except Exception as e:
                _log(f"transcribe error: {e!r}")
        if text:
            with self._lock:
                self._last_text = text
            try:
                inject_text(text)
            except Exception as e:
                _log(f"inject failed (Accessibility?): {e!r}")
            _log(f'-> "{text}"')
        with self._lock:
            self._status = "idle"

    # --------------------------------------------------------------- quit ---
    def _quit(self, _):
        try:
            if self.rec:
                self.rec.stop()
        except Exception:
            pass
        try:
            if self._listener:
                self._listener.stop()
        except Exception:
            pass
        _log("clean shutdown — mic released.")
        rumps.quit_application()
        os._exit(0)


def _preimport_heavy():
    """Import the native extension modules ON THE MAIN THREAD before the rumps event
    loop / worker thread start. Importing native libs (sentencepiece, onnxruntime)
    off the main thread can deadlock with dyld/ObjC during app init on macOS. Doing
    it here makes the later worker-thread construction just reuse loaded modules."""
    try:
        import onnxruntime  # noqa
        import sentencepiece  # noqa
        import soundfile  # noqa
        import onnx_engine  # noqa  (transitively imports the above; warms the import)
    except Exception as e:
        _log(f"pre-import warning: {e!r}")


def main():
    # selftest hook for headless build verification (NO mic, reads a WAV)
    if len(sys.argv) >= 3 and sys.argv[1] == "--selftest":
        from onnx_engine import StreamingTranscriber
        import soundfile as sf
        wav = sys.argv[2]
        md = model_dir() if model_present() else None
        kw = {"device": "cpu", "lang": "en-US"}
        if md:
            kw["onnx_dir"] = md
        trx = StreamingTranscriber(**kw)
        audio, sr = sf.read(wav, dtype="float32")
        trx.reset(); trx.feed(audio)
        print("[selftest]", trx.finalize())
        return
    _preimport_heavy()  # load native extensions on the main thread (avoid dyld deadlock)
    DictateApp(lang="auto").run()


if __name__ == "__main__":
    main()
