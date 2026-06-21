#!/usr/bin/env python
"""
Nemotron LIVE Dictation — menu-bar app, words appear in the focused field AS YOU SPEAK.

    Tap mic key (F5 🎤 → F18)  -> 🔴 listening: text streams into the app live
    Tap it again               -> ✍️ finalize -> 🎤 idle

Features:
  * Real-time streaming transcription (cache-aware NeMo, pure-append → smooth typing)
  * Floating on-screen indicator while recording (🔴 listening…)
  * Audio ducking: lowers other apps' volume while you talk, restores after
  * Pause / Resume: unload the model to free the GPU + RAM for HyperRead, reload on demand
  * Always-on: run from a LaunchAgent so it loads once at login (no terminal)

=========================  MACHINE-SAFETY  ====================================
  * Mic stream opened on record-start, closed on record-stop (try/finally). Never
    held open while idle. One model, loaded once, float32 (cache-aware requirement).
  * Pause fully unloads the model (torch.mps.empty_cache) → zero GPU/RAM, so you can
    run HyperRead's TTS with no two-models-on-the-GPU risk.
  * Audio ducking uses the system volume setting (osascript) — it does NOT open any
    CoreAudio device. We never open an audio output device.
  * SIGINT/SIGTERM reclaimed on the first UI tick (rumps/PyObjC steals them) + menu
    Quit → all tear the mic down cleanly. Never needs a force-kill.
===============================================================================
"""
import os
import sys
import time
import signal
import gc
import threading
import argparse
import subprocess
import json
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np
import sounddevice as sd

from live_inject import LiveTyper, diff_plan

MODEL_SR = 16000   # both engines use 16kHz mono (engine class is chosen in main)

import rumps
from pynput import keyboard

ICON = {"loading": "⏳", "idle": "🎤", "listening": "🔴", "finishing": "✍️", "paused": "⏸"}
CHUNK_MS = 320
CHUNK_SAMPLES = MODEL_SR * CHUNK_MS // 1000
MAX_LISTEN_SECONDS = 120
DUCK_TO_PERCENT = 0.20   # lower other audio to 20% of current while recording
DOUBLETAP_WINDOW = 0.40  # two Right-⌘ presses within this many seconds = toggle

SETTINGS_PATH = Path.home() / "Library/Application Support/NemotronDictate/config.json"


def _load_settings():
    try:
        return json.loads(SETTINGS_PATH.read_text())
    except Exception:
        return {}


def _save_settings(d):
    try:
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS_PATH.write_text(json.dumps(d, indent=2))
    except Exception:
        pass


# --------------------------------------------------------------------------- #
#  Audio ducking via the system volume setting (NOT a CoreAudio device open).  #
# --------------------------------------------------------------------------- #
def _get_output_volume():
    try:
        out = subprocess.run(["osascript", "-e", "output volume of (get volume settings)"],
                             capture_output=True, text=True, timeout=2)
        return int(out.stdout.strip())
    except Exception:
        return None


def _set_output_volume(v):
    try:
        subprocess.run(["osascript", "-e", f"set volume output volume {int(v)}"], timeout=2)
    except Exception:
        pass


def _fade_volume(frm, to, duration=0.6, steps=14):
    """Glide the system volume frm->to over `duration` (gradual, so restores don't
    blast the ears). Runs in a daemon thread; non-blocking."""
    if frm is None or to is None:
        return
    frm, to = int(frm), int(to)

    def run():
        for i in range(1, steps + 1):
            _set_output_volume(round(frm + (to - frm) * i / steps))
            time.sleep(duration / steps)
    threading.Thread(target=run, daemon=True).start()


# --------------------------------------------------------------------------- #
#  Floating on-screen indicator (borderless, click-through, always on top).    #
# --------------------------------------------------------------------------- #
class FloatingIndicator:
    """A Dynamic-Island-style glass pill that hugs the notch on the built-in display.
    Frosted (NSVisualEffectView), centered under the notch, with a pulsing red dot."""
    def __init__(self):
        self._panel = None
        self._dot = None
        self._pulse_on = True

    def _notch_screen(self):
        from AppKit import NSScreen
        # the built-in display is the one with a top safe-area inset (the notch)
        for s in NSScreen.screens():
            try:
                if s.safeAreaInsets().top > 0:
                    return s
            except Exception:
                pass
        return NSScreen.mainScreen()

    def _build(self):
        from AppKit import (NSPanel, NSColor, NSScreen, NSTextField, NSFont, NSView,
                            NSVisualEffectView, NSVisualEffectMaterialHUDWindow,
                            NSVisualEffectBlendingModeBehindWindow, NSVisualEffectStateActive,
                            NSWindowStyleMaskBorderless, NSWindowStyleMaskNonactivatingPanel,
                            NSBackingStoreBuffered, NSMakeRect)
        # status-window level so it floats above everything
        try:
            from AppKit import NSStatusWindowLevel as LEVEL
        except Exception:
            LEVEL = 25

        screen = self._notch_screen()
        f = screen.frame()
        vf = screen.visibleFrame()                 # excludes the menu bar / notch row
        w, h = 128.0, 24.0
        self._w, self._h = w, h
        x = f.origin.x + (f.size.width - w) / 2.0  # centered on full width → under the notch
        y = vf.origin.y + vf.size.height - h - 7.0 # hang just BELOW the menu bar (notch can't draw)
        style = NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel
        panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            ((x, y), (w, h)), style, NSBackingStoreBuffered, False)
        panel.setLevel_(LEVEL)
        panel.setOpaque_(False)
        panel.setBackgroundColor_(NSColor.clearColor())
        panel.setIgnoresMouseEvents_(True)
        panel.setHasShadow_(True)
        panel.setCollectionBehavior_((1 << 0) | (1 << 8))  # AllSpaces | FullScreenAuxiliary

        # frosted-glass pill (Apple Dynamic-Island vibrancy). HUD material = dark + translucent.
        effect = NSVisualEffectView.alloc().initWithFrame_(NSMakeRect(0, 0, w, h))
        effect.setMaterial_(NSVisualEffectMaterialHUDWindow)
        effect.setBlendingMode_(NSVisualEffectBlendingModeBehindWindow)
        effect.setState_(NSVisualEffectStateActive)
        effect.setWantsLayer_(True)
        effect.layer().setCornerRadius_(h / 2.0)
        effect.layer().setMasksToBounds_(True)
        panel.setContentView_(effect)
        content = effect

        dot = NSView.alloc().initWithFrame_(NSMakeRect(13, (h - 7) / 2.0, 7, 7))
        dot.setWantsLayer_(True)
        dot.layer().setCornerRadius_(3.5)
        dot.layer().setBackgroundColor_(
            NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 0.23, 0.19, 1.0).CGColor())
        content.addSubview_(dot)

        label = NSTextField.alloc().initWithFrame_(NSMakeRect(25, (h - 15) / 2.0, w - 32, 15))
        label.setBezeled_(False)
        label.setDrawsBackground_(False)
        label.setEditable_(False)
        label.setSelectable_(False)
        label.setAlignment_(0)  # left
        label.setTextColor_(NSColor.whiteColor())
        label.setFont_(NSFont.systemFontOfSize_(11.5))
        label.setStringValue_("Listening")
        content.addSubview_(label)

        self._panel, self._dot = panel, dot

    def _screen_for_mouse(self):
        """The screen the cursor is on — so the pill shows where you're working."""
        from AppKit import NSEvent, NSScreen
        try:
            loc = NSEvent.mouseLocation()
            for s in NSScreen.screens():
                fr = s.frame()
                if (fr.origin.x <= loc.x <= fr.origin.x + fr.size.width and
                        fr.origin.y <= loc.y <= fr.origin.y + fr.size.height):
                    return s
        except Exception:
            pass
        return self._notch_screen()

    def _position(self, screen):
        f = screen.frame()
        vf = screen.visibleFrame()
        w, h = self._w, self._h
        x = f.origin.x + (f.size.width - w) / 2.0
        y = vf.origin.y + vf.size.height - h - 7.0
        self._panel.setFrame_display_(((x, y), (w, h)), True)

    def show(self):
        try:
            if self._panel is None:
                self._build()
            self._position(self._screen_for_mouse())   # follow the active screen
            self._panel.orderFrontRegardless()
        except Exception as e:
            print(f"[live] indicator show failed (non-fatal): {e!r}", flush=True)

    def pulse(self):
        """Toggle the red dot opacity for a gentle blink. Call from the main-thread timer."""
        try:
            if self._dot is not None and self._panel is not None and self._panel.isVisible():
                self._pulse_on = not self._pulse_on
                self._dot.layer().setOpacity_(1.0 if self._pulse_on else 0.35)
        except Exception:
            pass

    def hide(self):
        try:
            if self._panel is not None:
                self._panel.orderOut_(None)
        except Exception:
            pass


def _resample_linear(x, sr_in, sr_out):
    if sr_in == sr_out or x.size == 0:
        return x.astype(np.float32)
    n = int(round(x.size * sr_out / sr_in))
    return np.interp(np.linspace(0, 1, n, endpoint=False),
                     np.linspace(0, 1, x.size, endpoint=False), x).astype(np.float32)


class StreamMic:
    """Continuous mic capture → float32 mono 16kHz, drained incrementally.
    Open only between start() and stop()."""
    def __init__(self):
        self._buf = []
        self._lock = threading.Lock()
        self._stream = None
        self._src_sr = MODEL_SR
        self._need_resample = False

    def _callback(self, indata, frames, time_info, status):
        with self._lock:
            self._buf.append(indata.copy().reshape(-1))

    def start(self):
        with self._lock:
            self._buf = []
        try:
            self._stream = sd.InputStream(samplerate=MODEL_SR, channels=1,
                                          dtype="float32", callback=self._callback)
            self._stream.start()
            self._src_sr, self._need_resample = MODEL_SR, False
            return
        except Exception:
            self._stream = None
        info = sd.query_devices(kind="input")
        self._src_sr = int(info["default_samplerate"])
        self._need_resample = self._src_sr != MODEL_SR
        self._stream = sd.InputStream(samplerate=self._src_sr, channels=1,
                                      dtype="float32", callback=self._callback)
        self._stream.start()

    def drain(self):
        with self._lock:
            if not self._buf:
                return np.zeros(0, dtype=np.float32)
            block = np.concatenate(self._buf)
            self._buf = []
        if self._need_resample:
            block = _resample_linear(block, self._src_sr, MODEL_SR)
        return block

    def stop(self):
        with self._lock:
            stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
            finally:
                stream.close()
        return self.drain()


class LiveDictateApp(rumps.App):
    def __init__(self, trigger="f18", device="mps", lang="auto", duck=True, engine_cls=None):
        super().__init__(ICON["loading"], quit_button=None)
        self.trigger_name = trigger
        self.device = device
        self._engine_cls = engine_cls
        _s = _load_settings()                      # remember language + duck + stop-on-enter
        self.lang = _s.get("lang", lang)
        self.duck_enabled = _s.get("duck", duck)
        self.stop_on_enter = _s.get("stop_on_enter", True)

        self._status = "loading"
        self._lock = threading.Lock()
        self._listen_started_at = None
        self._worker = None
        self._saved_vol = None
        self._indicator_on = False

        self.mic = StreamMic()
        self.typer = LiveTyper()
        self.indicator = FloatingIndicator()
        self.engine = None

        self.status_item = rumps.MenuItem("Loading model…")
        self.last_item = rumps.MenuItem("Last: —")
        self.pause_item = rumps.MenuItem("Pause (free GPU)", callback=self._toggle_pause)
        self.duck_item = rumps.MenuItem("Duck audio while recording", callback=self._toggle_duck)
        self.duck_item.state = 1 if self.duck_enabled else 0
        self.enter_item = rumps.MenuItem("Stop on Enter", callback=self._toggle_enter)
        self.enter_item.state = 1 if self.stop_on_enter else 0
        self.menu = [
            self.status_item,
            None,
            rumps.MenuItem("Start / Stop (manual)", callback=lambda _: self._toggle()),
            self.pause_item,
            self.duck_item,
            self.enter_item,
            self._lang_menu(),
            self.last_item,
            None,
            rumps.MenuItem("Quit", callback=self._quit),
        ]

        threading.Thread(target=self._load, daemon=True).start()
        self._listener = keyboard.Listener(on_press=self._on_press, suppress=False)
        self._listener.start()
        self._ui = rumps.Timer(self._refresh_ui, 0.12)
        self._ui.start()

    # ---- menu ----
    def _lang_menu(self):
        m = rumps.MenuItem("Set language")
        for code in ["auto", "en-US", "fr-FR"]:
            m.add(rumps.MenuItem(code, callback=self._set_lang))
        return m

    def _persist(self):
        _save_settings({"lang": self.lang, "duck": self.duck_enabled,
                        "stop_on_enter": self.stop_on_enter})

    def _set_lang(self, sender):
        self.lang = sender.title
        self._persist()
        rumps.notification("Nemotron Dictation", "Language",
                           f"Set to {self.lang} — Pause then Resume to apply.")

    def _toggle_duck(self, sender):
        self.duck_enabled = not self.duck_enabled
        sender.state = 1 if self.duck_enabled else 0
        self._persist()

    def _toggle_enter(self, sender):
        self.stop_on_enter = not self.stop_on_enter
        sender.state = 1 if self.stop_on_enter else 0
        self._persist()

    # ---- model load / unload ----
    def _load(self):
        try:
            self.engine = self._engine_cls(device=self.device, lang=self.lang)
            with self._lock:
                if self._status in ("loading",):
                    self._status = "idle"
        except Exception as e:
            print(f"[live] model load failed: {e!r}", flush=True)
            try:
                rumps.notification("Nemotron Dictation", "Model failed to load", str(e)[:140])
            except Exception:
                pass
            with self._lock:
                self._status = "paused"
            self.pause_item.title = "Resume (reload model)"

    def _toggle_pause(self, _):
        with self._lock:
            st = self._status
        if st == "paused":
            # resume
            with self._lock:
                self._status = "loading"
            self.pause_item.title = "Pause (free GPU)"
            threading.Thread(target=self._load, daemon=True).start()
        else:
            # pause: stop if listening, then unload
            if st == "listening":
                self._stop()
            self._unload()

    def _unload(self):
        with self._lock:
            self._status = "paused"
        eng, self.engine = self.engine, None
        try:
            if eng is not None:
                del eng
            gc.collect()
            import torch
            if hasattr(torch, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception as e:
            print(f"[live] unload note: {e!r}", flush=True)
        self.pause_item.title = "Resume (reload model)"
        print("[live] paused — model unloaded, GPU freed.", flush=True)

    # ---- UI timer (main thread): all Cocoa/title edits live here ----
    def _refresh_ui(self, _):
        if not getattr(self, "_sig_reasserted", False):
            signal.signal(signal.SIGINT, lambda *_: self._quit(None))
            signal.signal(signal.SIGTERM, lambda *_: self._quit(None))
            self._sig_reasserted = True
        with self._lock:
            st = self._status
        self.title = ICON.get(st, "🎤")
        trigger_label = ("double-tap Right ⌘" if self.trigger_name == "doubletap_cmd_r"
                         else f"tap your mic key ({self.trigger_name.upper()})")
        labels = {
            "loading": "Loading model…",
            "idle": f"Ready — {trigger_label}",
            "listening": "🔴 Listening… speak; tap to finish",
            "finishing": "✍️ Finalizing…",
            "paused": "Paused — model unloaded (GPU free)",
        }
        self.status_item.title = labels.get(st, "Ready")
        # floating indicator follows state (show/hide only on change)
        want = (st == "listening")
        if want and not self._indicator_on:
            self.indicator.show()
            self._indicator_on = True
        elif not want and self._indicator_on:
            self.indicator.hide()
            self._indicator_on = False
        if want:
            self._pulse_tick = getattr(self, "_pulse_tick", 0) + 1
            if self._pulse_tick % 5 == 0:   # ~0.6s blink
                self.indicator.pulse()
        # safety auto-stop
        if st == "listening" and self._listen_started_at:
            if time.time() - self._listen_started_at > MAX_LISTEN_SECONDS:
                threading.Thread(target=self._stop, daemon=True).start()

    # ---- hotkey ----
    def _on_press(self, key):
        # "Stop on Enter": while dictating, Return/Enter auto-finishes the dictation
        # (types the last words) AND still submits the chat message — one keystroke.
        # We finalize + type SYNCHRONOUSLY here so the final words land in the field
        # BEFORE this Enter reaches the app (else the tail would leak into the next
        # message). We do NOT suppress Enter (suppress=False on the listener), so it
        # still submits. When not listening, Enter is 100% normal (we do nothing).
        if (self.stop_on_enter and key in (keyboard.Key.enter, getattr(keyboard.Key, "return", None))):
            with self._lock:
                listening = self._status == "listening"
            if listening:
                self._stop(sync=True)   # blocking: finishes + types before we return
            return  # let Enter propagate to submit (suppress=False)

        # Double-tap Right-Command mode (no Karabiner needed): two cmd_r presses
        # within DOUBLETAP_WINDOW = toggle. Selected via trigger='doubletap_cmd_r'.
        if self.trigger_name == "doubletap_cmd_r":
            if key == keyboard.Key.cmd_r:
                now = time.time()
                if now - getattr(self, "_last_rcmd_tap", 0.0) < DOUBLETAP_WINDOW:
                    self._last_rcmd_tap = 0.0
                    self._toggle()
                else:
                    self._last_rcmd_tap = now
            return
        # Single-key mode (default: f18 from Karabiner, or any pynput Key name).
        if key == getattr(keyboard.Key, self.trigger_name, None):
            self._toggle()

    def _toggle(self):
        with self._lock:
            st = self._status
        if st == "idle":
            self._start()
        elif st == "listening":
            threading.Thread(target=self._stop, daemon=True).start()
        # ignore while loading/finishing/paused

    def _start(self):
        if self.engine is None:
            return
        try:
            self.engine.reset()
            self.typer.reset()
            self.mic.start()
        except Exception as e:
            print(f"[live] start failed: {e!r}", flush=True)
            try:
                rumps.notification("Nemotron Dictation", "Microphone error", str(e)[:140])
            except Exception:
                pass
            return
        if self.duck_enabled:
            self._saved_vol = _get_output_volume()
            if self._saved_vol is not None:
                _fade_volume(self._saved_vol, self._saved_vol * DUCK_TO_PERCENT, duration=0.25)
        with self._lock:
            self._status = "listening"
            self._listen_started_at = time.time()
        self._worker = threading.Thread(target=self._stream_loop, daemon=True)
        self._worker.start()

    def _stream_loop(self):
        pending = np.zeros(0, dtype=np.float32)
        # Backlog safety: if decode ever falls behind, DROP the oldest audio instead of
        # letting `pending` grow unbounded (which would make each feed bigger and wedge
        # the loop). We cap the backlog at MAX_BACKLOG_S of audio; anything older is
        # discarded. Per-feed work is already constant (incremental mel), so this is a
        # belt-and-suspenders guard — it keeps the app responsive no matter what.
        MAX_BACKLOG = int(MODEL_SR * 3.0)   # keep at most ~3s queued
        while True:
            with self._lock:
                if self._status != "listening":
                    break
            try:
                pending = np.concatenate([pending, self.mic.drain()])
            except Exception:
                break
            if len(pending) > MAX_BACKLOG:
                # falling behind -> drop the oldest, keep only the most recent audio
                pending = pending[-MAX_BACKLOG:]
            if len(pending) >= CHUNK_SAMPLES and self.engine is not None:
                try:
                    text = self.engine.feed(pending)
                except Exception as e:
                    print(f"[live] feed error (skipping): {e!r}", flush=True)
                    text = None
                pending = np.zeros(0, dtype=np.float32)
                if text:
                    self.typer.update(text)
            time.sleep(0.05)

    def _stop(self, sync=False):
        """Finish dictation — BULLETPROOF: the mic is ALWAYS closed and the app ALWAYS
        returns to idle, even if the engine is mid-decode/behind or throws. We close the
        mic FIRST (it can never stay 'orange stuck on'), then best-effort finalize+type,
        then ALWAYS restore audio + go idle in a finally. When sync=True (Enter path) this
        runs inline in the key handler so the last words land before Enter submits; a
        tight worker-join timeout keeps that latency small."""
        with self._lock:
            if self._status != "listening":
                return
            self._status = "finishing"
            self._listen_started_at = None
        # 1) ALWAYS close the mic first — never leave the input device open.
        remaining = None
        try:
            remaining = self.mic.stop()
        except Exception as e:
            print(f"[live] mic.stop error (ignored): {e!r}", flush=True)
        # 2) stop the worker (bounded wait; we already closed the mic, so even if it
        #    overruns the device is released).
        if self._worker:
            try:
                self._worker.join(timeout=0.3 if sync else 1.5)
            except Exception:
                pass
        try:
            # 3) best-effort finalize + type the last words.
            final = None
            try:
                if self.engine is not None:
                    if remaining is not None and len(remaining):
                        self.engine.feed(remaining)
                    final = self.engine.finalize()
            except Exception as e:
                print(f"[live] finalize error (ignored): {e!r}", flush=True)
            if final:
                try:
                    self.typer.update(final)
                    self.last_item.title = f"Last: {(final[:40] + '…') if len(final) > 40 else final}"
                    print(f"[live] -> \"{final}\"", flush=True)
                except Exception as e:
                    print(f"[live] type error (ignored): {e!r}", flush=True)
            try:
                self.typer.reset()
            except Exception:
                pass
        finally:
            # 4) ALWAYS restore audio + return to idle, no matter what happened above.
            if self.duck_enabled and self._saved_vol is not None:
                try:
                    _fade_volume(self._saved_vol * DUCK_TO_PERCENT, self._saved_vol, duration=0.8)
                except Exception:
                    pass
                self._saved_vol = None
            with self._lock:
                self._status = "idle"

    # ---- quit ----
    def _quit(self, _):
        try:
            with self._lock:
                self._status = "idle"
            self.mic.stop()
        except Exception:
            pass
        if self._saved_vol is not None:   # never leave audio ducked
            _set_output_volume(self._saved_vol)
        try:
            self.indicator.hide()
        except Exception:
            pass
        try:
            self._listener.stop()
        except Exception:
            pass
        print("[live] clean shutdown — mic released.", flush=True)
        rumps.quit_application()
        os._exit(0)


def selftest(wav, device, lang, engine_cls):
    """Headless integration check: stream a wav through engine+typer DIFF (no keyboard,
    no mic). Confirms growing text correct and backspaces (~0)."""
    import soundfile as sf
    audio, sr = sf.read(wav, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != MODEL_SR:
        audio = _resample_linear(audio, sr, MODEL_SR)
    eng = engine_cls(device=device, lang=lang)
    eng.reset()
    committed, total_back = "", 0
    for i in range(0, len(audio), CHUNK_SAMPLES):
        text = eng.feed(audio[i:i + CHUNK_SAMPLES])
        nb, _ = diff_plan(committed, text)
        total_back += nb
        committed = text
    final = eng.finalize()
    nb, _ = diff_plan(committed, final)
    total_back += nb
    print(f"[selftest] FINAL: {final!r}")
    print(f"[selftest] total backspaces: {total_back} "
          f"({'pure append ✅' if total_back == 0 else 'some revision'})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="onnx", choices=["onnx", "nemo"],
                    help="onnx = ~2s load, CPU, frees the GPU (default); nemo = MPS, best accuracy")
    ap.add_argument("--trigger", default="f18",
                    help="'doubletap-rcmd' (double-tap Right-⌘, no Karabiner) or a pynput Key "
                         "name like f18 (default, from Karabiner)")
    ap.add_argument("--device", default=None, choices=["mps", "cpu"], help="override the engine's default device")
    ap.add_argument("--lang", default="auto")
    ap.add_argument("--no-duck", action="store_true", help="disable audio ducking")
    ap.add_argument("--selftest", metavar="WAV", help="headless engine+typer check, no GUI")
    args = ap.parse_args()

    if args.engine == "onnx":
        from onnx_engine import StreamingTranscriber as EngineCls
        device = args.device or "cpu"
    else:
        from stream_engine import StreamingTranscriber as EngineCls
        device = args.device or "mps"

    if args.selftest:
        selftest(args.selftest, device, args.lang, EngineCls)
        return

    # accept the hyphenated CLI form and normalize to the internal trigger name
    trigger = "doubletap_cmd_r" if args.trigger in ("doubletap-rcmd", "doubletap_cmd_r") else args.trigger

    LiveDictateApp(trigger=trigger, device=device, lang=args.lang,
                   duck=not args.no_duck, engine_cls=EngineCls).run()


if __name__ == "__main__":
    main()
