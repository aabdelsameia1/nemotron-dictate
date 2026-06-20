#!/usr/bin/env python
"""
Nemotron Dictation — a tiny menu-bar app that replaces Apple dictation with
nvidia/nemotron-3.5-asr-streaming-0.6b, running fully local on Apple Silicon.

UX (the "mic button" flow):
    Tap your mic key  -> 🔴 recording   (menu-bar icon turns red)
    Tap it again      -> ✍️ transcribe  -> text is typed into the focused app
                      -> 🎤 idle again

The mic key (F5 🎤) is remapped by Karabiner to F18, and this app listens for
F18 as a TOGGLE. F18 is a "phantom" key (no physical key sends it), so it never
clashes with anything you type.

=========================  MACHINE-SAFETY (read this)  =========================
The 2026-06-08 audio-daemon disaster came from FORCE-KILLING a process mid
CoreAudio init -- NOT from the mic itself. This app is built so you NEVER have
to force-kill it:
  * Mic stream is OPENED on record-start and CLOSED on record-stop. Never held
    open while idle. Always stopped+closed in a finally block.
  * SIGINT/SIGTERM + the Quit menu item all tear the mic down cleanly, then exit.
  * The model is loaded ONCE, kept warm. One model at a time.
  * Transcription runs on a worker thread so the menu bar never freezes.
If it ever misbehaves: click Quit (or Ctrl-C in a terminal). Clean every time.
===============================================================================
"""
import os
import sys
import time
import signal
import threading
import argparse

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# Reuse the proven recorder/model/injection from the push-to-talk daemon.
from dictate import PushToTalkRecorder, Transcriber, inject_text

import rumps
from pynput import keyboard

# Icons for each state (menu-bar title text).
ICON = {
    "loading": "⏳",
    "idle": "🎤",
    "recording": "🔴",
    "transcribing": "✍️",
}

MAX_RECORD_SECONDS = 120  # safety auto-stop so the mic is never left open forever


class DictateApp(rumps.App):
    def __init__(self, trigger="f18", device="mps", lang="auto"):
        super().__init__(ICON["loading"], quit_button=None)  # custom Quit for clean teardown
        self.device = device
        self.lang = lang
        self.trigger_name = trigger

        # Shared state, mutated from listener/worker threads, read by the UI timer.
        self._status = "loading"
        self._last_text = ""
        self._lock = threading.Lock()
        self._rec_started_at = None

        self.rec = PushToTalkRecorder()
        self.trx = None  # loaded async so the menu bar shows up instantly

        # --- menu ---
        self.status_item = rumps.MenuItem("Loading model…")
        self.lang_item = rumps.MenuItem(f"Language: {self.lang}")
        self.last_item = rumps.MenuItem("Last: —")
        self.menu = [
            self.status_item,
            None,
            rumps.MenuItem("Start / Stop (manual)", callback=self._manual_toggle),
            self._lang_menu(),
            self.last_item,
            None,
            rumps.MenuItem("Quit", callback=self._quit),
        ]

        # Load model in the background.
        threading.Thread(target=self._load_model, daemon=True).start()

        # Keyboard listener (F18 toggle) on its own thread.
        self._listener = keyboard.Listener(on_press=self._on_press, suppress=False)
        self._listener.start()

        # Clean shutdown on signals (e.g. launchd stop / Ctrl-C).
        signal.signal(signal.SIGINT, lambda *_: self._quit(None))
        signal.signal(signal.SIGTERM, lambda *_: self._quit(None))

        # UI refresher: runs ON the main thread, so all title/menu edits are safe.
        self._ui_timer = rumps.Timer(self._refresh_ui, 0.15)
        self._ui_timer.start()

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

    # --------------------------------------------------------------- model --
    def _load_model(self):
        self.trx = Transcriber(device=self.device, lang=self.lang)
        with self._lock:
            self._status = "idle"

    # ------------------------------------------------------------ UI timer --
    def _refresh_ui(self, _):
        # QUIRK FIX: rumps/PyObjC's runEventLoop installs its OWN SIGINT handler on
        # run(), so the ones set in __init__ get overridden and Ctrl-C in a terminal
        # appears to do nothing. We reclaim them on the first timer tick (which runs
        # Python on the main thread), so Ctrl-C/SIGTERM now fire our clean _quit
        # within one tick (~0.15s). Menu "Quit" and `launchctl unload` also work.
        if not getattr(self, "_sig_reasserted", False):
            signal.signal(signal.SIGINT, lambda *_: self._quit(None))
            signal.signal(signal.SIGTERM, lambda *_: self._quit(None))
            self._sig_reasserted = True
        with self._lock:
            st = self._status
        self.title = ICON.get(st, "🎤")
        labels = {
            "loading": "Loading model…",
            "idle": f"Ready — tap your mic key ({self.trigger_name.upper()})",
            "recording": "🔴 Recording… tap mic key to stop",
            "transcribing": "✍️ Transcribing…",
        }
        self.status_item.title = labels.get(st, "Ready")
        if self._last_text:
            preview = (self._last_text[:40] + "…") if len(self._last_text) > 40 else self._last_text
            self.last_item.title = f"Last: {preview}"
        # Safety auto-stop.
        if st == "recording" and self._rec_started_at:
            if time.time() - self._rec_started_at > MAX_RECORD_SECONDS:
                threading.Thread(target=self._stop_and_transcribe, daemon=True).start()

    # -------------------------------------------------------------- hotkey --
    def _on_press(self, key):
        if key == getattr(keyboard.Key, self.trigger_name, None):
            self._toggle()

    def _manual_toggle(self, _):
        self._toggle()

    def _toggle(self):
        with self._lock:
            st = self._status
        if st == "idle":
            self._start()
        elif st == "recording":
            threading.Thread(target=self._stop_and_transcribe, daemon=True).start()
        # ignore taps while loading/transcribing

    def _start(self):
        try:
            self.rec.start()
        except Exception as e:
            print(f"[dictate] mic open failed: {e!r}", flush=True)
            return
        with self._lock:
            self._status = "recording"
            self._rec_started_at = time.time()

    def _stop_and_transcribe(self):
        audio = None
        try:
            audio = self.rec.stop()  # closes mic cleanly
        except Exception as e:
            print(f"[dictate] mic close issue: {e!r}", flush=True)
        with self._lock:
            self._status = "transcribing"
            self._rec_started_at = None
        text = ""
        if audio is not None and self.trx is not None:
            try:
                text = self.trx.transcribe_audio(audio)
            except Exception as e:
                print(f"[dictate] transcribe error: {e!r}", flush=True)
        if text:
            with self._lock:
                self._last_text = text
            inject_text(text)
            print(f"[dictate] -> \"{text}\"", flush=True)
        with self._lock:
            self._status = "idle"

    # --------------------------------------------------------------- quit ---
    def _quit(self, _):
        try:
            self.rec.stop()
        except Exception:
            pass
        try:
            self._listener.stop()
        except Exception:
            pass
        print("[dictate] clean shutdown — mic released.", flush=True)
        rumps.quit_application()
        os._exit(0)


def selftest(wav, device, lang):
    """Prove the pipeline with no GUI/keyboard: transcribe a file and print it."""
    trx = Transcriber(device=device, lang=lang)
    t0 = time.time()
    text = trx._run(wav)
    import re
    text = re.sub(r"\s*<[a-z]{2}-[A-Z]{2}>\s*", " ", text).strip()
    print(f"[selftest] {time.time()-t0:.2f}s -> {text!r}")
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trigger", default="f18", help="pynput Key name (default f18, fed by Karabiner from the mic key)")
    ap.add_argument("--device", default="mps", choices=["mps", "cpu"])
    ap.add_argument("--lang", default="auto", help="auto | en-US | fr-FR | …")
    ap.add_argument("--selftest", metavar="WAV", help="transcribe a wav and exit (no GUI)")
    args = ap.parse_args()

    if args.selftest:
        selftest(args.selftest, args.device, args.lang)
        return

    DictateApp(trigger=args.trigger, device=args.device, lang=args.lang).run()


if __name__ == "__main__":
    main()
