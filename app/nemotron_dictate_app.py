#!/usr/bin/env python
"""
Nemotron Dictate — PACKAGED app launcher (the FULL live experience).

This wraps live_dictate.LiveDictateApp (the real UX Abdallah uses in the terminal):
  * live streaming transcription (words appear as you speak),
  * the Dynamic-Island-style glass pill under the notch (FloatingIndicator),
  * audio ducking (gradual fade of other apps' volume while recording),
  * pause/resume, language + duck settings persisted.

It is ONNX-only (onnx_engine, CPU) and adds the packaging concerns that the dev
script doesn't need:
  * default hotkey = DOUBLE-TAP RIGHT-COMMAND (no Karabiner),
  * first-run model download (~2.4 GB) into Application Support,
  * native libs imported on the MAIN thread (dyld-deadlock safety in a bundle),
  * the bundled PortAudio dylib made loadable (py2app zips package data).

Run the FULL app: just open it. Headless build check: `--selftest <wav>`.
"""
import os
import sys
import time
import types
import threading

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")

# Make onnx_engine / live_dictate / live_inject importable from source OR bundle.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

APP_NAME = "Nemotron Dictate"
HF_REPO = "altunenes/parakeet-rs"
HF_SUBDIR = "nemotron-3.5-asr-streaming-0.6b-onnx"
MODEL_FILES = ["config.json", "encoder.onnx", "encoder.onnx.data",
               "decoder_joint.onnx", "tokenizer.model"]


# --------------------------------------------------------------------------- #
#  Paths / logging                                                             #
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
#  Bundled PortAudio: make `import sounddevice` find the unzipped dylib.        #
# --------------------------------------------------------------------------- #
def _resource_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    res = os.path.normpath(os.path.join(here, "..", "Resources"))
    return res if os.path.isdir(res) else here


def _ensure_portaudio_loadable():
    cand = os.path.join(_resource_dir(), "portaudio-binaries", "libportaudio.dylib")
    if not os.path.exists(cand) or "_sounddevice_data" in sys.modules:
        return
    mod = types.ModuleType("_sounddevice_data")
    mod.__path__ = [_resource_dir()]  # so __path__/portaudio-binaries/libportaudio.dylib resolves
    sys.modules["_sounddevice_data"] = mod


# --------------------------------------------------------------------------- #
#  First-run model download (real files into Application Support, no symlinks). #
# --------------------------------------------------------------------------- #
def download_model(progress_cb=None):
    from huggingface_hub import hf_hub_download
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    d = model_dir()
    os.makedirs(d, exist_ok=True)
    import shutil
    for i, fn in enumerate(MODEL_FILES):
        if progress_cb:
            progress_cb(f"Downloading {fn} ({i + 1}/{len(MODEL_FILES)})…")
        _log(f"download {fn}")
        path = hf_hub_download(repo_id=HF_REPO, filename=f"{HF_SUBDIR}/{fn}",
                               local_dir=os.path.join(d, "_dl"))
        dst = os.path.join(d, fn)
        if os.path.abspath(path) != os.path.abspath(dst):
            shutil.copy2(path, dst)  # real file (macOS TCC blocks open() on cross-dir symlinks)
    return model_present()


# --------------------------------------------------------------------------- #
#  Engine factory: onnx_engine.StreamingTranscriber pinned to App Support dir.  #
#  live_dictate calls engine_cls(device=, lang=); we inject onnx_dir.           #
# --------------------------------------------------------------------------- #
def _make_engine_cls():
    from onnx_engine import StreamingTranscriber as _Base

    def factory(device="cpu", lang="auto"):
        return _Base(device=device, lang=lang, onnx_dir=model_dir())

    return factory


# --------------------------------------------------------------------------- #
#  Packaged subclass of the FULL LiveDictateApp.                               #
#  Keeps pill + ducking + streaming; adds first-run download + main-thread load.#
# --------------------------------------------------------------------------- #
def _build_packaged_app_class():
    import rumps
    import live_dictate as ld

    class PackagedLiveDictateApp(ld.LiveDictateApp):
        """LiveDictateApp (pill + ducking + streaming) with packaging hardening:
        - model download on first run (worker thread; UI responsive),
        - model LOADED ON THE MAIN THREAD (native-init safety inside a py2app bundle).
        The base class kicks off `_load` on a daemon thread in its __init__; we
        suppress that and drive the load from a main-thread one-shot timer instead."""

        def __init__(self, **kw):
            self._pkg_allow_base_load = False  # gate base _load (see _load below)
            self._pkg_loaded = False
            self._pkg_download_done = model_present()
            super().__init__(**kw)
            self._pkg_timer = rumps.Timer(self._pkg_tick, 0.5)
            self._pkg_timer.start()
            if not self._pkg_download_done:
                threading.Thread(target=self._pkg_download, daemon=True).start()

        def _load(self):
            # Base __init__ starts this on a daemon thread; suppress until our
            # main-thread path explicitly enables it, so native init stays on main.
            if not self._pkg_allow_base_load:
                return
            super()._load()

        def _pkg_download(self):
            try:
                _log("first run: downloading model (~2.4 GB)…")
                with self._lock:
                    self._status = "loading"
                ok = download_model(progress_cb=lambda m: _log(m))
                if not ok:
                    _log("download incomplete — check network, reopen app.")
                    return
                self._pkg_download_done = True
            except Exception as e:
                _log(f"download failed: {e!r}")

        def _pkg_tick(self, _):
            if self._pkg_loaded or not self._pkg_download_done:
                return  # wait for the download (if any) to finish
            self._pkg_loaded = True
            self._pkg_timer.stop()
            self._pkg_allow_base_load = True
            _log("loading model on main thread…")
            try:
                super()._load()  # builds engine via the injected onnx_dir factory
                _log("model ready")
            except Exception as e:
                _log(f"model load failed: {e!r}")

    return PackagedLiveDictateApp


# --------------------------------------------------------------------------- #
#  Entry                                                                       #
# --------------------------------------------------------------------------- #
def _preimport_heavy():
    """Import native extensions on the MAIN thread before the event loop, to avoid
    dyld/ObjC deadlocks importing them off-thread during macOS app init."""
    try:
        import onnxruntime  # noqa
        import sentencepiece  # noqa
        import soundfile  # noqa
        import onnx_engine  # noqa
    except Exception as e:
        _log(f"pre-import warning: {e!r}")


def main():
    # headless build verification: stream a WAV through the SAME engine path. No mic.
    if len(sys.argv) >= 3 and sys.argv[1] == "--selftest":
        wav = sys.argv[2]
        _ensure_portaudio_loadable()
        if model_present():
            EngineCls = _make_engine_cls()
        else:
            from onnx_engine import StreamingTranscriber as Raw
            def EngineCls(device="cpu", lang="en-US"):
                return Raw(device=device, lang=lang)
        import live_dictate as ld
        lang = "fr-FR" if "fr" in os.path.basename(wav).lower() else "en-US"
        ld.selftest(wav, "cpu", lang, EngineCls)
        return

    _ensure_portaudio_loadable()
    _preimport_heavy()

    AppCls = _build_packaged_app_class()
    EngineCls = _make_engine_cls()
    # Default packaged hotkey: double-tap Right-Command (no Karabiner). Engine = ONNX/CPU.
    AppCls(trigger="doubletap_cmd_r", device="cpu", lang="auto",
           duck=True, engine_cls=EngineCls).run()


if __name__ == "__main__":
    main()
