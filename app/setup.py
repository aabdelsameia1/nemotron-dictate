"""
py2app build script for Nemotron Dictate (ONNX-only menu-bar dictation app).

Build:
    cd ~/Desktop/ai-models/nemotron-asr
    source .venv/bin/activate
    python app/setup.py py2app            # -> dist/Nemotron Dictate.app
    # (alias/dev mode for fast iteration):
    # python app/setup.py py2app -A

Weights are NOT bundled (download-on-first-run into Application Support), keeping
the .app small. torch / NeMo are explicitly excluded.
"""
import os
import sys

# py2app's modulegraph recurses deeply over big dep trees (scipy/librosa) and hits
# Python's default recursion cap. Raise it before the build.
sys.setrecursionlimit(20000)

from setuptools import setup

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

APP = [os.path.join(HERE, "nemotron_dictate_app.py")]

# onnx_engine.py lives at project root; include it as a top-level module so the
# bundled app can `import onnx_engine`.
sys.path.insert(0, ROOT)

# sounddevice ships libportaudio.dylib as package data; py2app would zip it and
# dylibs CANNOT be dlopen'd from inside a zip. Copy it to Contents/Resources/ (a
# real path) so the app can load it (the app points sounddevice at it on startup).
import _sounddevice_data  # noqa: E402
_PA_DIR = os.path.join(os.path.dirname(_sounddevice_data.__file__), "portaudio-binaries")

DATA_FILES = [("portaudio-binaries", [os.path.join(_PA_DIR, "libportaudio.dylib")])]

OPTIONS = {
    "argv_emulation": False,
    "iconfile": os.path.join(HERE, "AppIcon.icns"),
    # bring in onnx_engine.py as a module (it's at project root)
    "includes": [
        "onnx_engine",
        "rumps",
        "pynput", "pynput.keyboard", "pynput.mouse",
        "sounddevice", "_sounddevice_data", "cffi", "_cffi_backend",
        "sentencepiece",
        "numpy", "soundfile",
        "onnxruntime", "onnxruntime.capi", "onnxruntime.capi._pybind_state",
        "huggingface_hub",
        "objc", "Foundation", "AppKit", "Quartz", "AVFoundation", "ApplicationServices",
    ],
    "packages": [
        "onnxruntime", "sentencepiece", "soundfile",
        "huggingface_hub", "rumps", "pynput", "numpy",
    ],
    # The app uses a pure-numpy mel frontend (no librosa). Exclude librosa and its
    # transitive deps (pooch -> lzma -> liblzma dylib) which broke the bundle, plus
    # torch/NeMo and other heavy unused packages, to keep it lean + loadable.
    "excludes": [
        "torch", "torchaudio", "nemo", "nemo_toolkit", "stream_engine",
        "dictate", "live_dictate", "menubar_dictate", "transformers",
        "tensorflow", "matplotlib", "pandas", "tkinter", "lightning",
        "pytorch_lightning", "lhotse", "py2app", "PyInstaller",
        "librosa", "pooch", "scipy", "numba", "llvmlite", "audioread",
        "soxr", "lazy_loader", "joblib", "scikit_learn", "sklearn",
        "lzma", "_lzma", "bz2", "_bz2",
    ],
    "plist": {
        "CFBundleName": "Nemotron Dictate",
        "CFBundleDisplayName": "Nemotron Dictate",
        "CFBundleIdentifier": "com.abdallah.nemotrondictate",
        "CFBundleVersion": "1.0.0",
        "CFBundleShortVersionString": "1.0.0",
        "LSUIElement": True,  # menu-bar only, no dock icon
        "LSMinimumSystemVersion": "13.0",
        "NSMicrophoneUsageDescription":
            "Nemotron Dictate transcribes your speech locally. Audio never leaves your Mac.",
        "NSHumanReadableCopyright": "Local, private dictation. Model: NVIDIA Nemotron 3.5 ASR.",
        "LSApplicationCategoryType": "public.app-category.productivity",
    },
}

setup(
    app=APP,
    name="Nemotron Dictate",
    data_files=DATA_FILES,
    options={"py2app": OPTIONS},
    setup_requires=["py2app"],
)
