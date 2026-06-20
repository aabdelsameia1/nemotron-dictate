#!/usr/bin/env python
"""
Canonical inference wrapper for nvidia/nemotron-3.5-asr-streaming-0.6b on Apple Silicon.

Model: FastConformer-CacheAware-RNNT, 600M params, 40 language-locales.
Runs on MPS (Metal) or CPU. CUDA not required.

Usage:
    python run.py --audio assets/test_en.wav --lang en-US
    python run.py --audio file.wav --lang fr-FR --device cpu

Machine-safety: we NEVER open a CoreAudio output device. We only read audio
files from disk. As a defensive guard, `sounddevice` is stubbed before any
NeMo import so that even if a transitive dep tries to touch PortAudio, it
cannot open an audio device on Abdallah's machine.
"""
import os
import sys
import time
import json
import argparse
from pathlib import Path

# ---- Machine-safety guard: stub sounddevice BEFORE any heavy import ----
# We only ever read audio files. Never let any lib open a CoreAudio device.
import types as _types
_sd_stub = _types.ModuleType("sounddevice")
def _blocked(*a, **k):
    raise RuntimeError("sounddevice is stubbed (machine-safety): no audio device access allowed")
_sd_stub.play = _blocked
_sd_stub.rec = _blocked
_sd_stub.OutputStream = _blocked
_sd_stub.InputStream = _blocked
_sd_stub.Stream = _blocked
sys.modules.setdefault("sounddevice", _sd_stub)

# ---- MPS fallback: NeMo/PyTorch ops not yet on Metal silently fall to CPU ----
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# Keep things quiet & deterministic-ish for benchmarking.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import torch
import psutil


def pick_device(requested: str) -> str:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return requested


def load_model(model_name: str, device: str):
    """Load the NeMo ASR model and move it to the target device."""
    import nemo.collections.asr as nemo_asr  # heavy import, after stub + env
    t0 = time.time()
    model = nemo_asr.models.ASRModel.from_pretrained(model_name).eval()
    load_s = time.time() - t0
    # Move to device. MPS path; CPU is the no-op default in NeMo.
    if device == "mps":
        model = model.to(torch.device("mps"))
    elif device == "cpu":
        model = model.to(torch.device("cpu"))
    return model, load_s


def _audio_duration(path: str) -> float:
    import soundfile as sf
    info = sf.info(path)
    return info.frames / float(info.samplerate)


def transcribe(model, audio_paths, target_lang: str = "en-US", batch_size: int = 1):
    """Transcribe a list of audio files.

    The language-locale must reach each Lhotse cut's supervision via the manifest
    `lang` field: the prompt-conditioned model looks up cut.supervisions[0].language
    to pick the language prompt index. Bare path strings produce cuts with
    language=None, which crashes the prompt-index lookup on NeMo main. So we write
    a NeMo manifest (.json) that includes `lang` per entry and pass its path --
    NeMo reads it back verbatim, preserving the language field. We also pass
    target_lang so the dataloader's default prompt id matches.
    """
    import tempfile
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8")
    with tmp as fp:
        for p in audio_paths:
            entry = {
                "audio_filepath": os.path.abspath(p),
                "duration": _audio_duration(p),
                "text": "",
                "lang": target_lang,
            }
            fp.write(json.dumps(entry) + "\n")
    manifest_path = tmp.name
    t0 = time.time()
    with torch.inference_mode():
        out = model.transcribe([manifest_path], batch_size=batch_size, target_lang=target_lang)
    dt = time.time() - t0
    os.unlink(manifest_path)
    # NeMo returns a list of Hypothesis objects or plain strings depending on version.
    texts = []
    for h in out:
        if hasattr(h, "text"):
            texts.append(h.text)
        elif isinstance(h, (list, tuple)) and h and hasattr(h[0], "text"):
            texts.append(h[0].text)
        else:
            texts.append(str(h))
    return texts, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True, help="path to a 16kHz mono WAV (or any ffmpeg-readable audio)")
    ap.add_argument("--lang", default="en-US", help="target language-locale e.g. en-US, fr-FR, or 'auto'")
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    ap.add_argument("--model", default="nvidia/nemotron-3.5-asr-streaming-0.6b")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--out", default=None, help="optional path to write the transcript .txt")
    args = ap.parse_args()

    device = pick_device(args.device)
    proc = psutil.Process()
    rss0 = proc.memory_info().rss / 1e9

    print(f"[run] device={device}  model={args.model}", flush=True)
    model, load_s = load_model(args.model, device)
    print(f"[run] model loaded in {load_s:.1f}s  | target_lang={args.lang}", flush=True)

    audio = [args.audio]
    texts, infer_s = transcribe(model, audio, target_lang=args.lang, batch_size=args.batch_size)

    rss_peak = proc.memory_info().rss / 1e9
    mps_mem = None
    if device == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        mps_mem = torch.mps.current_allocated_memory() / 1e9

    print("\n=== TRANSCRIPTION ===")
    for p, t in zip(audio, texts):
        print(f"[{p}]\n{t}\n")

    print(f"[run] inference: {infer_s:.2f}s | RSS {rss0:.1f}->{rss_peak:.1f} GB"
          + (f" | MPS alloc {mps_mem:.2f} GB" if mps_mem is not None else ""))

    if args.out:
        Path(args.out).write_text("\n".join(texts), encoding="utf-8")
        print(f"[run] transcript written to {args.out}")


if __name__ == "__main__":
    main()
