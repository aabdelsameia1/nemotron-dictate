#!/usr/bin/env python
"""
Standardized benchmark harness for nvidia/nemotron-3.5-asr-streaming-0.6b.

Runs a fixed audio input, warm-up + N timed iterations, records wall-clock,
RTF (real-time factor), peak RAM, peak MPS memory, and the transcription
(for quality eyeballing). Writes a JSON to benchmarks/<pass>.json.

Usage:
    python benchmark.py --pass baseline --device cpu  --dtype fp32
    python benchmark.py --pass mps      --device mps  --dtype fp32
"""
import os
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime

import types as _types
_sd_stub = _types.ModuleType("sounddevice")
def _blocked(*a, **k):
    raise RuntimeError("sounddevice is stubbed (machine-safety)")
for _n in ("play", "rec", "OutputStream", "InputStream", "Stream"):
    setattr(_sd_stub, _n, _blocked)
sys.modules.setdefault("sounddevice", _sd_stub)

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import torch
import psutil
import soundfile as sf

ROOT = Path(__file__).parent
MODEL = "nvidia/nemotron-3.5-asr-streaming-0.6b"


def audio_dur(path):
    info = sf.info(path)
    return info.frames / float(info.samplerate)


def build_manifest(audio_paths, lang, tmp_path):
    with open(tmp_path, "w", encoding="utf-8") as fp:
        for p in audio_paths:
            fp.write(json.dumps({
                "audio_filepath": os.path.abspath(p),
                "duration": audio_dur(p),
                "text": "",
                "lang": lang,
            }) + "\n")
    return tmp_path


def extract_text(out):
    texts = []
    for h in out:
        if hasattr(h, "text"):
            texts.append(h.text)
        elif isinstance(h, (list, tuple)) and h and hasattr(h[0], "text"):
            texts.append(h[0].text)
        else:
            texts.append(str(h))
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass", dest="pass_name", required=True)
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--audio", default=str(ROOT / "assets" / "test_en.wav"))
    ap.add_argument("--lang", default="en-US")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()

    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    torch_dtype = dtype_map[args.dtype]

    import nemo.collections.asr as nemo_asr

    proc = psutil.Process()
    if args.device == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()

    t0 = time.time()
    model = nemo_asr.models.ASRModel.from_pretrained(MODEL).eval()
    load_s = time.time() - t0

    dev = torch.device(args.device)
    model = model.to(dev)
    if args.dtype != "fp32":
        try:
            model = model.to(torch_dtype)
        except Exception as e:
            print(f"[bench] dtype cast to {args.dtype} failed: {e!r}; staying fp32")
            args.dtype = "fp32"

    dur = audio_dur(args.audio)
    manifest = build_manifest([args.audio], args.lang, str(ROOT / "benchmarks" / f"_tmp_{args.pass_name}.json"))

    # warm-up
    with torch.inference_mode():
        warm = model.transcribe([manifest], batch_size=args.batch_size,
                                num_workers=args.num_workers, target_lang=args.lang, verbose=False)
    warm_text = extract_text(warm)

    if args.device == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()

    # timed
    times = []
    last_text = warm_text
    for i in range(args.iters):
        t = time.time()
        with torch.inference_mode():
            out = model.transcribe([manifest], batch_size=args.batch_size,
                                   num_workers=args.num_workers, target_lang=args.lang, verbose=False)
        if args.device == "mps" and hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()
        times.append(time.time() - t)
        last_text = extract_text(out)

    peak_ram = proc.memory_info().rss / 1e9
    mps_mem = None
    if args.device == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        mps_mem = torch.mps.current_allocated_memory() / 1e9

    best = min(times)
    mean = sum(times) / len(times)
    rtf = best / dur

    out_dir = ROOT / "outputs" / f"pass-{args.pass_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "transcript.txt").write_text("\n".join(last_text), encoding="utf-8")

    result = {
        "pass": args.pass_name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "device": args.device, "dtype": args.dtype, "batch_size": args.batch_size,
            "num_workers": args.num_workers, "iters": args.iters, "audio": args.audio,
            "lang": args.lang, "audio_dur_s": round(dur, 3),
        },
        "metrics": {
            "model_load_s": round(load_s, 2),
            "wall_clock_best_s": round(best, 3),
            "wall_clock_mean_s": round(mean, 3),
            "all_times_s": [round(x, 3) for x in times],
            "rtf_best": round(rtf, 4),
            "peak_ram_gb": round(peak_ram, 2),
            "peak_mps_mem_gb": round(mps_mem, 2) if mps_mem is not None else None,
        },
        "quality": {
            "output": last_text,
            "output_path": str(out_dir / "transcript.txt"),
        },
    }

    bj = ROOT / "benchmarks" / f"{args.pass_name}.json"
    bj.write_text(json.dumps(result, indent=2), encoding="utf-8")
    os.unlink(manifest)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
