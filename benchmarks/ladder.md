# Optimization Ladder — Nemotron 3.5 ASR on M4 Max

Fixed input A (short): `assets/test_en.wav` — 5.5s, English, macOS `say`.
Fixed input B (long):  `assets/test_long.wav` — 38.0s, English.
Metric = best of 3 warm iterations. RTF = wall_clock / audio_duration (lower is better).
Quality: word-accuracy vs known reference text.

## Short clip (5.5s) — dominated by fixed overhead

| # | Pass | Device | dtype | Best (s) | RTF | Peak RAM (GB) | MPS mem (GB) | Quality | Verdict |
|---|------|--------|-------|----------|-----|---------------|--------------|---------|---------|
| 0 | baseline-cpu-fp32 | CPU | fp32 | 1.202 | 0.221 | 4.13 | — | perfect | reference |
| 1 | mps-fp32 | MPS | fp32 | 0.281 | 0.052 | 1.65 | 2.59 | perfect | kept (4.3x faster) |
| 2 | mps-fp16 | MPS | fp16 | 0.274 | 0.050 | 1.66 | 2.58 | perfect | kept (best short) |
| 3 | mps-bf16 | MPS | bf16 | 0.278 | 0.051 | 1.66 | 2.58 | perfect | equal to fp16 |

## Long clip (38s) — true throughput signal

| # | Pass | Device | dtype | Best (s) | RTF | Peak RAM (GB) | MPS mem (GB) | Quality | Verdict |
|---|------|--------|-------|----------|-----|---------------|--------------|---------|---------|
| 4 | long-cpu-fp32 | CPU | fp32 | 1.65 | 0.043 | 4.54 | — | perfect | reference (23x RT) |
| 5 | long-mps-fp16 | MPS | fp16 | 0.79 | 0.021 | 1.66 | 2.58 | perfect | WINNER (48x RT, 2.1x vs CPU, 1/3 RAM) |

## Streaming (cache-aware, the model's marquee mode) — on MPS

| Pass | Device | att_context_size | Latency | Process time (38s audio) | Quality | Verdict |
|------|--------|------------------|---------|--------------------------|---------|---------|
| streaming-mps-320ms | MPS | [56,3] | 320ms | ~4s | perfect | works |
| streaming-mps-80ms  | MPS | [56,0] | 80ms (lowest) | 9.34s (RTF 0.25, real-time w/ 4x headroom) | very good (tiny degradation, as documented) | works |

## Takeaways
- MPS wins decisively on real audio: ~2x faster than CPU AND uses ~1/3 the RAM.
- fp16/bf16 ≈ fp32 in speed here (model not compute-bound at this size); fp16 chosen for the small memory edge, with zero quality loss observed.
- Lower streaming latency (80ms) costs accuracy slightly and adds per-chunk overhead — exactly the documented latency/accuracy Pareto tradeoff.
- Not pursued (low value here): 8-bit/4-bit quant (NeMo lacks a clean MPS quant path; the model is already tiny at 1.7GB MPS), torch.compile (inductor unsupported on MPS — see agent memory), ONNX export (fallback path, not needed since native NeMo works).
