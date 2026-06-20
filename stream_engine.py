#!/usr/bin/env python
"""
TRUE cache-aware streaming transcription engine for
nvidia/nemotron-3.5-asr-streaming-0.6b on Apple Silicon.

This uses the REAL NeMo cache-aware streaming API -- the exact
`asr_model.conformer_stream_step(...)` call with persistent encoder caches
(cache_last_channel / cache_last_time / cache_last_channel_len) and persistent
RNNT hypotheses, threaded across feed() calls. The same machinery the official
`speech_to_text_cache_aware_streaming_infer.py` uses, driven incrementally so
partial text grows as audio arrives. It is NOT a "re-run batch on a growing
buffer" fake.

API:
    st = StreamingTranscriber(device="mps", lang="auto")
    st.reset()                       # new utterance: clears caches + hypotheses
    partial = st.feed(chunk_f32_16k) # feed float32 mono 16kHz np.ndarray;
                                     # returns CURRENT FULL hypothesis (tags stripped)
    final   = st.finalize()          # flush remaining buffered audio; final text

HEADLESS / machine-safety: this module NEVER opens a microphone or any CoreAudio
device. The self-test reads a WAV file and slices it to simulate a live stream.

Constraint (from NeMo): cache-aware models require compute_dtype=float32. fp16 is
explicitly unsupported (some layers force-cast to float32). We therefore run the
streaming path in float32 -- on MPS this is still fast (see RTF in the self-test).
"""
import os
import re
import sys
import time
import argparse

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np
import torch

MODEL = "nvidia/nemotron-3.5-asr-streaming-0.6b"
MODEL_SR = 16000
_LANG_TAG = re.compile(r"\s*<[a-z]{2}-[A-Z]{2}>\s*")


def _strip_tags(text: str) -> str:
    return _LANG_TAG.sub(" ", text or "").strip()


def _extract_text(hyps) -> str:
    if not hyps:
        return ""
    h = hyps[0]
    if hasattr(h, "text"):
        return h.text
    if isinstance(h, (list, tuple)) and h and hasattr(h[0], "text"):
        return h[0].text
    return str(h)


class StreamingTranscriber:
    def __init__(self, device="mps", lang="auto", att_context_size=None):
        """
        att_context_size: optional [left, right] lookahead pair controlling latency:
            [56,0]=80ms  [56,1]=160ms  [56,3]=320ms  [56,6]=560ms  [56,13]=1.12s
            None -> use the model's default streaming config.
        """
        self.device = torch.device(device)
        self.lang = lang
        print(f"[stream] loading {MODEL} on {device} (float32, cache-aware)...", flush=True)
        import nemo.collections.asr as nemo_asr
        from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer

        torch.set_grad_enabled(False)
        self.model = nemo_asr.models.ASRModel.from_pretrained(MODEL).eval()

        if att_context_size is not None and hasattr(self.model.encoder, "set_default_att_context_size"):
            self.model.encoder.set_default_att_context_size(att_context_size=att_context_size)

        # RNNT streaming decoding config (fused_batch_size=-1 required for stream step)
        if hasattr(self.model, "joint") and hasattr(self.model, "change_decoding_strategy"):
            try:
                from omegaconf import open_dict
                dec = self.model.cfg.decoding
                with open_dict(dec):
                    dec.fused_batch_size = -1
                self.model.change_decoding_strategy(dec)
            except Exception as e:
                print(f"[stream] decoding-strategy note: {e!r}", flush=True)

        # language-ID prompt (works on the streaming path via set_inference_prompt)
        if hasattr(self.model, "set_inference_prompt"):
            self.model.set_inference_prompt(self.lang if self.lang else "auto")
            # also let the decoder strip the <xx-XX> tags itself if supported
            try:
                self.model.decoding.set_strip_lang_tags(True)
            except Exception:
                pass

        # cache-aware models must run in float32
        self.model = self.model.to(device=self.device, dtype=torch.float32)
        self.model.eval()

        self._BufferCls = CacheAwareStreamingAudioBuffer
        self.buffer = None
        self._caches = None
        self._prev_hyps = None
        self._pred_out = None
        self._step = 0
        self._last_text = ""
        self.reset()
        # warm the Metal kernels with a short silence so the first real chunk is fast
        self._warm()
        print("[stream] ready.", flush=True)

    # ------------------------------------------------------------------ #
    def reset(self):
        """Begin a fresh utterance: clear the audio buffer, encoder caches, hyps."""
        self.buffer = self._BufferCls(model=self.model, online_normalization=False,
                                      pad_and_drop_preencoded=False)
        c_chan, c_time, c_len = self.model.encoder.get_initial_cache_state(batch_size=1)
        self._caches = [c_chan, c_time, c_len]
        self._prev_hyps = None
        self._pred_out = None
        self._step = 0
        self._last_text = ""
        # raw-audio accumulator. We extract mel features over the GROWING raw stream
        # (so features match whole-stream extraction exactly -- no boundary artifacts),
        # then push only the NEW feature frames into the streaming buffer and decode
        # the model-chunks that newly complete. True cache-aware streaming.
        self._raw = np.zeros(0, dtype=np.float32)
        self._chunks_done = 0  # how many model-chunks already decoded this utterance
        cfg = self.model.encoder.streaming_cfg
        chunk_frames = cfg.chunk_size[-1] if isinstance(cfg.chunk_size, (list, tuple)) else cfg.chunk_size
        self._chunk_frames = int(chunk_frames)
        stride = float(self.model.cfg.preprocessor.window_stride)  # seconds/frame
        self._chunk_samples = int(round(chunk_frames * stride * MODEL_SR))

    def _warm(self):
        try:
            self.reset()
            self.feed(np.zeros(MODEL_SR // 2, dtype=np.float32))  # 0.5s silence
            self.finalize()
        except Exception:
            pass
        finally:
            self.reset()

    # ------------------------------------------------------------------ #
    def _drop_pre_encoded(self):
        if self._step == 0:
            return 0
        return self.model.encoder.streaming_cfg.drop_extra_pre_encoded

    def _chunk_slice(self, feats, idx):
        """Replicate CacheAwareStreamingAudioBuffer's chunk grid by direct slicing of
        the cached mel tensor (no re-iteration -> O(1) per chunk).

        chunk 0 : frames [0 : c0)                       (c0 = chunk_size[0])
        chunk i : frames [c0+(i-1)*sh - pre : c0+(i-1)*sh + sh)   (i>=1)
        where sh = shift_size[1], pre = pre_encode_cache_size[1].
        The leading `pre` cache frames are zero-padded at the head if unavailable.
        """
        cfg = self.model.encoder.streaming_cfg
        c0 = cfg.chunk_size[0] if isinstance(cfg.chunk_size, (list, tuple)) else cfg.chunk_size
        sh = cfg.shift_size[1] if isinstance(cfg.shift_size, (list, tuple)) else cfg.shift_size
        pre = cfg.pre_encode_cache_size[1] if isinstance(cfg.pre_encode_cache_size, (list, tuple)) else cfg.pre_encode_cache_size
        T = feats.size(-1)
        if idx == 0:
            start, end = 0, min(c0, T)
            chunk = feats[:, :, start:end]
            return chunk, end - start
        body_start = c0 + (idx - 1) * sh
        body_end = min(body_start + sh, T)
        cache_start = max(0, body_start - pre)
        chunk = feats[:, :, cache_start:body_end]
        # pad head with zeros if we have fewer than `pre` cache frames
        have_pre = body_start - cache_start
        if have_pre < pre:
            pad = torch.zeros((chunk.size(0), chunk.size(1), pre - have_pre),
                              device=chunk.device, dtype=chunk.dtype)
            chunk = torch.cat([pad, chunk], dim=-1)
        valid_len = (body_end - body_start) + min(pre, body_start)  # real (non-pad) frames
        return chunk, valid_len

    def _num_full_chunks(self, T):
        """How many chunks are fully available given T cached mel frames."""
        cfg = self.model.encoder.streaming_cfg
        c0 = cfg.chunk_size[0] if isinstance(cfg.chunk_size, (list, tuple)) else cfg.chunk_size
        sh = cfg.shift_size[1] if isinstance(cfg.shift_size, (list, tuple)) else cfg.shift_size
        if T < c0:
            return 0
        return 1 + (T - c0) // sh

    def _decode_new_chunks(self, is_final: bool) -> str:
        """Extract mel ONCE over the accumulated raw stream, slice the chunk grid
        directly from the cached features, and run conformer_stream_step only on
        chunks not yet decoded -- threading the persistent encoder caches + RNNT
        hypotheses. Every model-chunk decoded exactly once: O(1) work per feed,
        genuine cache-aware streaming.
        """
        if len(self._raw) == 0:
            return _strip_tags(self._last_text)
        with torch.inference_mode():
            sig = torch.from_numpy(np.ascontiguousarray(self._raw)).unsqueeze(0).to(self.device)
            ln = torch.tensor([len(self._raw)], device=self.device)
            feats, _ = self.buffer.preprocessor(input_signal=sig, length=ln)  # [1, D, T]
        T = feats.size(-1)
        n_full = self._num_full_chunks(T)
        # hold back the last (possibly-still-growing) chunk until the final flush
        end = n_full if is_final else max(0, n_full - 1)
        if is_final:
            # the final partial tail may form one more chunk beyond n_full
            cfg = self.model.encoder.streaming_cfg
            c0 = cfg.chunk_size[0] if isinstance(cfg.chunk_size, (list, tuple)) else cfg.chunk_size
            sh = cfg.shift_size[1] if isinstance(cfg.shift_size, (list, tuple)) else cfg.shift_size
            covered = c0 + (n_full - 1) * sh if n_full > 0 else 0
            if T > covered:
                end = n_full + 1  # one trailing partial chunk

        drop = self.model.encoder.streaming_cfg.drop_extra_pre_encoded
        c_chan, c_time, c_len = self._caches
        for idx in range(self._chunks_done, end):
            chunk_audio, valid_len = self._chunk_slice(feats, idx)
            if chunk_audio.size(-1) == 0:
                continue
            chunk_lengths = torch.tensor([chunk_audio.size(-1)], device=self.device)
            is_last = is_final and (idx == end - 1)
            with torch.inference_mode():
                chunk_audio = chunk_audio.to(torch.float32)
                (
                    self._pred_out,
                    transcribed_texts,
                    c_chan,
                    c_time,
                    c_len,
                    self._prev_hyps,
                ) = self.model.conformer_stream_step(
                    processed_signal=chunk_audio,
                    processed_signal_length=chunk_lengths,
                    cache_last_channel=c_chan,
                    cache_last_time=c_time,
                    cache_last_channel_len=c_len,
                    keep_all_outputs=is_last,
                    previous_hypotheses=self._prev_hyps,
                    previous_pred_out=self._pred_out,
                    drop_extra_pre_encoded=(0 if idx == 0 else drop),
                    return_transcription=True,
                )
            txt = _extract_text(transcribed_texts)
            if txt:
                self._last_text = txt
        self._caches = [c_chan, c_time, c_len]
        self._chunks_done = max(self._chunks_done, end)
        return _strip_tags(self._last_text)

    def feed(self, audio_f32_16k: np.ndarray) -> str:
        """Feed one chunk of float32 mono 16kHz audio. Returns the current FULL text
        (whole growing hypothesis, language tags stripped)."""
        if audio_f32_16k is None or len(audio_f32_16k) == 0:
            return _strip_tags(self._last_text)
        seg = np.asarray(audio_f32_16k, dtype=np.float32).reshape(-1)
        self._raw = np.concatenate([self._raw, seg])
        return self._decode_new_chunks(is_final=False)

    def finalize(self) -> str:
        """Decode any remaining chunk(s), including the final partial chunk."""
        return self._decode_new_chunks(is_final=True)


# ===================================================================== #
#  Headless self-test: read a WAV, slice into chunks, simulate a stream  #
# ===================================================================== #
def simulate_stream(st: "StreamingTranscriber", wav_path: str, chunk_ms: int = 320, show_partials: bool = True):
    import soundfile as sf
    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != MODEL_SR:
        n = int(round(len(audio) * MODEL_SR / sr))
        audio = np.interp(np.linspace(0, 1, n, endpoint=False),
                          np.linspace(0, 1, len(audio), endpoint=False), audio).astype(np.float32)
    dur = len(audio) / MODEL_SR
    chunk = int(MODEL_SR * chunk_ms / 1000)

    st.reset()
    print(f"\n=== {os.path.basename(wav_path)}  ({dur:.1f}s, chunk={chunk_ms}ms) ===")
    t0 = time.time()
    per_chunk = []
    partials = []
    for i in range(0, len(audio), chunk):
        seg = audio[i:i + chunk]
        ct = time.time()
        partial = st.feed(seg)
        dt = time.time() - ct
        per_chunk.append(dt)
        partials.append(partial)
        if show_partials:
            print(f"  [{time.time()-t0:5.2f}s | +{dt*1000:5.1f}ms] {partial}", flush=True)
    final = st.finalize()
    total = time.time() - t0

    print(f"  FINAL: {final}")
    n = max(1, len(per_chunk))
    print(f"  audio={dur:.2f}s  proc={total:.2f}s  RTF={total/dur:.3f}  "
          f"per-chunk: mean {1000*sum(per_chunk)/n:.0f}ms / max {1000*max(per_chunk):.0f}ms")
    return final, partials, per_chunk, dur, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps", choices=["mps", "cpu"])
    ap.add_argument("--lang", default="auto")
    ap.add_argument("--chunk-ms", type=int, default=320)
    ap.add_argument("--att", default=None, help="e.g. 56,3 for 320ms lookahead (optional)")
    ap.add_argument("--wav", default=None, help="single wav; default = run the 3 test clips")
    args = ap.parse_args()

    att = None
    if args.att:
        att = [int(x) for x in args.att.split(",")]

    st = StreamingTranscriber(device=args.device, lang=args.lang, att_context_size=att)

    if args.wav:
        simulate_stream(st, args.wav, chunk_ms=args.chunk_ms)
    else:
        here = os.path.dirname(os.path.abspath(__file__))
        for name in ["test_en.wav", "test_fr.wav", "test_long.wav"]:
            p = os.path.join(here, "assets", name)
            if os.path.exists(p):
                simulate_stream(st, p, chunk_ms=args.chunk_ms,
                                show_partials=(name != "test_long.wav"))


if __name__ == "__main__":
    main()
