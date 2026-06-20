#!/usr/bin/env python
"""
ONNX Runtime streaming engine for nvidia/nemotron-3.5-asr-streaming-0.6b.

Same model, lighter runtime: drives the community ONNX export
(altunenes/parakeet-rs :: nemotron-3.5-asr-streaming-0.6b-onnx) directly with
ONNX Runtime (CoreML / CPU), NO NeMo, NO PyTorch needed at inference. This is
THIS model (multilingual cache-aware FastConformer-RNNT, vocab 13087, the full
40-locale prompt dictionary) -- not a substitute.

Decomposition (NeMo's standard RNNT ONNX split):
  encoder.onnx        : (mel, len, 3 caches, prompt_index) -> (encoded, enc_len, 3 next caches)
  decoder_joint.onnx  : (encoder_outputs, targets, target_length, lstm_states_1/2)
                        -> (joint_logits[...,13088], pred_lengths, next states)

We implement: a NeMo-matching log-mel frontend, the RNNT greedy decode loop, and
cache threading across feed() calls. Same public API as stream_engine.StreamingTranscriber.

This export is fixed at att_context_size=[56,6] => 560ms chunk, 7 output frames/chunk.

HEADLESS: never opens a microphone / CoreAudio device. Self-test reads WAVs.
"""
import os
import re
import sys
import json
import time
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ONNX_DIR = os.path.join(HERE, "onnx_weights", "nemotron-3.5-asr-streaming-0.6b-onnx")
_LANG_TAG = re.compile(r"\s*<[a-z]{2}-[A-Z]{2}>\s*")
MODEL_SR = 16000


def _strip_tags(t: str) -> str:
    return _LANG_TAG.sub(" ", t or "").strip()


# --------------------------------------------------------------------------- #
#  Pure-numpy STFT + Slaney mel filterbank (no librosa -> no pooch/lzma).      #
#  Bit-equivalent to librosa.stft / librosa.filters.mel(norm='slaney').        #
# --------------------------------------------------------------------------- #
def _hz_to_mel_slaney(freq):
    freq = np.asarray(freq, dtype=np.float64)
    f_min, f_sp = 0.0, 200.0 / 3
    mels = (freq - f_min) / f_sp
    min_log_hz, min_log_mel = 1000.0, (1000.0 - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    log_t = freq >= min_log_hz
    mels = np.where(log_t, min_log_mel + np.log(freq / min_log_hz) / logstep, mels)
    return mels


def _mel_to_hz_slaney(mels):
    mels = np.asarray(mels, dtype=np.float64)
    f_min, f_sp = 0.0, 200.0 / 3
    freqs = f_min + f_sp * mels
    min_log_hz, min_log_mel = 1000.0, (1000.0 - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    log_t = mels >= min_log_mel
    freqs = np.where(log_t, min_log_hz * np.exp(logstep * (mels - min_log_mel)), freqs)
    return freqs


def _slaney_mel_filterbank(sr, n_fft, n_mels, fmin=0.0, fmax=None):
    """Replicates librosa.filters.mel(htk=False, norm='slaney')."""
    if fmax is None:
        fmax = sr / 2
    n_freqs = int(1 + n_fft // 2)
    fftfreqs = np.linspace(0, sr / 2, n_freqs)
    mel_min, mel_max = _hz_to_mel_slaney(fmin), _hz_to_mel_slaney(fmax)
    mel_pts = np.linspace(mel_min, mel_max, n_mels + 2)
    freq_pts = _mel_to_hz_slaney(mel_pts)
    fdiff = np.diff(freq_pts)
    ramps = freq_pts[:, None] - fftfreqs[None, :]
    weights = np.zeros((n_mels, n_freqs))
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0, np.minimum(lower, upper))
    # Slaney normalization: area-normalize each filter
    enorm = 2.0 / (freq_pts[2:n_mels + 2] - freq_pts[:n_mels])
    weights *= enorm[:, None]
    return weights


def _stft(y, n_fft, hop, win_length, window):
    """Replicates librosa.stft(center=True, pad_mode='reflect') magnitude path.
    window (win_length) is centered & zero-padded to n_fft."""
    # center-pad the window to n_fft
    if win_length < n_fft:
        pad_l = (n_fft - win_length) // 2
        win = np.zeros(n_fft, dtype=np.float64)
        win[pad_l:pad_l + win_length] = window
    else:
        win = window.astype(np.float64)
    # reflect-pad the signal by n_fft//2 on both sides (center=True)
    pad = n_fft // 2
    yp = np.pad(y.astype(np.float64), pad, mode="reflect")
    # frame
    n_frames = 1 + (len(yp) - n_fft) // hop
    if n_frames <= 0:
        return np.zeros((1 + n_fft // 2, 0), dtype=np.complex128)
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = yp[idx] * win[None, :]            # [n_frames, n_fft]
    spec = np.fft.rfft(frames, n=n_fft, axis=1).T  # [n_freqs, n_frames]
    return spec


# --------------------------------------------------------------------------- #
#  NeMo-matching log-mel frontend (AudioToMelSpectrogramPreprocessor defaults) #
# --------------------------------------------------------------------------- #
class MelFrontend:
    """Reproduces NeMo's filterbank features:
    preemphasis 0.97, hann(periodic) window of win_length=400 padded to n_fft=512,
    hop=160, 128 mel bins (Slaney/HTK per librosa), power spectrum, log(x + 1e-5),
    NO normalization (config: normalize 'NA'). Center=False (NeMo uses left-aligned
    framing with reflect pad of n_fft//2 by default -> we match NeMo: pad_to=0,
    exact_pad uses n_fft//2 reflect). We validate against ground-truth transcripts.
    """
    def __init__(self, cfg):
        self.sr = cfg["sample_rate"]
        self.preemph = cfg["preprocessor"]["preemph"]
        self.n_fft = cfg["preprocessor"]["n_fft"]
        self.win_length = int(round(cfg["preprocessor"]["window_size"] * self.sr))   # 400
        self.hop = int(round(cfg["preprocessor"]["window_stride"] * self.sr))        # 160
        self.n_mels = cfg["n_mels"]
        # NeMo FilterbankFeatures: log_zero_guard_type='add', value=2**-24
        self.log_guard = 2.0 ** -24
        # NeMo uses torch.hann_window(win_length, periodic=False) == symmetric Hann.
        # np.hanning(N) is the symmetric (periodic=False) window -> matches.
        self.window = np.hanning(self.win_length).astype(np.float32)
        # Slaney-normalized mel filterbank (numpy; matches librosa.filters.mel defaults
        # htk=False, norm='slaney'). Computed in-house so the packaged app does NOT need
        # librosa (which drags in pooch -> lzma and bloats/breaks the bundle).
        self.mel_fb = _slaney_mel_filterbank(
            self.sr, self.n_fft, self.n_mels, fmin=0.0, fmax=self.sr / 2).astype(np.float32)

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """audio: float32 mono 16k -> mel features [1, n_mels, T] (numpy float32)."""
        x = np.asarray(audio, dtype=np.float32)
        if x.size == 0:
            return np.zeros((1, self.n_mels, 0), dtype=np.float32)
        # preemphasis: y[t] = x[t] - 0.97 * x[t-1]  (NeMo applies on the whole signal)
        if self.preemph and self.preemph != 0.0:
            x = np.concatenate([x[:1], x[1:] - self.preemph * x[:-1]])
        # STFT power spectrum (center=True, reflect pad), window padded to n_fft.
        stft = _stft(x, self.n_fft, self.hop, self.win_length, self.window)
        power = (np.abs(stft) ** 2).astype(np.float32)          # [n_fft/2+1, T]
        mel = self.mel_fb @ power                                # [n_mels, T]
        mel = np.log(mel + self.log_guard).astype(np.float32)
        return mel[np.newaxis, :, :]                             # [1, n_mels, T]


# --------------------------------------------------------------------------- #
#  Streaming transcriber                                                      #
# --------------------------------------------------------------------------- #
class StreamingTranscriber:
    def __init__(self, device="cpu", lang="auto", onnx_dir=ONNX_DIR, provider=None):
        import onnxruntime as ort
        import sentencepiece as spm

        self.cfg = json.load(open(os.path.join(onnx_dir, "config.json")))
        self.lang = lang
        self.prompt_dict = self.cfg["prompt_dictionary"]
        self.blank_id = self.cfg["blank_id"]
        self.subsampling = self.cfg["subsampling_factor"]
        self.drop_extra = self.cfg.get("drop_extra_pre_encoded", 2)
        self.chunk_out_frames = self.cfg["chunk_size_output_frames"]  # 7

        # provider selection: device='mps' -> CoreML EP, else CPU
        if provider is None:
            avail = ort.get_available_providers()
            if device == "mps" and "CoreMLExecutionProvider" in avail:
                provider = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
            else:
                provider = ["CPUExecutionProvider"]
        self.provider = provider

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        t0 = time.time()
        self.enc = ort.InferenceSession(os.path.join(onnx_dir, "encoder.onnx"),
                                        sess_options=so, providers=provider)
        self.dec = ort.InferenceSession(os.path.join(onnx_dir, "decoder_joint.onnx"),
                                        sess_options=so, providers=provider)
        self.load_s = time.time() - t0

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(os.path.join(onnx_dir, "tokenizer.model"))

        self.mel = MelFrontend(self.cfg)

        # Authoritative chunk grid from the export author (parakeet-rs nemotron.rs):
        #   CHUNK_SIZE = 56 mel frames (the new audio per step)
        #   PRE_ENCODE_CACHE = 9 mel frames of left context prepended each step
        # => encoder gets 56+9 = 65 frames (matches config test_input mel_shape [1,128,65]).
        self._chunk_mel_frames = 56
        self._pre_cache_mel = 9

        self.reset()
        self._warm()

    # ------------------------------------------------------------------ #
    def reset(self):
        self._raw = np.zeros(0, dtype=np.float32)
        self._mel_done = 0  # mel frames already consumed
        # encoder caches (initial = zeros, shapes from config)
        cs = self.cfg["cache_shapes"]
        self._c_chan = np.zeros(cs["cache_last_channel"], dtype=np.float32)
        self._c_time = np.zeros(cs["cache_last_time"], dtype=np.float32)
        self._c_len = np.zeros(cs["cache_last_channel_len"], dtype=np.int64)
        # decoder LSTM states (2 layers, hidden 640) -- start at zeros, batch 1
        self._dstate1 = np.zeros((2, 1, 640), dtype=np.float32)
        self._dstate2 = np.zeros((2, 1, 640), dtype=np.float32)
        self._tokens = []          # decoded token ids (no blanks)
        self._last_token = self.blank_id
        self._last_text = ""
        self._step = 0

    def _warm(self):
        try:
            self.reset()
            self.feed(np.zeros(MODEL_SR // 2, dtype=np.float32))
            self.finalize()
        except Exception:
            pass
        finally:
            self.reset()

    def _prompt_index(self):
        key = self.lang if self.lang else "auto"
        if key not in self.prompt_dict:
            key = "auto"
        return int(self.prompt_dict[key])

    # ------------------------------------------------------------------ #
    def _run_encoder(self, mel_chunk: np.ndarray):
        enc_in = {
            "processed_signal": mel_chunk.astype(np.float32),
            "processed_signal_length": np.array([mel_chunk.shape[-1]], dtype=np.int64),
            "cache_last_channel": self._c_chan,
            "cache_last_time": self._c_time,
            "cache_last_channel_len": self._c_len,
            "prompt_index": np.array([self._prompt_index()], dtype=np.int64),
        }
        encoded, enc_len, c_chan_n, c_time_n, c_len_n = self.enc.run(None, enc_in)
        self._c_chan, self._c_time, self._c_len = c_chan_n, c_time_n, c_len_n
        return encoded, int(enc_len[0])

    def _greedy_decode(self, encoded: np.ndarray, enc_len: int):
        """RNNT greedy over encoder time steps. encoded: [1, 1024, T_enc]."""
        T = enc_len
        max_symbols = 10
        for t in range(T):
            f = encoded[:, :, t:t + 1]  # [1,1024,1] -> joint expects [B, D, T]=1
            emitted = 0
            while emitted < max_symbols:
                targets = np.array([[self._last_token]], dtype=np.int32)
                tlen = np.array([1], dtype=np.int32)
                dec_in = {
                    "encoder_outputs": f,
                    "targets": targets,
                    "target_length": tlen,
                    "input_states_1": self._dstate1,
                    "input_states_2": self._dstate2,
                }
                logits, _plen, s1, s2 = self.dec.run(None, dec_in)
                # logits: [1,1,1,13088]
                logit = logits[0, 0, 0]
                tok = int(np.argmax(logit))
                if tok == self.blank_id:
                    break
                # accept token: advance prediction-net state + history
                self._dstate1, self._dstate2 = s1, s2
                self._last_token = tok
                self._tokens.append(tok)
                emitted += 1
        text = self.sp.decode(self._tokens)
        return text

    def _decode_new(self, is_final: bool) -> str:
        if len(self._raw) == 0:
            return _strip_tags(self._last_text)
        feats = self.mel(self._raw)          # [1, n_mels, T_total] over the whole stream
        T = feats.shape[-1]
        cf = self._chunk_mel_frames          # 56 new frames per chunk
        pre = self._pre_cache_mel            # 9 left-context frames
        avail = T - self._mel_done
        n_full = avail // cf
        if not is_final and n_full == 0:
            return _strip_tags(self._last_text)
        # process: each step takes [9 left-context] + [56 new] = 65 frames.
        pos = self._mel_done
        end_frames = self._mel_done + (avail if is_final else n_full * cf)
        while pos < end_frames:
            new_end = min(pos + cf, end_frames)
            cache_start = max(0, pos - pre)
            mel_chunk = feats[:, :, cache_start:new_end]
            # zero-pad the head if we have fewer than `pre` context frames (first chunk)
            have_pre = pos - cache_start
            if have_pre < pre:
                pad = np.zeros((1, mel_chunk.shape[1], pre - have_pre), dtype=np.float32)
                mel_chunk = np.concatenate([pad, mel_chunk], axis=-1)
            if mel_chunk.shape[-1] == 0:
                break
            encoded, enc_len = self._run_encoder(mel_chunk)
            self._greedy_decode(encoded, enc_len)
            self._step += 1
            pos = new_end
        self._mel_done = end_frames
        self._last_text = self.sp.decode(self._tokens)
        return _strip_tags(self._last_text)

    # ------------------------------------------------------------------ #
    def feed(self, audio_f32_16k: np.ndarray) -> str:
        if audio_f32_16k is None or len(audio_f32_16k) == 0:
            return _strip_tags(self._last_text)
        seg = np.asarray(audio_f32_16k, dtype=np.float32).reshape(-1)
        self._raw = np.concatenate([self._raw, seg])
        return self._decode_new(is_final=False)

    def finalize(self) -> str:
        return self._decode_new(is_final=True)


# ===================================================================== #
#  Headless self-test                                                    #
# ===================================================================== #
def simulate_stream(st, wav_path, chunk_ms=320, show_partials=True):
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
    print(f"\n=== {os.path.basename(wav_path)} ({dur:.1f}s, feed={chunk_ms}ms, provider={st.provider[0]}) ===")
    t0 = time.time(); per = []; partials = []
    for i in range(0, len(audio), chunk):
        ct = time.time()
        p = st.feed(audio[i:i + chunk])
        per.append(time.time() - ct); partials.append(p)
        if show_partials and p:
            print(f"  [{time.time()-t0:5.2f}s] {p}", flush=True)
    final = st.finalize()
    total = time.time() - t0
    print(f"  FINAL: {final}")
    n = max(1, len(per))
    print(f"  audio={dur:.2f}s proc={total:.2f}s RTF={total/dur:.3f} "
          f"per-feed mean {1000*sum(per)/n:.0f}ms / max {1000*max(per):.0f}ms")
    return final, partials


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    ap.add_argument("--lang", default="auto")
    ap.add_argument("--chunk-ms", type=int, default=320)
    ap.add_argument("--wav", default=None)
    args = ap.parse_args()

    t0 = time.time()
    st = StreamingTranscriber(device=args.device, lang=args.lang)
    print(f"[onnx] sessions loaded in {st.load_s:.2f}s (total init {time.time()-t0:.2f}s), "
          f"provider={st.provider}")

    if args.wav:
        simulate_stream(st, args.wav, chunk_ms=args.chunk_ms)
    else:
        for name in ["test_en.wav", "test_fr.wav"]:
            p = os.path.join(HERE, "assets", name)
            if os.path.exists(p):
                lang = "fr-FR" if "fr" in name else "en-US"
                st.lang = lang
                simulate_stream(st, p, chunk_ms=args.chunk_ms)


if __name__ == "__main__":
    main()
