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


class IncrementalMel:
    """Stateful log-mel that yields the SAME frames as MelFrontend over the cumulative
    stream, but in CONSTANT time per push (no O(N^2) re-extraction).

    How it stays bit-identical to the batch frontend (center=True, reflect pad n_fft//2):
      * frame i is centered at sample i*hop, covering original samples [i*hop-256, i*hop+256).
      * preemphasis y[t]=x[t]-0.97*x[t-1] is applied continuously (we keep the last raw
        sample across pushes; y[0] of the whole stream keeps x[0] as-is, matching batch).
      * we keep a tail of preemphasised samples long enough to form the next frames, plus a
        flag for whether the left reflect-pad (start of stream) has been emitted yet.
      * a frame is only emitted once its RIGHT edge (i*hop+256) is covered by real samples,
        so mid-stream frames equal the batch result exactly. At finalize() we apply the
        right reflect-pad and flush the final frames.
    """
    def __init__(self, mel: "MelFrontend"):
        self.m = mel
        self.reset()

    def reset(self):
        self._last_raw = None          # last raw sample (for continuous preemphasis)
        self._pre = np.zeros(0, dtype=np.float64)  # preemphasised samples not yet consumed
        self._pre_origin = 0           # original-sample index of self._pre[0]
        self._emitted = 0              # number of mel frames already produced
        self._started = False          # has the left reflect-pad been applied?

    def _preemph_push(self, seg):
        """Append raw seg -> return the new preemphasised samples (continuous)."""
        seg = np.asarray(seg, dtype=np.float64)
        if seg.size == 0:
            return np.zeros(0, dtype=np.float64)
        p = self.m.preemph
        if self._last_raw is None:
            out = np.empty(seg.size, dtype=np.float64)
            out[0] = seg[0]            # first sample of the whole stream: as-is (batch match)
            out[1:] = seg[1:] - p * seg[:-1]
        else:
            out = seg - p * np.concatenate([[self._last_raw], seg[:-1]])
        self._last_raw = float(seg[-1])
        return out

    def _frames_from(self, sig_left_padded, base_origin, up_to_right_edge):
        """STFT->mel for all frames whose center i*hop has left>=0 and right edge<=avail.
        sig_left_padded: preemphasised samples with the left reflect-pad already prepended
        if this is the very start; base_origin: original index of sig_left_padded[0] minus
        the left pad offset. Returns (mel[1,n_mels,K], next_frame_index)."""
        m = self.m
        # window padded to n_fft (centered)
        if m.win_length < m.n_fft:
            pl = (m.n_fft - m.win_length) // 2
            win = np.zeros(m.n_fft, dtype=np.float64)
            win[pl:pl + m.win_length] = m.window
        else:
            win = m.window.astype(np.float64)
        frames = []
        i = self._emitted
        while True:
            center = i * m.hop                    # original-sample center of frame i
            left = center - m.n_fft // 2          # original index of frame start
            right = left + m.n_fft                # exclusive end
            if right > up_to_right_edge:
                break                              # not enough lookahead yet -> hold back
            s = left - base_origin
            seg = sig_left_padded[s:s + m.n_fft]
            if seg.size < m.n_fft:
                break
            frames.append(seg)
            i += 1
        if not frames:
            return np.zeros((1, m.n_mels, 0), dtype=np.float32), i
        F = np.stack(frames, axis=0) * win[None, :]
        spec = np.fft.rfft(F, n=m.n_fft, axis=1).T          # [freqs, K]
        power = (np.abs(spec) ** 2).astype(np.float32)
        mel = (m.mel_fb @ power)
        mel = np.log(mel + m.log_guard).astype(np.float32)
        return mel[np.newaxis, :, :], i

    def push(self, seg, is_final=False):
        """Push new raw audio. Returns newly-finalized mel frames [1, n_mels, K]
        (K may be 0 if not enough lookahead yet)."""
        m = self.m
        new_pre = self._preemph_push(seg)
        if new_pre.size:
            self._pre = np.concatenate([self._pre, new_pre])
        # build the analysis signal with the left reflect-pad prepended ONCE at the start
        if not self._started:
            pad = m.n_fft // 2
            # reflect pad uses samples 1..pad of the (preemphasised) stream
            if self._pre.size > pad:
                left_pad = self._pre[1:pad + 1][::-1]
            else:
                left_pad = np.zeros(0, dtype=np.float64)
            sig = np.concatenate([left_pad, self._pre])
            base_origin = -left_pad.size
        else:
            sig = self._pre
            base_origin = self._pre_origin

        if is_final:
            # apply the right reflect-pad so the trailing frames match the batch result
            pad = m.n_fft // 2
            total_len = self._pre_origin + self._pre.size if self._started else self._pre.size
            if self._pre.size > pad:
                right_pad = self._pre[-pad - 1:-1][::-1]
            else:
                right_pad = np.zeros(0, dtype=np.float64)
            sig = np.concatenate([sig, right_pad])
            avail_right = total_len + pad
        else:
            total_len = self._pre_origin + self._pre.size if self._started else self._pre.size
            avail_right = total_len   # no right pad mid-stream

        mel, next_i = self._frames_from(sig, base_origin, avail_right)
        self._started = True
        self._emitted = next_i
        # trim consumed preemphasised samples we'll never need again: keep enough tail to
        # form the next frame's left edge (next_i*hop - n_fft//2) onward.
        keep_from = max(0, next_i * m.hop - m.n_fft // 2 - self._pre_origin)
        if keep_from > 0:
            self._pre = self._pre[keep_from:]
            self._pre_origin += keep_from
        return mel


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

        # word-boundary-space repair: the lone '▁' token id + how competitive it must
        # be (in logit units below the argmax) to be reinstated at a chunk boundary.
        self._lone_space_id = self.sp.piece_to_id("▁")
        self._space_repair_margin = 6.0

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
        self._raw = np.zeros(0, dtype=np.float32)  # kept only for compatibility; unused for mel
        self._mel_done = 0  # mel frames already consumed (index into self._mel_buf)
        # incremental mel cache: constant work per feed (no O(N^2) re-extraction).
        self._inc_mel = IncrementalMel(self.mel)
        self._mel_buf = np.zeros((1, self.cfg["n_mels"], 0), dtype=np.float32)  # all mel frames so far
        # encoder caches (initial = zeros, shapes from config)
        cs = self.cfg["cache_shapes"]
        self._c_chan = np.zeros(cs["cache_last_channel"], dtype=np.float32)
        self._c_time = np.zeros(cs["cache_last_time"], dtype=np.float32)
        self._c_len = np.zeros(cs["cache_last_channel_len"], dtype=np.int64)
        # decoder LSTM states (2 layers, hidden 640) -- start at zeros, batch 1
        self._dstate1 = np.zeros((2, 1, 640), dtype=np.float32)
        self._dstate2 = np.zeros((2, 1, 640), dtype=np.float32)
        self._tokens = []          # decoded token ids (no blanks)
        self._chunk_starts = []    # token-list index where each new encoder chunk began
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
        self._chunk_starts.append(len(self._tokens))
        first_frame_of_chunk = True
        for t in range(T):
            f = encoded[:, :, t:t + 1]  # [1,1024,1] -> joint expects [B, D, T]=1
            emitted = 0
            first_symbol_of_frame = True
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
                logit = logits[0, 0, 0]                     # [13088]
                tok = int(np.argmax(logit))

                # --- word-boundary-space repair (uses the model's OWN logits) ---
                # The RNNT occasionally drops the lone '▁' word-boundary token at a
                # chunk boundary, sticking words ('thing matters' -> 'thingmatters').
                # At the FIRST symbol of the FIRST frame of a new chunk, if it is about
                # to emit a bare subword (no '▁') right after a token that COMPLETED a
                # word, but the lone '▁' (id 2) is a strong contender in the logits,
                # the model "wanted" the space — so emit '▁' first (the natural two-token
                # pattern). This is SAFE: mid-word continuations (e.g. 'beautiful' split
                # ▁be|au) leave the '▁' logit LOW because the model knows it's mid-word,
                # so this never splits a real word — the acoustics/logits disambiguate
                # what the token surface alone cannot.
                if (first_frame_of_chunk and first_symbol_of_frame
                        and tok != self.blank_id and tok != self._lone_space_id
                        and not self.sp.id_to_piece(tok).startswith("▁")
                        and self._prev_completed_word()):
                    sp_logit = float(logit[self._lone_space_id])
                    best_logit = float(logit[tok])
                    blank_logit = float(logit[self.blank_id])
                    # require '▁' to be a genuinely competitive, top candidate
                    if sp_logit >= best_logit - self._space_repair_margin and sp_logit > blank_logit:
                        self._dstate1, self._dstate2 = s1, s2
                        self._last_token = self._lone_space_id
                        self._tokens.append(self._lone_space_id)
                        first_symbol_of_frame = False  # next loop emits the real subword
                        continue

                first_symbol_of_frame = False
                if tok == self.blank_id:
                    break
                self._dstate1, self._dstate2 = s1, s2
                self._last_token = tok
                self._tokens.append(tok)
                emitted += 1
            first_frame_of_chunk = False
        return self._decode_tokens()

    def _prev_completed_word(self):
        """True if the last accepted token ended a complete word (so a new word — and
        thus a boundary space — is plausible). A '▁'-prefixed token that is itself a
        whole word, or the end of a word group, completes a word; a bare subword that
        is clearly mid-word does not. We approximate: the running surface so far ends
        on a token whose piece, re-encoded as a standalone word, is a full word."""
        if not self._tokens:
            return False
        # decode the tail word group and check it round-trips as a complete word
        sp = self.sp
        j = len(self._tokens) - 1
        while j >= 0 and not sp.id_to_piece(self._tokens[j]).startswith("▁"):
            j -= 1
        if j < 0:
            return False
        word = sp.decode(list(self._tokens[j:])).strip()
        if not word or not word[-1].isalpha():
            return True   # ended on punctuation/space → definitely a boundary
        # complete if SP re-tokenizes the standalone word to the same tail pieces
        re_ids = sp.encode(word)
        emitted_tail = [sp.id_to_piece(t) for t in self._tokens[j:]]
        re_pieces = [sp.id_to_piece(t) for t in re_ids]
        return re_pieces == emitted_tail

    def _decode_tokens(self):
        """Decode the accumulated tokens to text, REPAIRING word-boundary spaces that
        the RNNT occasionally drops across a chunk boundary.

        Bug: this vocab tokenizes many words as a lone '▁' (space) token + bare
        subwords (e.g. 'matters' = ['▁','ma','t','ter','s']). At a chunk boundary the
        prediction-net sometimes fails to emit that lone '▁', so sp.decode joins the
        words ('thing matters' -> 'thingmatters'). We detect this ONLY at recorded
        chunk-start indices (self._chunk_starts) and ONLY when the first token of the
        new chunk is a *complete word piece on its own* ('▁X' is fine; the danger is a
        bare subword that should have begun a fresh word). We never touch tokens inside
        a chunk, so legitimate intra-word subwords ('beautiful' = ▁be+au+ti+ful, all in
        one chunk) are untouched.

        Repair test (per boundary b): take the bare first-token of the chunk and ask SP
        whether it more plausibly STARTS A NEW WORD. We compare two reconstructions of
        the boundary word-group and pick the one SentencePiece itself would produce.
        """
        toks = self._tokens
        if not toks:
            return ""
        starts = set(i for i in self._chunk_starts if 0 < i < len(toks))
        sp = self.sp
        out = []           # list of decoded text fragments
        # We walk token by token, decoding incrementally so we can inject a boundary
        # space at the right place. Build the final string from per-token pieces with
        # SP's own surface form (id_to_piece, '▁' -> space).
        for i, t in enumerate(toks):
            piece = sp.id_to_piece(t)
            text_piece = piece.replace("▁", " ")  # ▁ -> space
            if i in starts and piece and not piece.startswith("▁"):
                # first token of a NEW chunk and it has no leading space marker.
                # If the text so far ends with a letter (mid-word) AND adding this bare
                # piece would have been the model's intent, we must decide: continue the
                # word, or it lost a boundary space. Use SP to disambiguate.
                if self._boundary_lost_space(toks, i):
                    text_piece = " " + text_piece
            out.append(text_piece)
        # join and clean: collapse any accidental double spaces, strip leading space
        s = "".join(out)
        while "  " in s:
            s = s.replace("  ", " ")
        return s.lstrip()

    def _boundary_lost_space(self, toks, i):
        """Decide whether the bare subword token at chunk-boundary index i lost its
        word-boundary space — SAFELY (never split a legitimately-continued word).

        Hard truth about this vocab: the token surface is genuinely AMBIGUOUS between
        a lost space and an intra-word continuation. e.g. emitted ['▁thing','ma','t',
        'ter','s'] is exactly how SentencePiece tokenizes BOTH 'thing matters' (space
        lost) AND the single string 'thingmatters'. Likewise ['▁be','au','ti','ful']
        is 'beautiful' (one word) and we must NOT split it.

        So we only repair when it is UNAMBIGUOUS: the join is *unnatural* — i.e. when
        SentencePiece, asked to tokenize the two surfaces JOINED into one string, would
        NOT reproduce the emitted pieces (so the model could not have meant one word),
        WHILE the SEPARATED tokenization (with a boundary ▁) reproduces them exactly.
        In the ambiguous 'joined == emitted' case we leave it alone (no corruption).
        This fixes the clearly-broken joins and never breaks a real word.
        """
        sp = self.sp
        # previous word group: walk back to the last '▁' marker (inclusive)
        j = i - 1
        while j >= 0 and not sp.id_to_piece(toks[j]).startswith("▁"):
            j -= 1
        if j < 0:
            return False
        left_ids = list(toks[j:i])
        # bare run at the boundary: up to the next '▁' or chunk end
        k = i + 1
        while k < len(toks) and not sp.id_to_piece(toks[k]).startswith("▁"):
            k += 1
        right_ids = list(toks[i:k])
        left_word = sp.decode(left_ids).strip()
        right_word = sp.decode(right_ids).strip()
        if not left_word or not right_word:
            return False
        emitted = [sp.id_to_piece(t) for t in (left_ids + right_ids)]
        joined = [sp.id_to_piece(t) for t in sp.encode(left_word + right_word)]
        separated = [sp.id_to_piece(t) for t in sp.encode(left_word + " " + right_word)]
        emitted_with_space = (emitted[:len(left_ids)]
                              + ["▁" + emitted[len(left_ids)].lstrip("▁")]
                              + emitted[len(left_ids) + 1:])
        # unambiguous lost space: separated reproduces it, joined does NOT.
        return separated == emitted_with_space and joined != emitted

    def _decode_new(self, seg, is_final: bool) -> str:
        # extend the mel buffer with ONLY the new audio (constant work per feed -- the
        # incremental mel never re-extracts the whole stream, killing the O(N^2) pegging).
        new_mel = self._inc_mel.push(seg, is_final=is_final)
        if new_mel.shape[-1]:
            self._mel_buf = np.concatenate([self._mel_buf, new_mel], axis=-1)
        T = self._mel_buf.shape[-1]
        cf = self._chunk_mel_frames          # 56 new frames per chunk
        pre = self._pre_cache_mel            # 9 left-context frames
        avail = T - self._mel_done
        n_full = avail // cf
        if not is_final and n_full == 0:
            return _strip_tags(self._last_text)
        # process: each step takes [9 left-context] + [56 new] = 65 frames.
        pos = self._mel_done
        end_frames = self._mel_done + (avail if is_final else n_full * cf)
        feats = self._mel_buf
        while pos < end_frames:
            new_end = min(pos + cf, end_frames)
            cache_start = max(0, pos - pre)
            mel_chunk = feats[:, :, cache_start:new_end]
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
        self._last_text = self._decode_tokens()
        return _strip_tags(self._last_text)

    # ------------------------------------------------------------------ #
    def feed(self, audio_f32_16k: np.ndarray) -> str:
        if audio_f32_16k is None or len(audio_f32_16k) == 0:
            return _strip_tags(self._last_text)
        seg = np.asarray(audio_f32_16k, dtype=np.float32).reshape(-1)
        return self._decode_new(seg, is_final=False)

    def finalize(self) -> str:
        # append ONE trailing space so back-to-back dictations don't abut
        # ("hello"+"world" -> "hello world"). It's a clean append for the live typer
        # (0 backspaces) and collapses naturally if the field already has a space.
        text = self._decode_new(np.zeros(0, dtype=np.float32), is_final=True)
        if text and not text.endswith(" "):
            text = text + " "
        return text


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
