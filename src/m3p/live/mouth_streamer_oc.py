import json
import math
import os
import math as _math
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any

import numpy as np
from m3p.live.vowel_classifier import classify_vowel
import traceback


def _classify_vowel_from_f1f2(f1_hz: Optional[float], f2_hz: Optional[float]) -> int:
    """Map (F1,F2) -> mouth_id(0..5). Returns 0 when unavailable.

    This is a thin wrapper around m3p.live.vowel_classifier.classify_vowel.
    """
    if f1_hz is None or f2_hz is None:
        return 0
    try:
        f1v = float(f1_hz)
        f2v = float(f2_hz)
        # guard NaN/inf
        if not _math.isfinite(f1v) or not _math.isfinite(f2v):
            return 0
        mid = classify_vowel(f1v, f2v)
        return int(mid) if mid is not None else 0
    except Exception:
        return 0

# STEP2: Parselmouth(Formant) による母音推定（存在しない場合は simple によるフォールバック）
try:
    import parselmouth  # type: ignore
except Exception:  # pragma: no cover
    parselmouth = None  # type: ignore

# STEP1: 旧M3互換VAD（energy.py）を使用するための import（環境により import 経路が異なる）
try:
    from m3p.vad.energy import run_energy_vad  # repo 内
except Exception:  # pragma: no cover
    try:
        from energy import run_energy_vad  # 単体実行（同階層/ PYTHONPATH）
    except Exception:
        run_energy_vad = None  # type: ignore


def _ensure_dir(p: str) -> None:
    d = os.path.dirname(p)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def _pcm16_to_float_mono(b: bytes) -> np.ndarray:
    """
    PCM16 little-endian mono -> float32 [-1,1]
    """
    x = np.frombuffer(b, dtype=np.int16)
    if x.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return (x.astype(np.float32) / 32768.0).copy()


def _resample_linear(x: np.ndarray, in_sr: int, out_sr: int) -> np.ndarray:
    """
    Simple linear resampling (mono).
    """
    if in_sr == out_sr or x.size == 0:
        return x
    ratio = out_sr / float(in_sr)
    n_out = int(round(x.size * ratio))
    if n_out <= 0:
        return np.zeros((0,), dtype=np.float32)
    t_in = np.arange(x.size, dtype=np.float32)
    t_out = np.linspace(0, x.size - 1, n_out, dtype=np.float32)
    y = np.interp(t_out, t_in, x).astype(np.float32)
    return y


def _frame_rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    v = float(np.sqrt(np.mean(np.square(x.astype(np.float32))) + 1e-12))
    return v


@dataclass
class MouthOCConfig:
    step_ms: int = 40
    window_ms: int = 240
    analysis_sr: int = 16000
    input_sr_default: int = 24000
    rms_thr: float = 0.020
    open_id: int = 1
    close_id: int = 0

    vad_energy_thr: float = 0.020
    vad_min_speech_ms: int = 80
    vad_min_silence_ms: int = 120

    flush_every_frames: int = 25
    max_buffer_s: float = 10.0

    # STEP2: vowel classifier
    vowel_mode: str = "formant"  # formant (Parselmouth) | simple (spectral centroid)
    formant_window_ms: int = 200
    formant_max_hz: int = 5500


class EnergyVADStreamConfig:
    def __init__(self, analysis_sr: int, step_ms: int, energy_thr: float, min_speech_ms: int, min_silence_ms: int):
        self.analysis_sr = int(analysis_sr)
        self.step_ms = int(step_ms)
        self.energy_thr = float(energy_thr)
        self.min_speech_ms = int(min_speech_ms)
        self.min_silence_ms = int(min_silence_ms)


class EnergyVADStream:
    def __init__(self, cfg: EnergyVADStreamConfig):
        self.cfg = cfg
        self._step_samples = int(round(self.cfg.analysis_sr * self.cfg.step_ms / 1000.0))
        self._min_speech_steps = max(1, int(round(self.cfg.min_speech_ms / self.cfg.step_ms)))
        self._min_silence_steps = max(1, int(round(self.cfg.min_silence_ms / self.cfg.step_ms)))

        self._energies: List[float] = []
        self.step_mask: List[int] = []

        self._state = 0
        self._run = 0

    def _update_state(self, active: int) -> int:
        if self._state == 0:
            if active == 1:
                self._run += 1
                if self._run >= self._min_speech_steps:
                    self._state = 1
                    self._run = 0
            else:
                self._run = 0
        else:
            if active == 0:
                self._run += 1
                if self._run >= self._min_silence_steps:
                    self._state = 0
                    self._run = 0
            else:
                self._run = 0
        return self._state

    def ensure_steps(self, n_steps: int, audio_f: np.ndarray) -> None:
        while len(self.step_mask) < n_steps:
            i = len(self.step_mask)
            s = i * self._step_samples
            e = s + self._step_samples
            if e > audio_f.size:
                break
            seg = audio_f[s:e]
            rms = _frame_rms(seg)
            self._energies.append(rms)
            active = 1 if rms >= self.cfg.energy_thr else 0
            st = self._update_state(active)
            self.step_mask.append(int(st))


class Strict2Online:
    def __init__(self, open_id: int = 1, close_id: int = 0):
        self.open_id = int(open_id)
        self.close_id = int(close_id)

    def apply(self, vad_active: int, mouth_id_raw: int) -> int:
        if int(vad_active) == 0:
            return self.close_id
        if int(mouth_id_raw) == self.close_id:
            return self.open_id
        return int(mouth_id_raw)


class SimpleVowelClassifier:
    def __init__(self, analysis_sr: int):
        self.sr = int(analysis_sr)

    def classify(self, seg: np.ndarray) -> int:
        if seg.size == 0:
            return 0
        x = seg.astype(np.float32)
        w = np.hanning(x.size).astype(np.float32)
        X = np.fft.rfft(x * w)
        mag = np.abs(X).astype(np.float32)
        if mag.size <= 1:
            return 0
        freqs = np.fft.rfftfreq(x.size, d=1.0 / self.sr).astype(np.float32)
        centroid = float(np.sum(freqs * mag) / (np.sum(mag) + 1e-9))

        if centroid < 500:
            return 3
        if centroid < 800:
            return 1
        if centroid < 1200:
            return 2
        if centroid < 1800:
            return 4
        return 5


# STEP2: Formant-based vowel classifier (Parselmouth)


class FormantVowelClassifier:
    def __init__(self, analysis_sr: int, window_ms: int = 200, max_hz: int = 5500):
        self.sr = int(analysis_sr)
        self.window_ms = int(window_ms)
        self.max_hz = int(max_hz)

    def classify_at(self, wav_buf: np.ndarray, t0_ms: int, center_ms: int) -> tuple[int, Optional[float], Optional[float]]:
        """Return (mouth_id, f1, f2). mouth_id=0 if cannot estimate.

        Phase18: causally clamp the formant window to available audio instead of
        refusing when the future half of formant_window is not yet buffered.
        Still real Parselmouth formant → KNN mouth (no mouth_closed fill).
        """
        if parselmouth is None:
            return 0, None, None

        half = self.window_ms // 2
        start_ms = int(center_ms - half)
        end_ms = int(center_ms + half)

        start = int(round((start_ms - t0_ms) * self.sr / 1000.0))
        end = int(round((end_ms - t0_ms) * self.sr / 1000.0))

        # Causal clamp to buffered samples (frontier is past-heavy; interior
        # frames still see a near-symmetric window once audio has advanced).
        if start < 0:
            start = 0
        if end > wav_buf.size:
            end = int(wav_buf.size)
        if end <= start:
            return 0, None, None

        seg = wav_buf[start:end].astype(np.float32)
        if seg.size < int(self.sr * 0.05):  # <50ms
            return 0, None, None

        snd = parselmouth.Sound(seg, self.sr)
        formant = snd.to_formant_burg(
            time_step=0.01,
            max_number_of_formants=5,
            maximum_formant=float(self.max_hz),
        )
        # Use mid-segment time in the (possibly clamped) analysis window.
        t = snd.get_total_duration() / 2.0
        f1 = formant.get_value_at_time(1, t)
        f2 = formant.get_value_at_time(2, t)
        try:
            f1v = float(f1) if f1 is not None else None
            f2v = float(f2) if f2 is not None else None
        except Exception:
            f1v, f2v = None, None

        mouth_id = _classify_vowel_from_f1f2(f1v, f2v)
        return mouth_id, f1v, f2v


class MouthStreamerOC:
    def __init__(self, out_json: str, session_id: str, cfg: MouthOCConfig):
        self.out_json = out_json
        self.session_id = session_id
        self.cfg = cfg

        _ensure_dir(self.out_json)

        self._analysis_buf = np.zeros((0,), dtype=np.float32)
        self._t0_ms = 0

        self._step_samples = int(round(self.cfg.analysis_sr * self.cfg.step_ms / 1000.0))
        self._window_samples = int(round(self.cfg.analysis_sr * self.cfg.window_ms / 1000.0))

        self._frames: List[Dict[str, Any]] = []
        self._vad_01 = []
        self._meta: Dict[str, Any] = {
            "session_id": self.session_id,
            "cfg": asdict(self.cfg),
            # どの母音推定が動いているかを out JSON から一発で確認できるように
            "vowel_mode_effective": str(self.cfg.vowel_mode),
            "parselmouth_available": bool(parselmouth is not None),

            "analysis_sr": self.cfg.analysis_sr,
            "step_ms": self.cfg.step_ms,
            "window_ms": self.cfg.window_ms,
            "rms_thr": self.cfg.rms_thr,
            "vad_energy_thr": self.cfg.vad_energy_thr,
            "vad_min_speech_ms": self.cfg.vad_min_speech_ms,
            "vad_min_silence_ms": self.cfg.vad_min_silence_ms,
            # Phase18: emit on step-complete; formant window is causally clamped.
            "formant_causal_emit": True,
        }

        # STEP1用: energy.py で事前計算した VAD mask
        self._pre_vad_mask: Optional[List[int]] = None
        self._pre_vad_meta: Optional[Dict[str, Any]] = None

        # formant 失敗時のフォールバック用（常に用意）
        self._vowel_clf_simple = SimpleVowelClassifier(self.cfg.analysis_sr)

        # STEP2: vowel classifier selection
        if self.cfg.vowel_mode == "formant" and parselmouth is not None:
            self._vowel_clf = FormantVowelClassifier(
                self.cfg.analysis_sr,
                window_ms=self.cfg.formant_window_ms,
                max_hz=self.cfg.formant_max_hz,
            )
        else:
            self._vowel_clf = self._vowel_clf_simple

        self._vad = EnergyVADStream(
            EnergyVADStreamConfig(
                analysis_sr=self.cfg.analysis_sr,
                step_ms=self.cfg.step_ms,
                energy_thr=self.cfg.vad_energy_thr,
                min_speech_ms=self.cfg.vad_min_speech_ms,
                min_silence_ms=self.cfg.vad_min_silence_ms,
            )
        )
        self._strict2 = Strict2Online(open_id=self.cfg.open_id, close_id=self.cfg.close_id)

        self._flush_counter = 0

    def set_precomputed_vad_from_energy_py(self, vad_out: Dict[str, Any]) -> None:
        frames = vad_out.get("frames", [])
        step_ms = int(vad_out.get("step_ms", self.cfg.step_ms))

        if step_ms != int(self.cfg.step_ms):
            raise ValueError(f"VAD step_ms mismatch: vad_out.step_ms={step_ms} cfg.step_ms={self.cfg.step_ms}")

        mask: List[int] = []
        for fr in frames:
            va = int(fr.get("voice_activity", 0))
            mask.append(1 if va > 0 else 0)

        self._pre_vad_mask = mask
        self._pre_vad_meta = vad_out

    def _slice_abs(self, start_abs: int, end_abs: int) -> np.ndarray:
        if end_abs <= start_abs:
            return np.zeros((0,), dtype=np.float32)
        start_abs = max(0, start_abs)
        end_abs = min(self._analysis_buf.size, end_abs)
        if end_abs <= start_abs:
            return np.zeros((0,), dtype=np.float32)
        return self._analysis_buf[start_abs:end_abs]

    def _append_audio(self, x: np.ndarray) -> None:
        if x.size == 0:
            return
        self._analysis_buf = np.concatenate([self._analysis_buf, x], axis=0)

        max_keep = int(round(self.cfg.max_buffer_s * self.cfg.analysis_sr)) + self._window_samples
        if self._analysis_buf.size > max_keep:
            drop = self._analysis_buf.size - max_keep
            self._analysis_buf = self._analysis_buf[drop:]
            self._t0_ms += int(round(drop * 1000.0 / self.cfg.analysis_sr))

    def push_pcm16_mono(self, b: bytes, input_sr: Optional[int] = None) -> int:
        if input_sr is None:
            input_sr = self.cfg.input_sr_default

        x = _pcm16_to_float_mono(b)
        x = _resample_linear(x, in_sr=input_sr, out_sr=self.cfg.analysis_sr)
        self._append_audio(x)

        before = len(self._frames)
        self._emit_ready_frames()
        after = len(self._frames)
        return after - before

    def _emit_ready_frames(self) -> None:
        buf_ms = int(round(self._analysis_buf.size * 1000.0 / self.cfg.analysis_sr))
        end_ms = self._t0_ms + buf_ms

        next_k = len(self._frames)
        next_t_ms = next_k * self.cfg.step_ms

        while True:
            # Phase18: emit as soon as one step of audio is present.
            # Previously formant mode waited for center+(formant_window/2) of
            # *future* samples (default +120ms), which produced a structural
            # mouth lag of ~80ms vs PCM (until - mouth_cov). classify_at now
            # causally clamps the analysis window so we keep real formant→KNN
            # mouth shapes without blocking emit on unavailable future audio.
            seg_end_ms = next_t_ms + self.cfg.step_ms
            need_end_ms = seg_end_ms

            if need_end_ms > end_ms:
                break

            seg_end_abs = int(round((seg_end_ms - self._t0_ms) * self.cfg.analysis_sr / 1000.0))
            seg_start_abs = seg_end_abs - self._step_samples
            win_start_abs = seg_end_abs - self._window_samples

            seg = self._slice_abs(seg_start_abs, seg_end_abs)
            rms = _frame_rms(seg)

            # VAD（STEP1で確定した旧M3互換 mask を優先）
            if self._pre_vad_mask is not None:
                vad_active = int(self._pre_vad_mask[next_k]) if next_k < len(self._pre_vad_mask) else 0
            else:
                self._vad.ensure_steps(next_k + 1, self._analysis_buf)
                vad_active = int(self._vad.step_mask[next_k]) if len(self._vad.step_mask) > next_k else 0

            # STEP2: 母音推定は VAD=1 のフレームのみ
            f1 = None
            f2 = None
            vowel_fallback = False
            fallback_reason = None
            fallback_exception = None

            if int(vad_active) == 1:
                try:
                    if hasattr(self._vowel_clf, "classify_at"):
                        center_ms = int(next_t_ms + (self.cfg.step_ms // 2))
                        mouth_id_raw, f1, f2 = self._vowel_clf.classify_at(self._analysis_buf, self._t0_ms, center_ms)
                        # classify_at が失敗を (open_id, None, None) で返す実装に備える
                        if (f1 is None) or (f2 is None):
                            vowel_fallback = True
                            fallback_reason = "formant_estimate_failed"

                        # formant が推定不能(0/None/NaN)なら simple にフォールバック
                        if int(mouth_id_raw) == 0:
                            mouth_raw = self._vowel_clf_simple.classify(self._slice_abs(win_start_abs, seg_end_abs))
                            mouth_id_raw = int(mouth_raw) if mouth_raw > 0 else self.cfg.open_id
                            vowel_fallback = True
                            fallback_reason = "simple_from_formant0"
                    else:
                        mouth_raw = self._vowel_clf.classify(self._slice_abs(win_start_abs, seg_end_abs))
                        if mouth_raw > 0:
                            mouth_id_raw = int(mouth_raw)
                        else:
                            mouth_id_raw = self.cfg.open_id
                            vowel_fallback = True
                            fallback_reason = "simple_classify_failed"
                
                except Exception as e:
                    mouth_raw = self._vowel_clf_simple.classify(self._slice_abs(win_start_abs, seg_end_abs))
                    mouth_id_raw = int(mouth_raw) if mouth_raw > 0 else self.cfg.open_id
                    vowel_fallback = True
                    fallback_reason = "exception_fallback_to_simple"

                    exc_type = type(e).__name__
                    exc_msg = str(e)
                    exc_tb = traceback.format_exc()
                    exc_tb_head = "\n".join(exc_tb.splitlines()[:8])

                    fallback_exception = {
                        "type": exc_type,
                        "msg": exc_msg,
                        "trace_head": exc_tb_head,
                    }

                    self._meta["formant_exception_count"] = int(self._meta.get("formant_exception_count", 0)) + 1
                    if "formant_exception_first" not in self._meta:
                        self._meta["formant_exception_first"] = fallback_exception
            else:
                mouth_id_raw = self.cfg.close_id

            mouth_id = int(self._strict2.apply(vad_active, mouth_id_raw))
            vad01 = int(vad_active)

            fr = {
                "t_ms": int(next_t_ms),
                "mouth_id": int(mouth_id),
                "meta": {
                    "rms": float(rms),
                    "mouth_id_raw": int(mouth_id_raw),
                    "f1_hz": (None if f1 is None else float(f1)),
                    "f2_hz": (None if f2 is None else float(f2)),
                    "vowel_fallback": bool(vowel_fallback),
                    "fallback_reason": fallback_reason,
                    "fallback_exception": (fallback_exception if fallback_reason == "exception_fallback_to_simple" else None),
                    "vad_active": int(vad_active),
                    "cfg": {
                        "rms_thr": float(self.cfg.rms_thr),
                        "vad_energy_thr": float(self.cfg.vad_energy_thr),
                        "vad_min_speech_ms": int(self.cfg.vad_min_speech_ms),
                        "vad_min_silence_ms": int(self.cfg.vad_min_silence_ms),
                    },
                },
            }
            self._frames.append(fr)
            self._vad_01.append(int(vad01))

            self._flush_counter += 1
            if self._flush_counter >= self.cfg.flush_every_frames:
                self.flush()
                self._flush_counter = 0

            next_k = len(self._frames)
            next_t_ms = next_k * self.cfg.step_ms

    def flush(self) -> None:
        if self._pre_vad_meta is not None:
            vad_meta = {
                "schema_version": "energy_vad.energy_py.strict2.v0.1",
                "step_ms": int(self.cfg.step_ms),
                "mask": self._vad_01,
                "algo": "energy_py",
                "params": {
                    "frame_ms": int(self._pre_vad_meta.get("frame_ms", 20)),
                    "hop_ms": int(self._pre_vad_meta.get("hop_ms", 10)),
                    "energy_thr": float(self._pre_vad_meta.get("energy_thr", 1e-4)),
                    "min_speech_ms": int(self._pre_vad_meta.get("min_speech_ms", 100)),
                    "min_silence_ms": int(self._pre_vad_meta.get("min_silence_ms", 100)),
                    "analysis_sr": int(self.cfg.analysis_sr),
                },
            }
        else:
            vad_meta = {
                "schema_version": "energy_vad.strict2.v0.1",
                "step_ms": int(self.cfg.step_ms),
                "mask": self._vad_01,
                "algo": "energy_stream",
                "params": {
                    "frame_ms": int(self.cfg.step_ms),
                    "hop_ms": int(self.cfg.step_ms),
                    "energy_thr": float(self.cfg.vad_energy_thr),
                    "min_speech_ms": int(self.cfg.vad_min_speech_ms),
                    "min_silence_ms": int(self.cfg.vad_min_silence_ms),
                    "analysis_sr": int(self.cfg.analysis_sr),
                },
            }

        out = {
            "format": "mouth_timeline_v0",
            "version": "mouth_timeline.live.v0.3",
            "session_id": self.session_id,
            "step_ms": int(self.cfg.step_ms),
           
            "frames": [
                {
                    "t_ms": fr["t_ms"],
                    "mouth_id": fr["mouth_id"],
                    "vad_active": (fr.get("meta") or {}).get("vad_active", None),
                    "rms": (fr.get("meta") or {}).get("rms", None),
                    "mouth_id_raw": (fr.get("meta") or {}).get("mouth_id_raw", None),
                    "f1_hz": (fr.get("meta") or {}).get("f1_hz", None),
                    "f2_hz": (fr.get("meta") or {}).get("f2_hz", None),
                    "vowel_fallback": (fr.get("meta") or {}).get("vowel_fallback", None),
                    "fallback_reason": (fr.get("meta") or {}).get("fallback_reason", None),
                }
                for fr in self._frames
            ],
            "vad_meta": vad_meta,
            "meta": self._meta,
            "debug_frames": self._frames,
        }
        with open(self.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

    def finalize(self) -> None:
        self._emit_ready_frames()
        self.flush()


def run_pseudo_live_from_wav(wav_path: str, streamer: MouthStreamerOC, chunk_ms: int = 40) -> None:
    if run_energy_vad is None:
        raise ImportError("run_energy_vad not available. Ensure m3p.vad.energy (or energy.py) is importable.")

    vad_cfg = {
        "session_id": getattr(streamer, "session_id", None),
        "utt_id": None,
        "frame_ms": 20,
        "hop_ms": 10,
        "step_ms": int(streamer.cfg.step_ms),
        "energy_thr": 1e-4,
        "min_speech_ms": 100,
        "min_silence_ms": 100,
    }
    vad_out = run_energy_vad(wav_path, vad_cfg)
    streamer.set_precomputed_vad_from_energy_py(vad_out)

    import wave

    def _downmix_pcm16_stereo_to_mono(b: bytes) -> bytes:
        x = np.frombuffer(b, dtype=np.int16)
        if x.size == 0:
            return b""
        if x.size % 2 != 0:
            x = x[: x.size - 1]
        x2 = x.reshape(-1, 2).astype(np.int32)
        mono = ((x2[:, 0] + x2[:, 1]) // 2).astype(np.int16)
        return mono.tobytes()

    with wave.open(wav_path, "rb") as wf:
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        sr = wf.getframerate()
        n = wf.getnframes()

        if sw != 2:
            raise RuntimeError(f"Only PCM16 supported (sampwidth=2). got ch={ch} sw={sw}")
        if ch not in (1, 2):
            raise RuntimeError(f"Only mono/stereo supported. got ch={ch} sw={sw}")

        chunk_frames = int(round(sr * chunk_ms / 1000.0))
        if chunk_frames <= 0:
            chunk_frames = 1

        remaining = n
        while remaining > 0:
            take = min(chunk_frames, remaining)
            b = wf.readframes(take)
            if ch == 2:
                b = _downmix_pcm16_stereo_to_mono(b)
            streamer.push_pcm16_mono(b, input_sr=sr)
            remaining -= take

if __name__ == "__main__":
    # tiny self-check
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--step_ms", type=int, default=40)
    ap.add_argument("--analysis_sr", type=int, default=16000)
    ap.add_argument("--rms_thr", type=float, default=0.020)
    ap.add_argument("--vad_rms_thr", type=float, default=0.020)
    ap.add_argument("--vad_energy_thr", type=float, default=None)
    ap.add_argument("--vad_min_speech_ms", type=int, default=80)
    ap.add_argument("--vad_min_silence_ms", type=int, default=120)
    ap.add_argument("--vowel_mode", type=str, default="formant", choices=["formant", "simple"])
    ap.add_argument("--formant_window_ms", type=int, default=200)
    ap.add_argument("--formant_max_hz", type=int, default=5500)
    args = ap.parse_args()

    st = MouthStreamerOC(
        out_json=args.out,
        session_id="selfcheck",
        cfg=MouthOCConfig(
            step_ms=args.step_ms,
            analysis_sr=args.analysis_sr,
            rms_thr=args.rms_thr,
            vad_energy_thr=(args.vad_energy_thr if args.vad_energy_thr is not None else float(args.vad_rms_thr) ** 2),
            vad_min_speech_ms=args.vad_min_speech_ms,
            vad_min_silence_ms=args.vad_min_silence_ms,
            vowel_mode=args.vowel_mode,
            formant_window_ms=args.formant_window_ms,
            formant_max_hz=args.formant_max_hz,
        ),
    )
    run_pseudo_live_from_wav(args.wav, st, chunk_ms=args.step_ms)
    st.finalize()
    print("[OK] wrote:", args.out)