from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from .session_manager import OrtSessionManager


def _load_audio(path: Path) -> Tuple[np.ndarray, int]:
    import soundfile as sf

    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    # soundfile: [T, C] -> [C, T]
    return wav.T, int(sr)


def _save_audio(path: Path, audio: np.ndarray, sample_rate: int) -> Path:
    import soundfile as sf

    audio_f = audio.astype(np.float32, copy=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    # [C, T] -> [T, C], 32-bit float WAV
    sf.write(str(path), audio_f.T, sample_rate, subtype="FLOAT")
    return path


def _resolve_baseline_audio(case_id: str) -> Path:
    summary_path = Path("fixtures/tensors") / f"{case_id}.json"
    if summary_path.exists():
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        audio_path = data.get("audio_path")
        if audio_path:
            candidate = Path(audio_path)
            if not candidate.is_absolute():
                candidate = Path.cwd() / candidate
            if candidate.exists():
                return candidate
    fallback = Path("fixtures/audio") / f"{case_id}.wav"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Baseline audio not found for case_id={case_id}")


def _audio_metrics(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    # a/b: [C, T]
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("Audio tensors must be 2D [channels, samples]")
    channels = min(a.shape[0], b.shape[0])
    length = min(a.shape[1], b.shape[1])
    a = a[:channels, :length].astype(np.float64)
    b = b[:channels, :length].astype(np.float64)
    diff = a - b
    rmse = float(np.sqrt(np.mean(diff * diff)))
    max_abs = float(np.max(np.abs(diff)))
    signal = float(np.mean(a * a))
    noise = float(np.mean(diff * diff))
    snr = float(10.0 * np.log10(signal / noise)) if noise > 0 else float("inf")
    denom = float(np.linalg.norm(a.ravel()) * np.linalg.norm(b.ravel()))
    corr = float(np.dot(a.ravel(), b.ravel()) / denom) if denom > 0 else 1.0
    return {"rmse": rmse, "max_abs": max_abs, "snr": snr, "corr": corr}


def decode_audio_from_npz(
    onnx_dir: Path,
    candidate_npz: Path,
    provider: str = "cpu",
) -> np.ndarray:
    arr = np.load(str(candidate_npz), allow_pickle=False)
    if "pred_latents" not in arr.files:
        raise ValueError("candidate npz missing pred_latents")
    latents = arr["pred_latents"].astype(np.float32)
    # [B, T, 64] -> [B, 64, T]
    latents = np.transpose(latents, (0, 2, 1))
    sessions = OrtSessionManager(onnx_dir)
    sessions.config.provider = provider
    audio = sessions.run("vae_decoder.onnx", {"latents": latents}, output_names=["audio"])[0]
    return audio.astype(np.float32)


def compare_audio(
    baseline_audio: Path,
    candidate_audio: np.ndarray,
    sample_rate: int,
) -> Dict[str, object]:
    base_wav, base_sr = _load_audio(baseline_audio)
    if base_sr != sample_rate:
        raise ValueError(f"Sample rate mismatch: baseline={base_sr} candidate={sample_rate}")
    metrics = _audio_metrics(base_wav, candidate_audio)
    return {
        "baseline_path": str(baseline_audio),
        "sample_rate": base_sr,
        "baseline_shape": list(base_wav.shape),
        "candidate_shape": list(candidate_audio.shape),
        "metrics": metrics,
    }


def run_audio_parity(
    case_id: str,
    onnx_dir: Path,
    candidate_npz: Path,
    out_audio: Path,
    out_json: Path,
    provider: str = "cpu",
    sample_rate: int = 48000,
) -> Dict[str, object]:
    baseline_audio = _resolve_baseline_audio(case_id)
    audio = decode_audio_from_npz(onnx_dir, candidate_npz, provider=provider)
    # use first batch element
    audio_0 = audio[0]
    _save_audio(out_audio, audio_0, sample_rate=sample_rate)
    report = compare_audio(baseline_audio, audio_0, sample_rate=sample_rate)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
