#!/usr/bin/env python3
"""Parity comparison utility for tensors (npz) and audio files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

try:
    import soundfile as sf
except Exception:
    sf = None


def _metric_dict(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    da = a.astype(np.float64, copy=False)
    db = b.astype(np.float64, copy=False)
    diff = da - db

    max_abs = float(np.max(np.abs(diff)))
    mean_abs = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    denom = float(np.linalg.norm(da.ravel()) * np.linalg.norm(db.ravel()))
    cos_sim = float(np.dot(da.ravel(), db.ravel()) / denom) if denom > 0 else 1.0

    return {
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "rmse": rmse,
        "cos_sim": cos_sim,
    }


def _audio_metrics(path_a: Path, path_b: Path) -> Dict[str, Any]:
    if sf is None:
        raise RuntimeError("soundfile is required for audio metrics")

    wav_a, sr_a = sf.read(str(path_a), always_2d=False)
    wav_b, sr_b = sf.read(str(path_b), always_2d=False)
    if sr_a != sr_b:
        raise ValueError(f"Sample rate mismatch: {sr_a} vs {sr_b}")

    wav_a = np.asarray(wav_a, dtype=np.float64).reshape(-1)
    wav_b = np.asarray(wav_b, dtype=np.float64).reshape(-1)
    if wav_a.shape != wav_b.shape:
        raise ValueError(f"Audio length mismatch: {wav_a.shape} vs {wav_b.shape}")

    noise = wav_a - wav_b
    sig = np.mean(wav_a * wav_a)
    noi = np.mean(noise * noise)
    snr = float(10.0 * np.log10((sig + 1e-12) / (noi + 1e-12)))
    corr = float(np.corrcoef(wav_a, wav_b)[0, 1]) if wav_a.size > 1 else 1.0

    return {
        "sample_rate": int(sr_a),
        "num_samples": int(wav_a.shape[0]),
        "snr_db": snr,
        "corr": corr,
    }


def _compare_npz(baseline_npz: Path, candidate_npz: Path) -> Dict[str, Any]:
    a = np.load(str(baseline_npz), allow_pickle=False)
    b = np.load(str(candidate_npz), allow_pickle=False)
    common = sorted(set(a.files).intersection(b.files))
    only_a = sorted(set(a.files) - set(b.files))
    only_b = sorted(set(b.files) - set(a.files))

    per_tensor: Dict[str, Dict[str, float]] = {}
    for key in common:
        per_tensor[key] = _metric_dict(a[key], b[key])

    return {
        "common_keys": common,
        "baseline_only": only_a,
        "candidate_only": only_b,
        "per_tensor": per_tensor,
    }


def _check_thresholds(
    report: Dict[str, Any],
    max_abs: float,
    rmse: float,
    min_corr: float,
    min_snr: float,
) -> Dict[str, Any]:
    failures: List[str] = []
    for name, metric in report.get("per_tensor", {}).items():
        if metric["max_abs"] > max_abs:
            failures.append(f"tensor {name}: max_abs {metric['max_abs']:.3e} > {max_abs:.3e}")
        if metric["rmse"] > rmse:
            failures.append(f"tensor {name}: rmse {metric['rmse']:.3e} > {rmse:.3e}")

    audio = report.get("audio")
    if audio:
        if audio["corr"] < min_corr:
            failures.append(f"audio corr {audio['corr']:.6f} < {min_corr:.6f}")
        if audio["snr_db"] < min_snr:
            failures.append(f"audio snr {audio['snr_db']:.3f} < {min_snr:.3f}")

    return {"pass": len(failures) == 0, "failures": failures}


def main() -> int:
    p = argparse.ArgumentParser(description="Compare baseline and candidate artifacts")
    p.add_argument("--baseline-npz", type=Path, required=True)
    p.add_argument("--candidate-npz", type=Path, required=True)
    p.add_argument("--baseline-audio", type=Path, default=None)
    p.add_argument("--candidate-audio", type=Path, default=None)
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--max-abs", type=float, default=1e-4)
    p.add_argument("--rmse", type=float, default=1e-5)
    p.add_argument("--min-corr", type=float, default=0.98)
    p.add_argument("--min-snr", type=float, default=35.0)
    args = p.parse_args()

    report = _compare_npz(args.baseline_npz, args.candidate_npz)
    if args.baseline_audio and args.candidate_audio:
        report["audio"] = _audio_metrics(args.baseline_audio, args.candidate_audio)

    report["thresholds"] = {
        "max_abs": args.max_abs,
        "rmse": args.rmse,
        "min_corr": args.min_corr,
        "min_snr": args.min_snr,
    }
    report["verdict"] = _check_thresholds(report, args.max_abs, args.rmse, args.min_corr, args.min_snr)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote: {args.out_json}")
    print(f"PASS: {report['verdict']['pass']}")
    if not report["verdict"]["pass"]:
        for failure in report["verdict"]["failures"]:
            print(f"  - {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
