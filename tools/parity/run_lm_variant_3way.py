#!/usr/bin/env python3
"""Run simple_mode 3-way demos per LM variant.

Outputs:
- reports/model_demos/<variant_tag>/baseline/{audio,tensors}
- reports/model_demos/<variant_tag>/py_onnx
- reports/model_demos/<variant_tag>/rust
- reports/model_demos/<variant_tag>/parity_3way
- reports/model_demos/<variant_tag>/listening/<case>/{python_original.wav,python_onnx.wav,rust_onnx.wav}
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


LM_VARIANTS = {
    "0.6B": {"lm_model_path": "acestep-5Hz-lm-0.6B", "tag": "lm_0p6"},
    "1.7B": {"lm_model_path": "acestep-5Hz-lm-1.7B", "tag": "lm_1p7"},
}


def _run(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _tensor_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    da = a.astype(np.float64)
    db = b.astype(np.float64)
    diff = da - db
    rmse = float(np.sqrt(np.mean(diff * diff)))
    max_abs = float(np.max(np.abs(diff)))
    denom = float(np.linalg.norm(da.ravel()) * np.linalg.norm(db.ravel()))
    cos_sim = float(np.dot(da.ravel(), db.ravel()) / denom) if denom > 0 else 1.0
    return {"rmse": rmse, "max_abs": max_abs, "cos_sim": cos_sim}


def _compare_npz(path_a: Path, path_b: Path) -> dict[str, Any]:
    a = np.load(path_a, allow_pickle=False)
    b = np.load(path_b, allow_pickle=False)
    keys = sorted(set(a.files) & set(b.files))
    metrics: dict[str, dict[str, float]] = {}
    for key in keys:
        if a[key].shape == b[key].shape:
            metrics[key] = _tensor_metrics(a[key], b[key])
    return {"keys": keys, "metrics": metrics}


def _audio_metrics(path_a: Path, path_b: Path) -> dict[str, Any]:
    a, sr1 = sf.read(path_a, dtype="float32", always_2d=True)
    b, sr2 = sf.read(path_b, dtype="float32", always_2d=True)
    a = a.T.astype(np.float64)
    b = b.T.astype(np.float64)
    channels = min(a.shape[0], b.shape[0])
    samples = min(a.shape[1], b.shape[1])
    a = a[:channels, :samples]
    b = b[:channels, :samples]
    diff = a - b
    rmse = float(np.sqrt(np.mean(diff * diff)))
    max_abs = float(np.max(np.abs(diff)))
    signal = float(np.mean(a * a))
    noise = float(np.mean(diff * diff))
    snr = float(10.0 * np.log10(signal / noise)) if noise > 0 else float("inf")
    denom = float(np.linalg.norm(a.ravel()) * np.linalg.norm(b.ravel()))
    corr = float(np.dot(a.ravel(), b.ravel()) / denom) if denom > 0 else 1.0
    return {
        "sample_rate_a": int(sr1),
        "sample_rate_b": int(sr2),
        "shape_a": [int(channels), int(samples)],
        "shape_b": [int(channels), int(samples)],
        "metrics": {"rmse": rmse, "max_abs": max_abs, "snr": snr, "corr": corr},
    }


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run LM variant 3-way demos for simple_mode cases")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--onnx-dir", type=Path, default=Path("artifacts/onnx_runtime"))
    parser.add_argument("--cases-dir", type=Path, default=Path("fixtures/cases"))
    parser.add_argument("--out-root", type=Path, default=Path("reports/model_demos"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--variants", nargs="+", choices=sorted(LM_VARIANTS.keys()), default=["0.6B", "1.7B"])
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    cases = sorted(args.cases_dir.glob("simple_mode_*.json"))
    if not cases:
        raise RuntimeError(f"No simple_mode case found in {args.cases_dir}")

    py = project_root / ".venv/Scripts/python.exe"
    if not py.exists():
        py = Path("python")

    all_summary: dict[str, Any] = {"variants": {}}

    for variant in args.variants:
        info = LM_VARIANTS[variant]
        tag = str(info["tag"])
        variant_root = (project_root / args.out_root / tag).resolve()
        baseline_root = variant_root / "baseline"
        py_root = variant_root / "py_onnx"
        rust_root = variant_root / "rust"
        par3_root = variant_root / "parity_3way"
        listening_root = variant_root / "listening"
        for d in [baseline_root, py_root, rust_root, par3_root, listening_root]:
            d.mkdir(parents=True, exist_ok=True)

        variant_summary: dict[str, Any] = {"variant": variant, "tag": tag, "cases": []}

        for case_path in cases:
            case_id = case_path.stem
            print(f"[{variant}] case={case_id}")

            _run(
                [
                    str(py),
                    "tools/baseline/dump_baseline.py",
                    "--case",
                    str(case_path),
                    "--project-root",
                    str(project_root),
                    "--output-dir",
                    str(baseline_root),
                    "--lm-model-path",
                    str(info["lm_model_path"]),
                    "--device",
                    args.device,
                    "--simple-mode-temperature",
                    "0",
                    "--simple-mode-top-k",
                    "0",
                    "--simple-mode-top-p",
                    "1.0",
                    "--simple-mode-repetition-penalty",
                    "1.0",
                ],
                cwd=project_root,
            )

            baseline_npz = baseline_root / "tensors" / f"{case_id}.npz"
            baseline_wav = baseline_root / "audio" / f"{case_id}.wav"
            py_npz = py_root / f"{case_id}_pyonnx.npz"
            py_wav = py_root / f"{case_id}_pyonnx.wav"
            rust_npz = rust_root / f"{case_id}_rust.npz"
            rust_wav = rust_root / f"{case_id}_rust.wav"

            _run(
                [
                    str(py),
                    "-m",
                    "runtime_py_ort.cli",
                    "full-generate",
                    "--case",
                    str(case_path),
                    "--inputs-npz",
                    str(baseline_npz),
                    "--onnx-dir",
                    str(args.onnx_dir),
                    "--provider",
                    args.device,
                    "--out-dir",
                    str(py_root),
                    "--out-npz",
                    str(py_npz),
                    "--out-audio",
                    str(py_wav),
                    "--mode",
                    "simple_mode",
                    "--lm-model",
                    str(variant),
                    "--lm-constrained",
                    "true",
                    "--deterministic",
                    "true",
                    "--max-new-tokens",
                    "768",
                ],
                cwd=project_root,
            )

            _run(
                [
                    str(py),
                    "-m",
                    "runtime_py_ort.cli",
                    "full-parity",
                    "--case",
                    str(case_path),
                    "--onnx-dir",
                    str(args.onnx_dir),
                    "--provider",
                    args.device,
                    "--out-dir",
                    str(py_root),
                    "--baseline-npz",
                    str(baseline_npz),
                    "--candidate-npz",
                    str(py_npz),
                ],
                cwd=project_root,
            )

            _run(
                [
                    "cargo",
                    "run",
                    "--release",
                    "--",
                    "generate",
                    "--case",
                    str(case_path),
                    "--inputs-npz",
                    str(baseline_npz),
                    "--onnx-dir",
                    str(args.onnx_dir),
                    "--out-npz",
                    str(rust_npz),
                    "--out-wav",
                    str(rust_wav),
                    "--online-lm-simple-mode",
                    "--lm-model",
                    str(variant),
                    "--lm-max-new-tokens",
                    "768",
                ],
                cwd=project_root,
            )
            rust_report = rust_root / f"{case_id}_rust_report.json"
            rust_audio_report = rust_root / f"{case_id}_rust_audio_report.json"
            _run(
                [
                    "cargo",
                    "run",
                    "--release",
                    "--",
                    "parity",
                    "--baseline",
                    str(baseline_npz),
                    "--candidate",
                    str(rust_npz),
                    "--out-json",
                    str(rust_report),
                ],
                cwd=project_root,
            )
            _run(
                [
                    "cargo",
                    "run",
                    "--release",
                    "--",
                    "audio-parity",
                    "--case",
                    str(case_path),
                    "--candidate-npz",
                    str(rust_npz),
                    "--out-wav",
                    str(rust_wav),
                    "--out-json",
                    str(rust_audio_report),
                    "--onnx-dir",
                    str(args.onnx_dir),
                ],
                cwd=project_root,
            )

            tensor_bp = _compare_npz(baseline_npz, py_npz)
            tensor_br = _compare_npz(baseline_npz, rust_npz)
            tensor_pr = _compare_npz(py_npz, rust_npz)
            audio_bp = _audio_metrics(baseline_wav, py_wav)
            audio_br = _audio_metrics(baseline_wav, rust_wav)
            audio_pr = _audio_metrics(py_wav, rust_wav)

            _write_json(par3_root / f"{case_id}_tensor_baseline_vs_py.json", tensor_bp)
            _write_json(par3_root / f"{case_id}_tensor_baseline_vs_rust.json", tensor_br)
            _write_json(par3_root / f"{case_id}_tensor_py_vs_rust.json", tensor_pr)
            _write_json(par3_root / f"{case_id}_audio_baseline_vs_py.json", audio_bp)
            _write_json(par3_root / f"{case_id}_audio_baseline_vs_rust.json", audio_br)
            _write_json(par3_root / f"{case_id}_audio_py_vs_rust.json", audio_pr)

            listen_case_dir = listening_root / case_id
            listen_case_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(baseline_wav, listen_case_dir / "python_original.wav")
            shutil.copy2(py_wav, listen_case_dir / "python_onnx.wav")
            shutil.copy2(rust_wav, listen_case_dir / "rust_onnx.wav")

            rust_tensor_pass = None
            if rust_report.exists():
                rust_obj = json.loads(rust_report.read_text(encoding="utf-8"))
                rust_tensor_pass = bool(rust_obj.get("verdict", {}).get("pass", False))

            variant_summary["cases"].append(
                {
                    "case_id": case_id,
                    "baseline_npz": str(baseline_npz.relative_to(project_root)).replace("\\", "/"),
                    "baseline_wav": str(baseline_wav.relative_to(project_root)).replace("\\", "/"),
                    "py_npz": str(py_npz.relative_to(project_root)).replace("\\", "/"),
                    "py_wav": str(py_wav.relative_to(project_root)).replace("\\", "/"),
                    "rust_npz": str(rust_npz.relative_to(project_root)).replace("\\", "/"),
                    "rust_wav": str(rust_wav.relative_to(project_root)).replace("\\", "/"),
                    "listening_dir": str(listen_case_dir.relative_to(project_root)).replace("\\", "/"),
                    "rust_tensor_pass": rust_tensor_pass,
                }
            )

        _write_json(variant_root / "summary.json", variant_summary)
        all_summary["variants"][tag] = {
            "variant": variant,
            "summary": str((variant_root / "summary.json").relative_to(project_root)).replace("\\", "/"),
        }

    _write_json((project_root / args.out_root / "summary.json"), all_summary)
    print(f"Wrote {(project_root / args.out_root / 'summary.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
