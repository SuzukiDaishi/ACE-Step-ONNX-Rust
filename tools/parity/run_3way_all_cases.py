#!/usr/bin/env python3
"""Run 3-way parity for all fixture cases (python original baseline / py_onnx / rust_onnx)."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


def _run(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _run_with_fallback(
    cmd: list[str],
    cwd: Path,
    case_id: str,
    label: str,
    provider_value_index: int | None,
    fallback_provider: str | None,
) -> str:
    used_provider = cmd[provider_value_index] if provider_value_index is not None else "n/a"
    try:
        _run(cmd, cwd=cwd)
        return used_provider
    except subprocess.CalledProcessError:
        if provider_value_index is None or fallback_provider is None:
            raise
        original_provider = cmd[provider_value_index]
        if original_provider == fallback_provider:
            raise
        print(
            f"[fallback] case={case_id} {label} failed on provider={original_provider}; retry with {fallback_provider}"
        )
        retry_cmd = list(cmd)
        retry_cmd[provider_value_index] = fallback_provider
        _run(retry_cmd, cwd=cwd)
        return fallback_provider


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
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _metric_exceeds(metric: dict[str, float], rmse_th: float, max_abs_th: float) -> bool:
    return float(metric.get("rmse", 0.0)) > rmse_th or float(metric.get("max_abs", 0.0)) > max_abs_th


def _classify_fail_reasons(
    mode: str,
    tensor_metrics: dict[str, dict[str, float]],
    audio_metrics: dict[str, Any],
    tensor_rmse_th: float,
    tensor_max_abs_th: float,
    audio_snr_th: float,
    audio_corr_th: float,
) -> list[str]:
    tags: list[str] = []
    if mode == "simple_mode":
        for key in ["encoder_hidden_states", "context_latents", "encoder_attention_mask"]:
            m = tensor_metrics.get(key)
            if m and _metric_exceeds(m, tensor_rmse_th, tensor_max_abs_th):
                tags.append("LM")
                break
    for key in ["vt_steps", "xt_steps", "pred_latents"]:
        m = tensor_metrics.get(key)
        if m and _metric_exceeds(m, tensor_rmse_th, tensor_max_abs_th):
            tags.append("DiT")
            break
    am = audio_metrics.get("metrics", {})
    if float(am.get("snr", float("inf"))) < audio_snr_th or float(am.get("corr", 1.0)) < audio_corr_th:
        tags.append("VAE")
    out = sorted(set(tags))
    return out


def _load_case_thresholds(path: Path | None) -> dict[str, dict[str, float]]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Case threshold override file not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("case thresholds must be a JSON object")
    thresholds: dict[str, dict[str, float]] = {}
    for case_id, data in raw.items():
        if not isinstance(data, dict):
            continue
        entry: dict[str, float] = {}
        for key in ["tensor_rmse", "tensor_max_abs", "audio_snr", "audio_corr"]:
            if key in data:
                entry[key] = float(data[key])
        if entry:
            thresholds[str(case_id)] = entry
    return thresholds


def main() -> int:
    parser = argparse.ArgumentParser(description="Run 3-way parity for all 10 fixture cases")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--onnx-dir", type=Path, default=Path("artifacts/onnx_runtime"))
    parser.add_argument("--cases-dir", type=Path, default=Path("fixtures/cases"))
    parser.add_argument("--baseline-tensors", type=Path, default=Path("fixtures/tensors"))
    parser.add_argument("--baseline-audio", type=Path, default=Path("fixtures/audio"))
    parser.add_argument("--out-root", type=Path, default=Path("reports"))
    parser.add_argument("--provider", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--audio-provider", choices=["cpu", "cuda"])
    parser.add_argument("--fallback-provider", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--lm-model", choices=["0.6B", "1.7B"], default="1.7B")
    parser.add_argument("--onnx-profile", choices=["fp32", "fp16", "int8_dynamic", "int8_static"], default="fp32")
    parser.add_argument("--tensor-rmse-threshold", type=float, default=5e-4)
    parser.add_argument("--tensor-max-abs-threshold", type=float, default=1e-1)
    parser.add_argument("--audio-snr-threshold", type=float, default=35.0)
    parser.add_argument("--audio-corr-threshold", type=float, default=0.98)
    parser.add_argument("--case-thresholds", type=Path, default=None)
    parser.add_argument("--case-ids", type=str, default="")
    parser.add_argument("--skip-audio", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    cases = sorted(args.cases_dir.glob("*.json"))
    if not cases:
        raise RuntimeError(f"No cases found in {args.cases_dir}")
    if args.case_ids:
        wanted = {c.strip() for c in args.case_ids.split(",") if c.strip()}
        cases = [p for p in cases if p.stem in wanted]
        if not cases:
            raise RuntimeError(f"No matching cases for: {sorted(wanted)}")

    py = project_root / ".venv/Scripts/python.exe"
    if not py.exists():
        py = Path("python")

    py_root = (project_root / args.out_root / "parity_py_ort").resolve()
    rust_root = (project_root / args.out_root / "parity_rust").resolve()
    par3_root = (project_root / args.out_root / "parity_3way").resolve()
    listen_root = (project_root / args.out_root / "listening").resolve()
    for d in [py_root, rust_root, par3_root, listen_root]:
        d.mkdir(parents=True, exist_ok=True)

    resolved_onnx_dir = (project_root / args.onnx_dir).resolve()
    if args.onnx_profile != "fp32":
        resolved_onnx_dir = (project_root / "artifacts" / "onnx_runtime_optimized" / args.onnx_profile).resolve()
    if not resolved_onnx_dir.exists():
        raise FileNotFoundError(f"ONNX profile directory not found: {resolved_onnx_dir}")

    try:
        onnx_dir_display = str(resolved_onnx_dir.relative_to(project_root)).replace("\\", "/")
    except ValueError:
        onnx_dir_display = str(resolved_onnx_dir)

    summary: dict[str, Any] = {
        "lm_model": args.lm_model,
        "provider": args.provider,
        "onnx_profile": args.onnx_profile,
        "onnx_dir": onnx_dir_display,
        "cases": [],
    }
    case_thresholds = _load_case_thresholds(args.case_thresholds)

    for case_path in cases:
        case_obj = json.loads(case_path.read_text(encoding="utf-8"))
        case_id = str(case_obj.get("case_id", case_path.stem))
        mode = str(case_obj.get("mode", "text2music"))
        print(f"[{mode}] {case_id}")

        baseline_npz = (project_root / args.baseline_tensors / f"{case_id}.npz").resolve()
        baseline_wav = (project_root / args.baseline_audio / f"{case_id}.wav").resolve()
        if not baseline_npz.exists() or not baseline_wav.exists():
            raise FileNotFoundError(f"Missing baseline artifact for {case_id}")

        py_npz = py_root / f"{case_id}_pyonnx.npz"
        py_wav = py_root / f"{case_id}_pyonnx.wav"
        rust_npz = rust_root / f"{case_id}_rust.npz"
        rust_wav = rust_root / f"{case_id}_rust.wav"
        done_marker = par3_root / f"{case_id}_failure_tags.json"
        listen_case_dir = listen_root / case_id
        if args.resume and done_marker.exists():
            if args.skip_audio:
                processed = True
            else:
                expected_wavs = [
                    listen_case_dir / "python_original.wav",
                    listen_case_dir / "python_onnx.wav",
                    listen_case_dir / "rust_onnx.wav",
                ]
                processed = all(p.exists() for p in expected_wavs)
            if processed:
                print(f"[resume] Skipping {case_id} (already processed)")
                failure_obj = json.loads(done_marker.read_text(encoding="utf-8"))
                fail_tags = failure_obj.get("fail_tags", [])
                thresholds_used = failure_obj.get("thresholds", {})
                rust_report = rust_root / f"{case_id}_rust_report.json"
                rust_tensor_pass = None
                if rust_report.exists():
                    rust_obj = json.loads(rust_report.read_text(encoding="utf-8"))
                    rust_tensor_pass = bool(rust_obj.get("verdict", {}).get("pass", False))
                summary["cases"].append(
                    {
                        "case_id": case_id,
                        "mode": mode,
                        "baseline_npz": str(baseline_npz.relative_to(project_root)).replace("\\", "/"),
                        "baseline_wav": str(baseline_wav.relative_to(project_root)).replace("\\", "/"),
                        "py_npz": str(py_npz.relative_to(project_root)).replace("\\", "/"),
                        "py_wav": str(py_wav.relative_to(project_root)).replace("\\", "/"),
                        "rust_npz": str(rust_npz.relative_to(project_root)).replace("\\", "/"),
                        "rust_wav": str(rust_wav.relative_to(project_root)).replace("\\", "/"),
                        "listening_dir": str(listen_case_dir.relative_to(project_root)).replace("\\", "/"),
                        "rust_tensor_pass": rust_tensor_pass,
                        "failure_tags": fail_tags,
                        "py_core_provider_used": failure_obj.get("py_core_provider_used", args.provider),
                        "py_audio_provider_used": failure_obj.get(
                            "py_audio_provider_used",
                            args.audio_provider or failure_obj.get("py_core_provider_used", args.provider),
                        ),
                        "thresholds": thresholds_used,
                    }
                )
                continue

        fg_cmd = [
            str(py),
            "-m",
            "runtime_py_ort.cli",
            "full-generate",
            "--case",
            str(case_path),
            "--inputs-npz",
            str(baseline_npz),
            "--onnx-dir",
            str(resolved_onnx_dir),
            "--provider",
            args.provider,
            "--audio-provider",
            (args.audio_provider or args.provider),
            "--out-dir",
            str(py_root),
            "--out-npz",
            str(py_npz),
            "--out-audio",
            str(py_wav),
            "--mode",
            mode,
            "--deterministic",
            "true",
        ]
        if args.skip_audio:
            fg_cmd.extend(["--skip-audio", "true"])
        if mode == "simple_mode":
            fg_cmd.extend(
                [
                    "--lm-model",
                    args.lm_model,
                    "--lm-constrained",
                    "true",
                    "--max-new-tokens",
                    "768",
                ]
            )
        fg_provider_idx = fg_cmd.index("--provider") + 1
        py_core_provider_used = _run_with_fallback(
            cmd=fg_cmd,
            cwd=project_root,
            case_id=case_id,
            label="python_full_generate",
            provider_value_index=fg_provider_idx,
            fallback_provider=args.fallback_provider if args.provider != args.fallback_provider else None,
        )
        py_audio_provider_used = args.audio_provider or py_core_provider_used

        if not args.skip_audio:
            py_parity_cmd = [
                str(py),
                "-m",
                "runtime_py_ort.cli",
                "full-parity",
                "--case",
                str(case_path),
                "--onnx-dir",
                str(resolved_onnx_dir),
                "--provider",
                py_core_provider_used,
                "--audio-provider",
                py_audio_provider_used,
                "--out-dir",
                str(py_root),
                "--baseline-npz",
                str(baseline_npz),
                "--candidate-npz",
                str(py_npz),
            ]
            py_parity_provider_idx = py_parity_cmd.index("--provider") + 1
            py_core_provider_used = _run_with_fallback(
                cmd=py_parity_cmd,
                cwd=project_root,
                case_id=case_id,
                label="python_full_parity",
                provider_value_index=py_parity_provider_idx,
                fallback_provider=args.fallback_provider if py_core_provider_used != args.fallback_provider else None,
            )
            py_audio_provider_used = args.audio_provider or py_core_provider_used

        rust_gen_cmd = [
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
            str(resolved_onnx_dir),
            "--out-npz",
            str(rust_npz),
            "--out-wav",
            str(rust_wav),
        ]
        if mode == "simple_mode":
            rust_gen_cmd.extend(
                [
                    "--online-lm-simple-mode",
                    "--lm-model",
                    args.lm_model,
                    "--lm-max-new-tokens",
                    "768",
                ]
            )
        _run(rust_gen_cmd, cwd=project_root)

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
        if not args.skip_audio:
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
                    str(resolved_onnx_dir),
                ],
                cwd=project_root,
            )

        tensor_bp = _compare_npz(baseline_npz, py_npz)
        tensor_br = _compare_npz(baseline_npz, rust_npz)
        tensor_pr = _compare_npz(py_npz, rust_npz)
        if args.skip_audio:
            audio_bp = {}
            audio_br = {}
            audio_pr = {}
        else:
            audio_bp = _audio_metrics(baseline_wav, py_wav)
            audio_br = _audio_metrics(baseline_wav, rust_wav)
            audio_pr = _audio_metrics(py_wav, rust_wav)

        overrides = case_thresholds.get(case_id, {})
        tensor_rmse_th = float(overrides.get("tensor_rmse", args.tensor_rmse_threshold))
        tensor_max_abs_th = float(overrides.get("tensor_max_abs", args.tensor_max_abs_threshold))
        audio_snr_th = float(overrides.get("audio_snr", args.audio_snr_threshold))
        audio_corr_th = float(overrides.get("audio_corr", args.audio_corr_threshold))

        _write_json(par3_root / f"{case_id}_tensor_baseline_vs_py.json", tensor_bp)
        _write_json(par3_root / f"{case_id}_tensor_baseline_vs_rust.json", tensor_br)
        _write_json(par3_root / f"{case_id}_tensor_py_vs_rust.json", tensor_pr)
        if not args.skip_audio:
            _write_json(par3_root / f"{case_id}_audio_baseline_vs_py.json", audio_bp)
            _write_json(par3_root / f"{case_id}_audio_baseline_vs_rust.json", audio_br)
            _write_json(par3_root / f"{case_id}_audio_py_vs_rust.json", audio_pr)

        fail_tags = _classify_fail_reasons(
            mode=mode,
            tensor_metrics=tensor_br["metrics"],
            audio_metrics=audio_br,
            tensor_rmse_th=tensor_rmse_th,
            tensor_max_abs_th=tensor_max_abs_th,
            audio_snr_th=audio_snr_th,
            audio_corr_th=audio_corr_th,
        )
        _write_json(
            par3_root / f"{case_id}_failure_tags.json",
            {
                "case_id": case_id,
                "mode": mode,
                "onnx_profile": args.onnx_profile,
                "py_core_provider_used": py_core_provider_used,
                "py_audio_provider_used": py_audio_provider_used,
                "fail_tags": fail_tags,
                "thresholds": {
                    "tensor_rmse": tensor_rmse_th,
                    "tensor_max_abs": tensor_max_abs_th,
                    "audio_snr": audio_snr_th,
                    "audio_corr": audio_corr_th,
                },
            },
        )

        listen_case_dir = listen_root / case_id
        if not args.skip_audio:
            listen_case_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(baseline_wav, listen_case_dir / "python_original.wav")
            shutil.copy2(py_wav, listen_case_dir / "python_onnx.wav")
            shutil.copy2(rust_wav, listen_case_dir / "rust_onnx.wav")

        rust_tensor_pass = None
        if rust_report.exists():
            rust_obj = json.loads(rust_report.read_text(encoding="utf-8"))
            rust_tensor_pass = bool(rust_obj.get("verdict", {}).get("pass", False))

        summary["cases"].append(
            {
                "case_id": case_id,
                "mode": mode,
                "baseline_npz": str(baseline_npz.relative_to(project_root)).replace("\\", "/"),
                "baseline_wav": str(baseline_wav.relative_to(project_root)).replace("\\", "/"),
                "py_npz": str(py_npz.relative_to(project_root)).replace("\\", "/"),
                "py_wav": str(py_wav.relative_to(project_root)).replace("\\", "/"),
                "rust_npz": str(rust_npz.relative_to(project_root)).replace("\\", "/"),
                "rust_wav": str(rust_wav.relative_to(project_root)).replace("\\", "/"),
                "listening_dir": str(listen_case_dir.relative_to(project_root)).replace("\\", "/"),
                "rust_tensor_pass": rust_tensor_pass,
                "failure_tags": fail_tags,
                "py_core_provider_used": py_core_provider_used,
                "py_audio_provider_used": py_audio_provider_used,
                "thresholds": {
                    "tensor_rmse": tensor_rmse_th,
                    "tensor_max_abs": tensor_max_abs_th,
                    "audio_snr": audio_snr_th,
                    "audio_corr": audio_corr_th,
                },
            }
        )

    summary_path = par3_root / "summary_all_cases.json"
    _write_json(summary_path, summary)
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
