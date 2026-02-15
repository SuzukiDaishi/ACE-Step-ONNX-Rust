#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_mean(values: list[float]) -> float | None:
    return float(mean(values)) if values else None


def _safe_min(values: list[float]) -> float | None:
    return float(min(values)) if values else None


def _safe_max(values: list[float]) -> float | None:
    return float(max(values)) if values else None


def _safe_p95(values: list[float]) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    idx = min(len(xs) - 1, int(round(0.95 * (len(xs) - 1))))
    return float(xs[idx])


def _collect_profile_metrics(report_dir: Path) -> dict[str, Any]:
    parity_dir = report_dir / "parity_3way"
    summary_path = parity_dir / "summary_all_cases.json"
    if not summary_path.exists():
        return {
            "cases_total": 0,
            "cases_pass": 0,
            "case_pass_rate": 0.0,
            "tensor_rmse": None,
            "audio_snr": None,
            "audio_corr": None,
        }

    summary = _load_json(summary_path)
    cases = list(summary.get("cases", []))
    cases_total = len(cases)
    cases_pass = sum(1 for c in cases if bool(c.get("rust_tensor_pass", False)))

    rmse_values: list[float] = []
    max_abs_values: list[float] = []
    snr_values: list[float] = []
    corr_values: list[float] = []

    for case in cases:
        case_id = str(case.get("case_id"))
        tensor_path = parity_dir / f"{case_id}_tensor_baseline_vs_rust.json"
        audio_path = parity_dir / f"{case_id}_audio_baseline_vs_rust.json"
        if tensor_path.exists():
            t = _load_json(tensor_path).get("metrics", {})
            for key in ("pred_latents", "xt_steps", "vt_steps"):
                m = t.get(key)
                if isinstance(m, dict):
                    if "rmse" in m:
                        rmse_values.append(float(m["rmse"]))
                    if "max_abs" in m:
                        max_abs_values.append(float(m["max_abs"]))
        if audio_path.exists():
            a = _load_json(audio_path).get("metrics", {})
            if "snr" in a:
                snr_values.append(float(a["snr"]))
            if "corr" in a:
                corr_values.append(float(a["corr"]))

    pass_rate = (cases_pass / cases_total) if cases_total > 0 else 0.0
    return {
        "cases_total": cases_total,
        "cases_pass": cases_pass,
        "case_pass_rate": pass_rate,
        "tensor_rmse": _safe_mean(rmse_values),
        "tensor_rmse_p95": _safe_p95(rmse_values),
        "tensor_max_abs_max": _safe_max(max_abs_values),
        "audio_snr": _safe_mean(snr_values),
        "audio_snr_min": _safe_min(snr_values),
        "audio_corr": _safe_mean(corr_values),
        "audio_corr_min": _safe_min(corr_values),
    }


def _update_profile_manifest(manifest_path: Path, metrics: dict[str, Any]) -> None:
    if not manifest_path.exists():
        return
    obj = _load_json(manifest_path)
    obj["metrics"] = metrics
    manifest_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fill quantized profile manifests with parity metrics")
    parser.add_argument("--manifest-root", type=Path, default=Path("artifacts/onnx_runtime_optimized"))
    parser.add_argument("--reports-root", type=Path, default=Path("reports/profiles"))
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=["fp16", "int8_dynamic", "int8_static"],
        default=["fp16", "int8_dynamic", "int8_static"],
    )
    args = parser.parse_args()

    manifest_root = args.manifest_root.resolve()
    reports_root = args.reports_root.resolve()

    for profile in args.profiles:
        profile_manifest = manifest_root / profile / "manifest.json"
        profile_report_dir = reports_root / profile
        metrics = _collect_profile_metrics(profile_report_dir)
        _update_profile_manifest(profile_manifest, metrics)
        print(f"[{profile}] metrics updated: {profile_manifest}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
