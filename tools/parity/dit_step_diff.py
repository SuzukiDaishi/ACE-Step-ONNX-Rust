#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    da = a.astype(np.float64, copy=False)
    db = b.astype(np.float64, copy=False)
    diff = da - db
    rmse = float(np.sqrt(np.mean(diff * diff)))
    max_abs = float(np.max(np.abs(diff)))
    denom = float(np.linalg.norm(da.ravel()) * np.linalg.norm(db.ravel()))
    cos_sim = float(np.dot(da.ravel(), db.ravel()) / denom) if denom > 0 else 1.0
    return {"rmse": rmse, "max_abs": max_abs, "cos_sim": cos_sim}


def _per_step_metrics(
    a: np.ndarray,
    b: np.ndarray,
    rmse_threshold: float,
    max_abs_threshold: float,
) -> dict[str, Any]:
    if a.shape != b.shape:
        return {
            "shape_match": False,
            "shape_a": list(a.shape),
            "shape_b": list(b.shape),
            "first_divergence_step": None,
            "steps": [],
        }
    if a.ndim < 1:
        return {"shape_match": True, "shape_a": list(a.shape), "first_divergence_step": None, "steps": []}

    steps: list[dict[str, Any]] = []
    first_div: int | None = None
    for idx in range(a.shape[0]):
        m = _metrics(a[idx], b[idx])
        fail = m["rmse"] > rmse_threshold or m["max_abs"] > max_abs_threshold
        if first_div is None and fail:
            first_div = idx
        steps.append({"step": idx, **m, "fail": fail})
    return {
        "shape_match": True,
        "shape": list(a.shape),
        "first_divergence_step": first_div,
        "steps": steps,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="DiT step-level parity diff for vt_steps/xt_steps")
    parser.add_argument("--baseline-npz", type=Path, required=True)
    parser.add_argument("--candidate-npz", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--rmse-threshold", type=float, default=5e-4)
    parser.add_argument("--max-abs-threshold", type=float, default=1e-1)
    args = parser.parse_args()

    base = np.load(args.baseline_npz, allow_pickle=False)
    cand = np.load(args.candidate_npz, allow_pickle=False)

    result: dict[str, Any] = {
        "baseline_npz": str(args.baseline_npz),
        "candidate_npz": str(args.candidate_npz),
        "thresholds": {
            "rmse": float(args.rmse_threshold),
            "max_abs": float(args.max_abs_threshold),
        },
        "keys": {},
    }

    for key in ["vt_steps", "xt_steps", "pred_latents"]:
        if key not in base.files or key not in cand.files:
            result["keys"][key] = {"present": False}
            continue
        a = base[key]
        b = cand[key]
        if key in {"vt_steps", "xt_steps"}:
            detail = _per_step_metrics(a, b, args.rmse_threshold, args.max_abs_threshold)
            summary = _metrics(a, b) if a.shape == b.shape else None
            result["keys"][key] = {
                "present": True,
                "summary": summary,
                "per_step": detail,
            }
        else:
            result["keys"][key] = {
                "present": True,
                "summary": _metrics(a, b) if a.shape == b.shape else None,
                "shape_a": list(a.shape),
                "shape_b": list(b.shape),
            }

    fail_tags: list[str] = []
    vt = result["keys"].get("vt_steps", {})
    xt = result["keys"].get("xt_steps", {})
    pl = result["keys"].get("pred_latents", {})
    for k in [vt, xt, pl]:
        s = k.get("summary")
        if isinstance(s, dict):
            if s.get("rmse", 0.0) > args.rmse_threshold or s.get("max_abs", 0.0) > args.max_abs_threshold:
                fail_tags.append("DiT")
                break
    result["fail_tags"] = sorted(set(fail_tags))

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
