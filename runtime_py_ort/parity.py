from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np


def tensor_metrics(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    da = a.astype(np.float64)
    db = b.astype(np.float64)
    diff = da - db
    rmse = float(np.sqrt(np.mean(diff * diff)))
    max_abs = float(np.max(np.abs(diff)))
    denom = float(np.linalg.norm(da.ravel()) * np.linalg.norm(db.ravel()))
    cos_sim = float(np.dot(da.ravel(), db.ravel()) / denom) if denom > 0 else 1.0
    return {"rmse": rmse, "max_abs": max_abs, "cos_sim": cos_sim}


def compare_npz(baseline: Path, candidate: Path) -> Dict[str, object]:
    a = np.load(str(baseline), allow_pickle=False)
    b = np.load(str(candidate), allow_pickle=False)
    common = sorted(set(a.files) & set(b.files))

    out = {"keys": common, "metrics": {}}
    for key in common:
        if a[key].shape == b[key].shape:
            out["metrics"][key] = tensor_metrics(a[key], b[key])
    return out


def write_report(report: Dict[str, object], out_json: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
