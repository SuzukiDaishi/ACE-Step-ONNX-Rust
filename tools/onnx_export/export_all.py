#!/usr/bin/env python3
"""Export all ACE-Step ONNX artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def _run(cmd: list[str], workdir: Path) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=str(workdir), check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export all ACE-Step models to ONNX")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/onnx_runtime"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--provider", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--external-data-policy", choices=["auto", "single", "external"], default="auto")
    parser.add_argument("--skip-sanity", action="store_true")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    out_dir = args.out_dir
    manifest = args.manifest or (out_dir / "manifest.json")
    out_dir.mkdir(parents=True, exist_ok=True)

    common_args = [
        "--project-root",
        str(project_root),
        "--out-dir",
        str(out_dir),
        "--manifest",
        str(manifest),
        "--provider",
        args.provider,
        "--opset",
        str(args.opset),
        "--external-data-policy",
        args.external_data_policy,
    ]
    if args.skip_sanity:
        common_args.append("--skip-sanity")

    py = sys.executable
    _run([py, str(project_root / "tools/onnx_export/export_core.py"), *common_args], project_root)
    _run([py, str(project_root / "tools/onnx_export/export_qwen_embedding.py"), *common_args], project_root)
    _run([py, str(project_root / "tools/onnx_export/export_lm.py"), *common_args, "--variant", "0.6B"], project_root)
    _run([py, str(project_root / "tools/onnx_export/export_lm.py"), *common_args, "--variant", "1.7B"], project_root)

    print("All exports completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

