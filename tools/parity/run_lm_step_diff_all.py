#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def _run(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run lm_step_diff for all simple_mode fixtures")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--cases-dir", type=Path, default=Path("fixtures/cases"))
    parser.add_argument("--onnx-dir", type=Path, default=Path("artifacts/onnx_runtime"))
    parser.add_argument("--variant", choices=["0.6B", "1.7B"], default="1.7B")
    parser.add_argument("--out-dir", type=Path, default=Path("reports/parity_3way/lm_step_diff"))
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--constrained", action="store_true")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    cases = sorted((project_root / args.cases_dir).glob("simple_mode_*.json"))
    if not cases:
        raise RuntimeError(f"no simple_mode cases found in {args.cases_dir}")

    py = project_root / ".venv/Scripts/python.exe"
    if not py.exists():
        py = Path("python")

    out_dir = (project_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "variant": args.variant,
        "constrained": bool(args.constrained),
        "cases": [],
    }

    for case in cases:
        out_json = out_dir / f"{case.stem}_pt_vs_onnx_lm_step.json"
        cmd = [
            str(py),
            "tools/parity/lm_step_diff.py",
            "--case",
            str(case),
            "--onnx-dir",
            str(args.onnx_dir),
            "--variant",
            args.variant,
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--out-json",
            str(out_json),
            "--device",
            args.device,
        ]
        if args.constrained:
            cmd.append("--constrained")
        _run(cmd, cwd=project_root)

        rep = json.loads(out_json.read_text(encoding="utf-8"))
        summary["cases"].append(
            {
                "case_id": rep.get("case_id", case.stem),
                "steps": rep.get("steps"),
                "first_token_divergence_step": rep.get("first_token_divergence_step"),
                "token_exact_match": rep.get("token_exact_match"),
                "report": str(out_json.relative_to(project_root)).replace("\\", "/"),
            }
        )

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
