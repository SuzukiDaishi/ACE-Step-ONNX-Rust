#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy_contracts(src_dir: Path, dst_dir: Path) -> None:
    for p in src_dir.rglob("*.json"):
        rel = p.relative_to(src_dir)
        out = dst_dir / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out)


def _copy_sidecars(src_onnx: Path, dst_onnx: Path) -> None:
    copied = False
    try:
        import onnx  # type: ignore

        model = onnx.load(str(src_onnx), load_external_data=False)
        locations: set[str] = set()
        for tensor in model.graph.initializer:
            for kv in tensor.external_data:
                if kv.key == "location" and kv.value:
                    locations.add(kv.value)
        for rel in sorted(locations):
            src_side = (src_onnx.parent / rel).resolve()
            if not src_side.exists():
                continue
            out_side = (dst_onnx.parent / rel).resolve()
            out_side.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_side, out_side)
            copied = True
    except Exception:
        copied = False

    if copied:
        return

    base = src_onnx.with_suffix("")
    src_candidates = list(src_onnx.parent.glob(f"{src_onnx.name}.data")) + list(src_onnx.parent.glob(f"{base.name}*"))
    for side in src_candidates:
        if side == src_onnx:
            continue
        if side.suffix in {".onnx", ".json"}:
            continue
        out = dst_onnx.parent / side.name
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(side, out)


def _validate_onnx(path: Path) -> tuple[bool, str]:
    try:
        import onnx  # type: ignore
    except Exception as e:
        return False, f"onnx package unavailable ({e})"
    try:
        model = onnx.load(str(path), load_external_data=True)
        onnx.checker.check_model(model)
        return True, "ok"
    except Exception as e:
        try:
            # Path-based checker handles large external-data models better.
            onnx.checker.check_model(str(path))
            return True, "ok(path-check)"
        except Exception:
            return False, str(e)


def _convert_or_copy(profile: str, src: Path, dst: Path) -> tuple[bool, str]:
    try:
        converted, note = _convert_one(profile, src, dst)
    except Exception as e:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        _copy_sidecars(src, dst)
        return False, f"conversion failed, copied original ({e})"

    valid, reason = _validate_onnx(dst)
    if valid:
        return converted, note

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    _copy_sidecars(src, dst)
    return False, f"{note}; invalid converted model, copied original ({reason})"


def _quantize_fp16(src: Path, dst: Path) -> tuple[bool, str]:
    try:
        import onnx  # type: ignore
        from onnxruntime.transformers.float16 import convert_float_to_float16  # type: ignore
    except Exception as e:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        _copy_sidecars(src, dst)
        return False, f"fp16 converter unavailable, copied original ({e})"

    model = onnx.load(str(src), load_external_data=True)
    fp16_model = convert_float_to_float16(model, keep_io_types=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(fp16_model, str(dst), save_as_external_data=False)
    return True, "converted"


def _quantize_int8_dynamic(src: Path, dst: Path) -> tuple[bool, str]:
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic  # type: ignore
    except Exception as e:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        _copy_sidecars(src, dst)
        return False, f"onnxruntime.quantization unavailable, copied original ({e})"

    dst.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        model_input=str(src),
        model_output=str(dst),
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
    )
    return True, "converted"


def _quantize_int8_static(src: Path, dst: Path) -> tuple[bool, str]:
    # Static quantization needs calibration data per-model and is repository-specific.
    # We intentionally fallback to copy when no calibrator is provided.
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    _copy_sidecars(src, dst)
    return False, "no calibration dataset configured, copied original"


def _convert_one(profile: str, src: Path, dst: Path) -> tuple[bool, str]:
    if profile == "fp16":
        return _quantize_fp16(src, dst)
    if profile == "int8_dynamic":
        return _quantize_int8_dynamic(src, dst)
    if profile == "int8_static":
        return _quantize_int8_static(src, dst)
    raise ValueError(f"unknown profile: {profile}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate optimized ONNX profiles (fp16/int8)")
    parser.add_argument("--src-dir", type=Path, default=Path("artifacts/onnx_runtime"))
    parser.add_argument("--out-root", type=Path, default=Path("artifacts/onnx_runtime_optimized"))
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=["fp16", "int8_dynamic", "int8_static"],
        default=["fp16", "int8_dynamic", "int8_static"],
    )
    args = parser.parse_args()

    src_dir = args.src_dir.resolve()
    out_root = args.out_root.resolve()
    if not src_dir.exists():
        raise FileNotFoundError(f"source onnx dir not found: {src_dir}")

    onnx_files = sorted([p for p in src_dir.rglob("*.onnx") if ".optimized" not in str(p)])
    if not onnx_files:
        raise RuntimeError(f"no onnx files found in {src_dir}")

    all_manifest: dict[str, Any] = {"source_dir": str(src_dir), "profiles": {}}

    for profile in args.profiles:
        profile_dir = out_root / profile
        profile_dir.mkdir(parents=True, exist_ok=True)
        _copy_contracts(src_dir, profile_dir)
        model_rows: list[dict[str, Any]] = []

        for src in onnx_files:
            rel = src.relative_to(src_dir)
            dst = profile_dir / rel
            converted, note = _convert_or_copy(profile, src, dst)
            row = {
                "profile": profile,
                "model_relpath": str(rel).replace("\\", "/"),
                "source_sha256": _sha256(src),
                "output_sha256": _sha256(dst),
                "source_bytes": src.stat().st_size,
                "output_bytes": dst.stat().st_size,
                "converted": bool(converted),
                "note": note,
            }
            model_rows.append(row)
            print(f"[{profile}] {rel} -> {note}")

        manifest = {
            "profile": profile,
            "source_dir": str(src_dir),
            "models": model_rows,
            "metrics": {
                "tensor_rmse": None,
                "audio_snr": None,
                "audio_corr": None,
            },
        }
        (profile_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        all_manifest["profiles"][profile] = {
            "dir": str(profile_dir),
            "manifest": str((profile_dir / "manifest.json")),
            "models": len(model_rows),
        }

    (out_root / "manifest.json").write_text(json.dumps(all_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {out_root / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
