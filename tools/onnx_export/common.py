from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Callable


EXTERNAL_DATA_POLICY_VALUES = {"auto", "single", "external"}


@dataclass
class ExportResult:
    model_path: Path
    external_data: bool
    external_files: list[Path]
    input_names: list[str]
    output_names: list[str]
    sha256: str
    size_bytes: int


def ensure_real_onnx_module() -> None:
    """
    Ensure `import onnx` resolves to site-packages instead of local artifact folders.
    """
    cwd = str(Path.cwd())
    removed: list[str] = []
    for entry in ("", cwd):
        if entry in sys.path:
            sys.path.remove(entry)
            removed.append(entry)
    try:
        sys.modules.pop("onnx", None)
        try:
            import onnx as _onnx  # noqa: F401
        except ModuleNotFoundError as exc:
            raise RuntimeError("`onnx` is required. Install it with `uv pip install onnx`.") from exc
    finally:
        for entry in reversed(removed):
            sys.path.insert(0, entry)


def normalize_external_data_policy(policy: str) -> str:
    normalized = policy.strip().lower()
    if normalized not in EXTERNAL_DATA_POLICY_VALUES:
        allowed = ", ".join(sorted(EXTERNAL_DATA_POLICY_VALUES))
        raise ValueError(f"Invalid external data policy: {policy!r}. Allowed: {allowed}")
    return normalized


def _cleanup_export_files(model_path: Path) -> None:
    if model_path.exists():
        try:
            for f in _external_files_from_onnx(model_path):
                if f.exists() and f.is_file():
                    f.unlink()
        except Exception:
            pass
    if model_path.exists():
        model_path.unlink()
    for f in model_path.parent.glob(f"{model_path.name}*"):
        if f != model_path and f.is_file():
            f.unlink()


def _external_files_from_onnx(model_path: Path) -> list[Path]:
    import onnx

    model = onnx.load(str(model_path), load_external_data=False)
    out: set[Path] = set()
    for tensor in model.graph.initializer:
        if tensor.data_location != onnx.TensorProto.EXTERNAL:
            continue
        for kv in tensor.external_data:
            if kv.key == "location":
                out.add(model_path.parent / kv.value)
    return sorted(out)


def _repack_external_data_single_file(model_path: Path) -> None:
    import onnx

    external_files = _external_files_from_onnx(model_path)
    if not external_files:
        return
    model = onnx.load(str(model_path), load_external_data=True)
    location = f"{model_path.name}.data"
    onnx.save_model(
        model,
        str(model_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=location,
        size_threshold=0,
        convert_attribute=False,
    )
    # Remove stale shard files after repacking.
    for f in external_files:
        if f.name == location:
            continue
        if f.exists() and f.is_file():
            f.unlink()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as r:
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def onnx_io_names(model_path: Path) -> tuple[list[str], list[str]]:
    import onnx

    model = onnx.load(str(model_path), load_external_data=False)
    initializer_names = {x.name for x in model.graph.initializer}
    inputs = [x.name for x in model.graph.input if x.name not in initializer_names]
    outputs = [x.name for x in model.graph.output]
    return inputs, outputs


def export_with_policy(
    model_path: Path,
    policy: str,
    export_fn: Callable[[bool], None],
) -> ExportResult:
    normalized = normalize_external_data_policy(policy)
    attempts = {
        "single": [False],
        "external": [True],
        "auto": [False, True],
    }[normalized]

    last_error: Exception | None = None
    for use_external_data in attempts:
        before_files = {p for p in model_path.parent.iterdir() if p.is_file()}
        _cleanup_export_files(model_path)
        try:
            export_fn(use_external_data)
            if not model_path.exists():
                raise RuntimeError(f"Exporter did not create model file: {model_path}")
            _repack_external_data_single_file(model_path)
            inputs, outputs = onnx_io_names(model_path)
            external_files = _external_files_from_onnx(model_path)
            missing_external = [p for p in external_files if not p.exists()]
            if missing_external:
                missing_list = ", ".join(str(p) for p in missing_external[:5])
                raise RuntimeError(f"missing external data files for {model_path}: {missing_list}")
            if not external_files:
                after_files = {p for p in model_path.parent.iterdir() if p.is_file()}
                external_files = sorted([p for p in (after_files - before_files) if p != model_path])
            return ExportResult(
                model_path=model_path,
                external_data=bool(external_files) or use_external_data,
                external_files=external_files,
                input_names=inputs,
                output_names=outputs,
                sha256=_sha256(model_path),
                size_bytes=model_path.stat().st_size,
            )
        except Exception as exc:  # noqa: BLE001
            after_files = {p for p in model_path.parent.iterdir() if p.is_file()}
            for p in (after_files - before_files):
                if p.is_file():
                    p.unlink()
            last_error = exc

    if last_error is None:
        raise RuntimeError(f"Failed to export model: {model_path}")
    raise RuntimeError(f"Failed to export model: {model_path}") from last_error


def ort_sanity_load(model_path: Path, provider: str = "cpu") -> tuple[bool, str]:
    try:
        import onnxruntime as ort
    except Exception as exc:  # noqa: BLE001
        return False, f"onnxruntime not available: {exc}"

    providers = ["CPUExecutionProvider"]
    if provider.lower() == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    try:
        ort.InferenceSession(str(model_path), providers=providers)
        return True, "ok"
    except Exception as exc:  # noqa: BLE001
        if provider.lower() == "cuda":
            try:
                ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
                return True, "ok (cuda failed, cpu fallback)"
            except Exception as cpu_exc:  # noqa: BLE001
                return False, f"cuda failed ({exc}); cpu failed ({cpu_exc})"
        return False, str(exc)


def _rel(path: Path, base_dir: Path) -> str:
    try:
        return str(path.relative_to(base_dir)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def update_manifest(
    manifest_path: Path,
    out_dir: Path,
    entries: list[ExportResult],
    *,
    model_group: str,
) -> dict[str, Any]:
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {}
    manifest.setdefault("generated_at", datetime.now(timezone.utc).isoformat())
    manifest.setdefault("artifact_root", _rel(out_dir, out_dir))
    manifest.setdefault("models", {})
    model_section = manifest["models"].setdefault(model_group, {})

    for entry in entries:
        key = _rel(entry.model_path, out_dir)
        model_section[key] = {
            "path": key,
            "external_data": entry.external_data,
            "external_files": [_rel(p, out_dir) for p in entry.external_files],
            "inputs": entry.input_names,
            "outputs": entry.output_names,
            "sha256": entry.sha256,
            "size_bytes": entry.size_bytes,
        }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
