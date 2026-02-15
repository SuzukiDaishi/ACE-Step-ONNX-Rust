from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import onnxruntime as ort
import soundfile as sf

from runtime_py_ort.case_schema import CaseSpec
from runtime_py_ort.scheduler import resolve_timesteps


def _session(path: Path, provider: str) -> ort.InferenceSession:
    providers = ["CPUExecutionProvider"]
    if provider == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    try:
        return ort.InferenceSession(str(path), providers=providers)
    except Exception:
        if provider != "cuda":
            raise
        return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def _load_contract(onnx_dir: Path) -> Dict[str, Dict[str, List[str]]]:
    contract_path = onnx_dir / "io_contract_core.json"
    if not contract_path.exists():
        raise FileNotFoundError(f"missing contract: {contract_path}")
    return json.loads(contract_path.read_text(encoding="utf-8"))


def _crop_vt(vt: np.ndarray, orig_len: int) -> np.ndarray:
    if vt.shape[1] == orig_len:
        return vt.astype(np.float32)
    return vt[:, :orig_len, :].astype(np.float32)


def _dll_bind(lib: ctypes.CDLL) -> None:
    lib.ace_create_context.argtypes = [ctypes.c_char_p]
    lib.ace_create_context.restype = ctypes.c_void_p
    lib.ace_free_context.argtypes = [ctypes.c_void_p]
    lib.ace_free_context.restype = None
    lib.ace_string_free.argtypes = [ctypes.c_void_p]
    lib.ace_string_free.restype = None
    lib.ace_last_error.argtypes = []
    lib.ace_last_error.restype = ctypes.c_void_p
    lib.ace_prepare_step_inputs.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.ace_prepare_step_inputs.restype = ctypes.c_int32
    lib.ace_scheduler_step.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.ace_scheduler_step.restype = ctypes.c_int32
    lib.ace_apply_lm_constraints.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.ace_apply_lm_constraints.restype = ctypes.c_int32
    lib.ace_finalize_metadata.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.ace_finalize_metadata.restype = ctypes.c_int32


def _dll_last_error(lib: ctypes.CDLL) -> str:
    ptr = lib.ace_last_error()
    if not ptr:
        return "unknown"
    try:
        raw = ctypes.cast(ptr, ctypes.c_char_p).value
        return (raw or b"unknown").decode("utf-8", errors="replace")
    finally:
        lib.ace_string_free(ptr)


def _dll_prepare_step(lib: ctypes.CDLL, ctx: int, shift: float, steps: int, step: int, xt: np.ndarray) -> None:
    payload = {"shift": float(shift), "inference_steps": int(steps), "current_step": int(step)}
    out_json = ctypes.c_void_p()
    buf = np.ascontiguousarray(xt.reshape(-1), dtype=np.float32)
    rc = lib.ace_prepare_step_inputs(
        ctx,
        json.dumps(payload).encode("utf-8"),
        buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_size_t(buf.size),
        ctypes.byref(out_json),
    )
    if rc != 0:
        raise RuntimeError(f"ace_prepare_step_inputs failed: {_dll_last_error(lib)}")
    lib.ace_string_free(out_json)


def _dll_scheduler_step(lib: ctypes.CDLL, ctx: int, xt: np.ndarray, vt: np.ndarray, dt: float) -> np.ndarray:
    x = np.ascontiguousarray(xt.reshape(-1), dtype=np.float32)
    v = np.ascontiguousarray(vt.reshape(-1), dtype=np.float32)
    out = np.empty_like(x)
    rc = lib.ace_scheduler_step(
        ctx,
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        v.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_size_t(x.size),
        ctypes.c_float(dt),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    if rc != 0:
        raise RuntimeError(f"ace_scheduler_step failed: {_dll_last_error(lib)}")
    return out.reshape(xt.shape)


def _run_condition(
    sess: ort.InferenceSession,
    arr: Dict[str, np.ndarray],
    contract_inputs: List[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if all(k in arr for k in ("encoder_hidden_states", "encoder_attention_mask", "context_latents")):
        return (
            arr["encoder_hidden_states"].astype(np.float32),
            arr["encoder_attention_mask"].astype(np.float32),
            arr["context_latents"].astype(np.float32),
        )

    feeds: Dict[str, np.ndarray] = {}
    for name in contract_inputs:
        if name not in arr:
            raise ValueError(f"inputs npz missing condition input: {name}")
        value = arr[name]
        if name == "refer_audio_order_mask":
            feeds[name] = value.astype(np.int64, copy=False)
        elif name == "is_covers":
            feeds[name] = value.astype(bool, copy=False)
        else:
            feeds[name] = value.astype(np.float32, copy=False)
    out_names = [x.name for x in sess.get_outputs()]
    out = sess.run(out_names, feeds)
    out_map = {k: v for k, v in zip(out_names, out)}
    return (
        out_map["encoder_hidden_states"].astype(np.float32),
        out_map["encoder_attention_mask"].astype(np.float32),
        out_map["context_latents"].astype(np.float32),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Python ONNXRuntime + Rust DLL demo")
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--inputs-npz", type=Path)
    parser.add_argument("--onnx-dir", type=Path, default=Path("artifacts/onnx_runtime"))
    parser.add_argument("--dll", type=Path, default=Path("runtime_rust_dll/target/release/acestep_runtime.dll"))
    parser.add_argument("--provider", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--out-wav", type=Path, default=Path("reports/listening/demo/python_ort_dll.wav"))
    parser.add_argument("--out-npz", type=Path, default=Path("reports/listening/demo/python_ort_dll.npz"))
    args = parser.parse_args()

    case = CaseSpec.from_path(args.case)
    inputs_npz = args.inputs_npz or Path(f"fixtures/tensors/{case.case_id}.npz")
    npz = np.load(str(inputs_npz), allow_pickle=False)
    arr = {k: npz[k] for k in npz.files}
    onnx_dir = args.onnx_dir
    contract = _load_contract(onnx_dir)

    lib = ctypes.CDLL(str(args.dll.resolve()))
    _dll_bind(lib)
    ctx = lib.ace_create_context(json.dumps({"seed": int(case.seed)}).encode("utf-8"))
    if not ctx:
        raise RuntimeError(f"ace_create_context failed: {_dll_last_error(lib)}")

    try:
        condition_sess = _session(onnx_dir / "condition_encoder.onnx", args.provider)
        encoder_hidden_states, encoder_attention_mask, context_latents = _run_condition(
            condition_sess,
            arr,
            contract.get("inputs", {}).get("condition_encoder", []),
        )

        src_latents = arr["src_latents"].astype(np.float32)
        xt = arr["xt_steps"][0].astype(np.float32).copy() if "xt_steps" in arr else src_latents.copy()
        latent_masks = arr.get("latent_masks")
        attention_mask = latent_masks.astype(np.float32) if latent_masks is not None else np.ones((xt.shape[0], xt.shape[1]), dtype=np.float32)

        orig_len = int(xt.shape[1])
        pad_len = (-orig_len) % 2
        context_latents_padded = (
            np.pad(context_latents, ((0, 0), (0, pad_len), (0, 0)), mode="constant") if pad_len else context_latents
        )
        attention_mask_padded = (
            np.pad(attention_mask, ((0, 0), (0, pad_len)), mode="constant") if pad_len else attention_mask
        )
        timesteps = resolve_timesteps(case.shift, None, max_steps=max(1, int(case.inference_steps)))

        has_kv = (onnx_dir / "dit_prefill_kv.onnx").exists() and (onnx_dir / "dit_decode_kv.onnx").exists()
        if has_kv:
            dit_prefill = _session(onnx_dir / "dit_prefill_kv.onnx", args.provider)
            dit_decode = _session(onnx_dir / "dit_decode_kv.onnx", args.provider)
            prefill_inputs = [x.name for x in dit_prefill.get_inputs()]
            prefill_outputs = [x.name for x in dit_prefill.get_outputs()]
            decode_inputs = [x.name for x in dit_decode.get_inputs()]
            decode_outputs = [x.name for x in dit_decode.get_outputs()]
            cache_map: Dict[str, np.ndarray] = {}
        else:
            dit = _session(onnx_dir / "dit_decoder.onnx", args.provider)
            dit_inputs = {x.name for x in dit.get_inputs()}

        xt_steps: List[np.ndarray] = []
        vt_steps: List[np.ndarray] = []
        for idx, t in enumerate(timesteps):
            _dll_prepare_step(lib, ctx, case.shift, len(timesteps), idx, xt)
            t_vec = np.full((xt.shape[0],), t, dtype=np.float32)
            xt_in = np.pad(xt, ((0, 0), (0, pad_len), (0, 0)), mode="constant") if pad_len else xt

            if has_kv:
                base = {
                    "hidden_states": xt_in,
                    "timestep": t_vec,
                    "timestep_r": t_vec,
                    "attention_mask": attention_mask_padded,
                    "encoder_hidden_states": encoder_hidden_states,
                    "encoder_attention_mask": encoder_attention_mask,
                    "context_latents": context_latents_padded,
                }
                if idx == 0:
                    feeds = {k: base[k] for k in prefill_inputs if k in base}
                    out = dit_prefill.run(prefill_outputs, feeds)
                    out_map = {k: v for k, v in zip(prefill_outputs, out)}
                else:
                    feeds = {}
                    for name in decode_inputs:
                        if name.startswith("past_"):
                            present_name = "present_" + name[len("past_") :]
                            if present_name not in cache_map:
                                raise ValueError(f"missing cache for {present_name}")
                            feeds[name] = cache_map[present_name]
                        elif name in base:
                            feeds[name] = base[name]
                    out = dit_decode.run(decode_outputs, feeds)
                    out_map = {k: v for k, v in zip(decode_outputs, out)}
                vt = _crop_vt(out_map["vt"], orig_len)
                cache_map = {k: v.astype(np.float32) for k, v in out_map.items() if k.startswith("present_")}
            else:
                feeds = {
                    "hidden_states": xt_in,
                    "timestep": t_vec,
                    "timestep_r": t_vec,
                    "encoder_hidden_states": encoder_hidden_states,
                    "context_latents": context_latents_padded,
                }
                if "attention_mask" in dit_inputs:
                    feeds["attention_mask"] = attention_mask_padded
                if "encoder_attention_mask" in dit_inputs:
                    feeds["encoder_attention_mask"] = encoder_attention_mask
                vt = _crop_vt(dit.run(["vt"], feeds)[0], orig_len)

            xt_steps.append(xt.copy())
            vt_steps.append(vt.copy())
            dt = float(t) if idx == len(timesteps) - 1 else float(t - timesteps[idx + 1])
            xt = _dll_scheduler_step(lib, ctx, xt, vt, dt)

        pred_latents = xt
        vae = _session(onnx_dir / "vae_decoder.onnx", args.provider)
        latents = np.transpose(pred_latents, (0, 2, 1)).astype(np.float32)
        audio = vae.run(["audio"], {"latents": latents})[0].astype(np.float32)
        audio_0 = audio[0]
        args.out_wav.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(args.out_wav), audio_0.T, 48_000, subtype="FLOAT")

        args.out_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(args.out_npz),
            pred_latents=pred_latents.astype(np.float32),
            xt_steps=np.asarray(xt_steps, dtype=np.float32),
            vt_steps=np.asarray(vt_steps, dtype=np.float32),
            encoder_hidden_states=encoder_hidden_states.astype(np.float32),
            encoder_attention_mask=encoder_attention_mask.astype(np.float32),
            context_latents=context_latents.astype(np.float32),
        )

        # Demonstrate LM constraint / metadata APIs for simple_mode use.
        logits = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
        masked = np.empty_like(logits)
        _ = lib.ace_apply_lm_constraints(
            ctx,
            logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_size_t(logits.size),
            masked.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        token_ids = np.array([1, 2, 3], dtype=np.int64)
        out_json = ctypes.c_void_p()
        rc = lib.ace_finalize_metadata(
            ctx,
            token_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            ctypes.c_size_t(token_ids.size),
            ctypes.byref(out_json),
        )
        if rc == 0:
            lib.ace_string_free(out_json)

        print(f"Wrote wav: {args.out_wav}")
        print(f"Wrote npz: {args.out_npz}")
        return 0
    finally:
        lib.ace_free_context(ctx)


if __name__ == "__main__":
    raise SystemExit(main())
