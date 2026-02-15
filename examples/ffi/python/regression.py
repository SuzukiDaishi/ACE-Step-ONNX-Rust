from __future__ import annotations

import ctypes
import json
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DLL = ROOT / "runtime_rust_dll" / "target" / "release" / "acestep_runtime.dll"
DLL_PATH = Path(os.environ.get("ACESTEP_RUNTIME_DLL", str(DEFAULT_DLL)))


def _expect(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def _decode_json_ptr(lib: ctypes.CDLL, ptr: int) -> dict:
    try:
        raw = ctypes.cast(ptr, ctypes.c_char_p).value
        _expect(raw is not None, "null json pointer")
        return json.loads(raw.decode("utf-8"))
    finally:
        lib.ace_string_free(ptr)


def _last_error(lib: ctypes.CDLL) -> str:
    ptr = lib.ace_last_error()
    if not ptr:
        return "unknown"
    try:
        raw = ctypes.cast(ptr, ctypes.c_char_p).value
        return (raw or b"unknown").decode("utf-8", errors="replace")
    finally:
        lib.ace_string_free(ptr)


def main() -> int:
    lib = ctypes.CDLL(str(DLL_PATH))

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

    cfg = {"seed": 42, "blocked_token_ids": [1, 3], "forced_token_id": 2}
    ctx = lib.ace_create_context(json.dumps(cfg).encode("utf-8"))
    _expect(bool(ctx), "ace_create_context failed")
    try:
        state = {"shift": 3.0, "inference_steps": 8, "current_step": 0}
        in_buf = (ctypes.c_float * 4)(1.0, 2.0, 3.0, 4.0)
        out_json = ctypes.c_void_p()
        rc = lib.ace_prepare_step_inputs(
            ctx,
            json.dumps(state).encode("utf-8"),
            in_buf,
            4,
            ctypes.byref(out_json),
        )
        _expect(rc == 0, f"ace_prepare_step_inputs failed: {_last_error(lib)}")
        payload = _decode_json_ptr(lib, out_json.value)
        _expect(payload["seed"] == 42, "seed mismatch")
        _expect(payload["inference_steps"] == 8, "inference_steps mismatch")
        _expect(abs(payload["timestep"] - 1.0) < 1e-7, "timestep mismatch")
        _expect(abs(payload["next_timestep"] - (0.875 ** 3)) < 1e-7, "next_timestep mismatch")

        xt = (ctypes.c_float * 4)(1.0, 1.0, 1.0, 1.0)
        vt = (ctypes.c_float * 4)(0.1, 0.2, 0.3, 0.4)
        out = (ctypes.c_float * 4)()
        rc = lib.ace_scheduler_step(ctx, xt, vt, 4, ctypes.c_float(0.5), out)
        _expect(rc == 0, f"ace_scheduler_step failed: {_last_error(lib)}")
        expected = [0.95, 0.9, 0.85, 0.8]
        for got, exp in zip(list(out), expected):
            _expect(math.isclose(got, exp, rel_tol=0.0, abs_tol=1e-7), f"scheduler mismatch: got={got}, exp={exp}")

        logits = (ctypes.c_float * 5)(0.0, 1.0, 2.0, 3.0, 4.0)
        masked = (ctypes.c_float * 5)()
        rc = lib.ace_apply_lm_constraints(ctx, logits, 5, masked)
        _expect(rc == 0, f"ace_apply_lm_constraints failed: {_last_error(lib)}")
        _expect(abs(masked[2] - 2.0) < 1e-7, "forced token mismatch")
        for i, value in enumerate(masked):
            if i != 2:
                _expect(value < -1e29, f"token {i} should be masked, got={value}")
    finally:
        lib.ace_free_context(ctx)

    print("python ffi regression: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
