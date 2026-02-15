from __future__ import annotations

import ctypes
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DLL = ROOT / "runtime_rust_dll" / "target" / "release" / "acestep_runtime.dll"

lib = ctypes.CDLL(str(DLL))

lib.ace_create_context.argtypes = [ctypes.c_char_p]
lib.ace_create_context.restype = ctypes.c_void_p
lib.ace_free_context.argtypes = [ctypes.c_void_p]
lib.ace_string_free.argtypes = [ctypes.c_void_p]

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


def cstr(ptr: int) -> str:
    return ctypes.cast(ptr, ctypes.c_char_p).value.decode("utf-8")


def main() -> int:
    cfg = {"seed": 42, "blocked_token_ids": [0]}
    ctx = lib.ace_create_context(json.dumps(cfg).encode("utf-8"))
    if not ctx:
        raise RuntimeError("failed to create context")

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
        if rc != 0:
            raise RuntimeError("ace_prepare_step_inputs failed")
        print("prepare:", cstr(out_json.value))
        lib.ace_string_free(out_json)

        xt = (ctypes.c_float * 4)(1.0, 1.0, 1.0, 1.0)
        vt = (ctypes.c_float * 4)(0.1, 0.2, 0.3, 0.4)
        out = (ctypes.c_float * 4)()
        rc = lib.ace_scheduler_step(ctx, xt, vt, 4, ctypes.c_float(0.5), out)
        if rc != 0:
            raise RuntimeError("ace_scheduler_step failed")
        print("scheduler:", list(out))
    finally:
        lib.ace_free_context(ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
