# ACE-Step Rust DLL API (`runtime_rust_dll`)

## Scope
- DLL is for preprocessing / intermediate processing / postprocessing only.
- ONNXRuntime is not linked from this DLL.
- Host language (Python/C#/Node) runs ONNX inference and passes tensors to DLL APIs.

## Build

```bash
cargo build --release --manifest-path runtime_rust_dll/Cargo.toml
```

Output (Windows):
- `runtime_rust_dll/target/release/acestep_runtime.dll`

## Regression

Run Rust unit + Python/C# FFI regression:

```bash
cargo test --manifest-path runtime_rust_dll/Cargo.toml
powershell -ExecutionPolicy Bypass -File examples/ffi/run_regression.ps1
```

## ONNXRuntime + DLL Demo (Python)

```bash
python examples/ffi/python/onnxruntime_dll_demo.py \
  --case fixtures/cases/text2music_05.json \
  --onnx-dir artifacts/onnx_runtime \
  --dll runtime_rust_dll/target/release/acestep_runtime.dll \
  --provider cpu
```

## Release Packaging

```bash
powershell -ExecutionPolicy Bypass -File scripts/dll_release.ps1 \
  -Profile Release \
  -OutDir dist/dll_release \
  -OnnxDir artifacts/onnx_runtime \
  -ArchiveFormat auto
```

Notes:
- `-ArchiveFormat auto` picks `zip` for smaller payloads and `tar` for large payloads (>~1.8GB).
- ONNX with large external data will typically be packaged as `dist/dll_release.tar`.

### Package Contents
- `bin/acestep_runtime.dll`
- `include/acestep_runtime.h`
- `docs/dll_api.md`
- `examples/ffi/python/*`
- `examples/ffi/csharp/*`
- `onnx/*` (including any `.onnx.data` sidecar files)
- `manifest_release.json` (SHA256 + size listing)

### Verification
```bash
powershell -ExecutionPolicy Bypass -File scripts/dll_release.ps1 -Profile Release
powershell -ExecutionPolicy Bypass -File examples/ffi/run_regression.ps1
python examples/ffi/python/onnxruntime_dll_demo.py \
  --case fixtures/cases/text2music_05.json \
  --onnx-dir artifacts/onnx_runtime \
  --dll runtime_rust_dll/target/release/acestep_runtime.dll \
  --provider cpu
```

## Memory Rules
- Any `char*` returned by DLL must be released with `ace_string_free`.
- `AceContext*` must be released with `ace_free_context`.

## API

- `ace_create_context(config_json)`
  - Input: JSON string
  - Output: `AceContext*` or `NULL`
- `ace_free_context(ctx)`
  - Frees context
- `ace_last_error()`
  - Returns last error string

- `ace_prepare_step_inputs(ctx, state_json, in_tensor_ptr, in_tensor_len, out_json)`
  - Prepares timestep/state payload JSON

- `ace_scheduler_step(ctx, xt_ptr, vt_ptr, len, dt, out_xt_ptr)`
  - Computes `xt_next = xt - vt * dt`

- `ace_apply_lm_constraints(ctx, logits_ptr, vocab_size, out_masked_logits_ptr)`
  - Applies blocklist / forced-token constraints

- `ace_finalize_metadata(ctx, token_ids_ptr, len, out_json)`
  - Decodes token ids (if tokenizer configured), parses `<think>` metadata fields, returns JSON

## Config JSON Example

```json
{
  "seed": 42,
  "blocked_token_ids": [151645],
  "forced_token_id": null,
  "tokenizer_path": "checkpoints/acestep-5Hz-lm-1.7B/tokenizer.json"
}
```

## `state_json` Example (`ace_prepare_step_inputs`)

```json
{
  "shift": 3.0,
  "inference_steps": 8,
  "current_step": 0
}
```
