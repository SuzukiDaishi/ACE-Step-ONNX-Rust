# ONNX I/O Contract (Artifacts Runtime)

This project exports ONNX artifacts to `artifacts/onnx_runtime/`.
The source of truth for exact names is each `io_contract_*.json` generated from ONNX graph metadata.

## Output Layout

- `artifacts/onnx_runtime/condition_encoder.onnx`
- `artifacts/onnx_runtime/dit_decoder.onnx`
- `artifacts/onnx_runtime/audio_tokenizer.onnx`
- `artifacts/onnx_runtime/audio_detokenizer.onnx`
- `artifacts/onnx_runtime/vae_decoder.onnx`
- `artifacts/onnx_runtime/lm_0p6_prefill.onnx`
- `artifacts/onnx_runtime/lm_0p6_decode.onnx`
- `artifacts/onnx_runtime/lm_1p7_prefill.onnx`
- `artifacts/onnx_runtime/lm_1p7_decode.onnx`
- `artifacts/onnx_runtime/qwen3_embedding_0p6.onnx`

If a model exceeds ONNX protobuf size limits, export auto-retries with external data and writes sibling files.
Depending on exporter behavior this can be a single `*.onnx.data` file or multiple tensor-named files.

## Contract Files

- `artifacts/onnx_runtime/io_contract_core.json`
- `artifacts/onnx_runtime/io_contract_lm_0p6.json`
- `artifacts/onnx_runtime/io_contract_lm_1p7.json`
- `artifacts/onnx_runtime/io_contract_qwen3_embedding_0p6.json`
- `artifacts/onnx_runtime/manifest.json`

## Core Contract

`io_contract_core.json` contains:
- `inputs.condition_encoder`
- `outputs.condition_encoder`
- `inputs.dit_decoder`
- `outputs.dit_decoder`
- `inputs.audio_tokenizer`
- `outputs.audio_tokenizer`
- `inputs.audio_detokenizer`
- `outputs.audio_detokenizer`
- `inputs.vae_decoder`
- `outputs.vae_decoder`

Important parity input:
- `dit_decoder` keeps `encoder_attention_mask` as an explicit input for cross-attention masking.

## LM Contract

Each LM contract file contains:
- `variant` (`0.6B` or `1.7B`)
- `num_layers`
- `prefill_path`
- `decode_path`
- `prefill.inputs` / `prefill.outputs`
- `decode.inputs` / `decode.outputs`

Both prefill/decode expose per-layer KV cache tensors:
- prefill outputs: `present_key_i`, `present_value_i`
- decode inputs: `past_key_i`, `past_value_i`
- decode outputs: `present_key_i`, `present_value_i`

### LM Prompt Chat Template Contract

Simple mode LM prompt must use the same logical template in Python ONNX and Rust ONNX:

- `system`:
  - `# Instruction`
  - `Expand the user's input into a more detailed and specific musical description:`
- `user`:
  - `{simple_mode_query}`
  - blank line
  - `instrumental: {true|false}`
- generation prompt: assistant turn suffix must be appended

This is currently implemented by:
- `runtime_py_ort/pipeline_lm.py` (`_build_inspiration_prompt`)
- `src/pipeline/lm.rs` (`build_inspiration_prompt`)

### Tokenizer Normalization Contract

- LM tokenization:
  - `add_special_tokens=False` for prompt id generation
  - decoding uses `skip_special_tokens=False`
- Qwen embedding tokenization:
  - empty input must normalize to a single space (`" "`)
  - truncation is host-side by `max_tokens`
- string normalization:
  - comparisons for constrained transitions are lowercased and left-trim aware
  - emitted metadata text remains original model output text except explicit post-processing rules

### Mask Shape / Dtype Contract

- `encoder_attention_mask` (DiT cross-attention):
  - must be passed explicitly to `dit_decoder`
  - runtime canonical dtype is float (`0.0/1.0`) for host-managed comparisons
- `attention_mask` (LM/Qwen):
  - ONNX input dtype is `int64`
  - host must provide shape `[B, T]`
- `refer_audio_order_mask`:
  - condition encoder input dtype is `int64`
- `is_covers`:
  - condition encoder input dtype is boolean

For parity tooling, if a bool mask is emitted by ONNX, convert to `float32` (`0/1`) before report serialization.

## Embedding Contract

`io_contract_qwen3_embedding_0p6.json` contains:
- `path`
- `inputs` (`input_ids`, `attention_mask`)
- `outputs` (`last_hidden_state`)

## Manifest

`manifest.json` records:
- file path
- whether external data is used
- external sidecar file list
- ONNX input/output names
- `sha256`
- file size
