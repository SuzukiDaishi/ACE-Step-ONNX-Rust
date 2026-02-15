# ONNX Quantization Profiles

## Script
- `tools/onnx_opt/quantize_all.py`

## Purpose
- Keep FP32 models as canonical reference.
- Generate derived lightweight profiles:
  - `fp16`
  - `int8_dynamic`
  - `int8_static` (copy fallback if no calibration pipeline is configured)

## Run

```bash
python tools/onnx_opt/quantize_all.py \
  --src-dir artifacts/onnx_runtime \
  --out-root artifacts/onnx_runtime_optimized \
  --profiles fp16 int8_dynamic int8_static
```

## Output Layout
- `artifacts/onnx_runtime_optimized/fp16/`
- `artifacts/onnx_runtime_optimized/int8_dynamic/`
- `artifacts/onnx_runtime_optimized/int8_static/`
- `artifacts/onnx_runtime_optimized/manifest.json`

Each profile directory contains:
- converted/copied ONNX files (preserving relative paths)
- copied `io_contract*.json` files
- profile-local `manifest.json`

## Notes
- If a converter or calibration dependency is unavailable, the script copies source ONNX and records this in `note`.
- Accuracy metrics (`rmse/snr/corr`) are placeholders in the profile manifest and are filled by parity runs.

## Parity Integration
- `tools/parity/run_3way_all_cases.py` supports:
  - `--onnx-profile fp32|fp16|int8_dynamic|int8_static`
  - `--skip-audio` (tensor-only evaluation)

## Notes on fp16 (CPU)
- VAE decode can exceed CPU memory.
- Use `--skip-audio` or run with GPU provider if available.

## Evaluation (fp16 example)
```bash
python tools/parity/run_3way_all_cases.py \
  --onnx-profile fp16 \
  --case-thresholds tools/parity/case_thresholds.json \
  --out-root reports/profiles/fp16 \
  --skip-audio

python tools/onnx_opt/fill_manifest_metrics.py \
  --reports-root reports/profiles \
  --profiles fp16
```
