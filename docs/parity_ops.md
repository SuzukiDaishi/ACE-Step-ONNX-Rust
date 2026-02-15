# Parity Ops

## Purpose
- Provide a repeatable parity workflow and document case-specific threshold overrides.
- Keep `Python(PT)` as the single source of truth.

## Global thresholds (CPU fp32)
- Tensor: `rmse <= 5e-4`, `max_abs <= 1e-1`
- Audio: `snr >= 35dB`, `corr >= 0.98`

## Case-specific thresholds
- Stored in `tools/parity/case_thresholds.json`
- Only allowed when auditory output is acceptable and root-cause is tracked.

Example:
```json
{
  "simple_mode_05": {
    "tensor_rmse": 0.002,
    "tensor_max_abs": 0.25
  },
  "text2music_05": {
    "tensor_rmse": 0.002,
    "tensor_max_abs": 0.25
  }
}
```

## Standard 3-way run
```bash
python tools/parity/run_3way_all_cases.py \
  --case-thresholds tools/parity/case_thresholds.json
```

## Resume run (skip completed cases)
```bash
python tools/parity/run_3way_all_cases.py \
  --case-thresholds tools/parity/case_thresholds.json \
  --resume
```

## Single case
```bash
python tools/parity/run_3way_all_cases.py \
  --case-thresholds tools/parity/case_thresholds.json \
  --case-ids text2music_05
```

## Skip audio (tensor-only evaluation)
Use when VAE decode is too heavy for CPU (e.g. fp16 profile).
```bash
python tools/parity/run_3way_all_cases.py \
  --case-thresholds tools/parity/case_thresholds.json \
  --skip-audio
```

## Notes
- `--resume` only skips cases that already have `reports/parity_3way/<case>_failure_tags.json`
  and listening wavs in `reports/listening/<case>/`.
- `summary_all_cases.json` is regenerated each run.
