# ACE-Step ONNX + Rust Migration Plan (Detailed)

## 1) Goal

Migrate the current Python-first ACE-Step pipeline into a production-ready stack:

1. ONNX export
2. Python (minimal dependencies) + ONNX Runtime with behavior parity
3. Rust (Python-free) + ONNX Runtime with behavior parity

Primary rule: lock behavior first, optimize second.

---

## 2) Scope

## 2.1 In scope (initial release)
- Task type: `text2music` only
- Model target: `acestep-v15-turbo`
- Inference method: `ode` first
- Output: end-to-end audio generation (`wav` and optional `mp3`)
- Verification: intermediate tensor parity + final audio similarity

## 2.2 Out of scope (later phases)
- Full parity for `cover/repaint/lego/extract/complete`
- ADG/APG and advanced guidance variants
- Full 5Hz LLM parity
- Aggressive quantization and heavy runtime optimization

---

## 3) Current baseline (as implemented today)

Key files:
- `acestep/inference.py`
- `acestep/handler.py`
- `checkpoints/acestep-v15-turbo/modeling_acestep_v15_turbo.py`
- `checkpoints/acestep-v15-turbo/configuration_acestep_v15.py`

Observed architecture:
- Host-side iterative loop already exists in Python (`generate_audio` path)
- Discrete timestep schedules are built around shift values (1, 2, 3)
- Conditioning comes from `prepare_condition`
- VAE decode is separate from DiT generation

Implication:
- Keep scheduler/guidance on host (Python/Rust), not inside ONNX graph
- Export by component instead of one huge graph where possible

---

## 4) Deliverables

## 4.1 Code deliverables
- `tools/onnx_export/` (export-only scripts)
- `runtime_py_ort/` (minimal Python runtime)
- `runtime_rust_ort/` (Rust runtime)
- `fixtures/` (frozen test vectors and case definitions)
- `reports/` (parity + perf reports)

## 4.2 Model deliverables
- `onnx/dit_decoder.onnx`
- `onnx/vae_decoder.onnx`
- Optional (phase extension):
  - `onnx/condition_encoder.onnx`
  - `onnx/text_encoder.onnx`
  - `onnx/lyric_encoder.onnx`
- Tokenizer/config artifacts required by runtime

## 4.3 Documentation deliverables
- `docs/onnx_export.md`
- `docs/python_ort_runtime.md`
- `docs/rust_ort_runtime.md`
- `docs/parity_ops.md`
- `docs/io_contract.md`
 - `docs/dll_api.md`

---

## 5) Milestones and exit criteria

## M0: Baseline freeze
- [ ] Freeze model version and checkpoint hashes
- [x] Freeze representative test cases (at least 10)
- [x] Save baseline outputs from current implementation
- [x] Define intermediate tensor capture points

Exit criteria:
- Reproducible baseline package exists for all test cases

## M1: ONNX export succeeds
- [x] Export scripts exist and run reproducibly
- [x] Exported ONNX graphs load in ORT (CPU EP minimum)
- [ ] Dynamic axes policy is finalized
- [x] One-step sanity run passes for each exported graph

Exit criteria:
- Required ONNX graphs load and execute with stable I/O contracts

## M2: Python ORT parity runtime
- [ ] Runtime without Torch/Transformers/Diffusers at inference time
- [ ] Host scheduler (`ode`) reproduces current behavior
- [ ] Step-level tensor parity tests pass
- [ ] End-to-end audio outputs within tolerance

Exit criteria:
- All parity metrics pass for representative test set

## M3: Rust ORT runtime
- [x] Python-free Rust pipeline runs end-to-end
- [x] Rust outputs match Python ORT outputs within tolerance
- [x] CLI path supports real generation workflow

Exit criteria:
- Rust runtime is functionally equivalent for target scope

---

## 6) Detailed task breakdown

## Phase 0: Baseline freeze (1-2 days)

### P0.1 Test case set
- [x] Create `fixtures/cases/*.json`
- [x] Fix `seed`, `inference_steps`, `infer_method`, `timesteps`, `duration`
- [ ] Keep first version strictly on `text2music`

### P0.2 Intermediate tensor dump
- [x] Capture at least:
  - `encoder_hidden_states`
  - `context_latents`
  - per-step `vt`
  - per-step `xt`
  - pre-VAE latent
- [x] Save in `fixtures/tensors/<case_id>.npz`

### P0.3 Baseline metrics
- [x] Build comparison script for:
  - `max_abs`, `mean_abs`, `rmse`, cosine similarity
- [x] Audio checks:
  - sample count equality
  - SNR / L2 / correlation

---

## Phase 1: ONNX export (3-7 days)

### P1.1 Export strategy
- [x] Start with minimum viable split:
  - `dit_decoder.onnx`
  - `vae_decoder.onnx`
- [x] Add condition encoders after core parity is proven

### P1.2 Export wrappers
- [ ] Implement clean wrappers around model calls
- [ ] Ensure `eval()` and no-grad export path
- [ ] Avoid unsupported fused paths when exporting

### P1.3 Dynamic axes contract
- [ ] `B` (batch)
- [ ] `T_latent` (latent time length)
- [ ] `T_text`, `T_lyric` where required
- [ ] Keep dynamic axes minimal to reduce ORT surprises

### P1.4 ONNX validation
- [x] ORT load checks
- [x] One-batch inference sanity checks
- [ ] Shape/dtype assertions and logging

### P1.5 Version lock
- [x] Lock opset (candidate 17/18)
- [ ] Lock export environment versions

---

## Phase 2: Python minimal runtime + ORT (4-10 days)

### P2.1 Runtime dependency policy
- Required:
  - `onnxruntime` / `onnxruntime-gpu`
  - `numpy`
  - `tokenizers`
- Not allowed at inference runtime:
  - `torch`
  - `transformers`
  - `diffusers`

### P2.2 Runtime modules
- [x] `runtime_py_ort/session_manager.py`
- [x] `runtime_py_ort/scheduler.py`
- [ ] `runtime_py_ort/pipeline.py`
- [ ] `runtime_py_ort/audio_io.py`
- [x] `runtime_py_ort/cli.py`

### P2.3 Scheduler parity
- [x] Mirror current timestep behavior for shift schedules
- [x] Implement update exactly:
  - `xt = xt - vt * dt`
  - final step uses x0 reconstruction path
- [ ] Keep custom timestep mapping rule identical

### P2.4 Parity harness
- [x] Step-level tensor comparison against baseline npz
- [x] Pre-VAE latent comparison
- [x] Final audio similarity report
- [x] Persist results in `reports/parity_py_ort/`

### P2.5 CLI and reproducibility
- [x] Case runner from fixture JSON
- [ ] Optional intermediate dump switch
- [x] Deterministic seed behavior

---

## Phase 3: Rust runtime + ORT (7-14 days)

### P3.1 Rust project structure
- [x] `runtime_rust_ort/Cargo.toml`
- [ ] Modules:
  - `src/ort/`
  - `src/scheduler/`
  - `src/pipeline/`
  - `src/tokenizer/`
  - `src/audio/`
  - `src/bin/generate.rs`

### P3.2 ORT session layer
- [x] Session creation and model lifecycle
- [x] Strong input/output shape checks
- [x] Clear diagnostics for shape/dtype mismatch

### P3.3 Scheduler port
- [x] Exact port from Python ORT implementation
- [x] Same seed policy and random source constraints

### P3.4 End-to-end pipeline
- [x] Input preprocessing
- [x] Iterative denoising loop
- [x] VAE decode
- [x] Audio save pipeline

### P3.5 Rust vs Python parity
- [x] Compare intermediate tensors where possible
- [x] Compare final waveforms and report metrics

### P3.6 CLI
- [ ] Example:
  - `cargo run --release --bin generate -- --case fixtures/cases/case01.json`
- [ ] Include output path + timing + parity summary

---

## 7) Fastest implementation route

1. Restrict to `text2music + turbo + ode`
2. Export and validate `dit_decoder.onnx` first
3. Pass one-step parity
4. Pass full-loop parity
5. Attach `vae_decoder.onnx` and verify audio path
6. Port the verified Python ORT loop to Rust
7. Expand scope only after parity is stable

---

## 8) I/O contract checklist

- [x] Input/output names fixed and documented
- [x] Dtypes fixed (start with fp32)
- [x] Shape semantics fixed (batch/time/channel order)
- [x] Mask conventions fixed (0/1 and tensor dtype)
- [x] Tokenizer normalization behavior fixed

Output:
- `docs/io_contract.md`

---

## 9) Acceptance thresholds

## 9.1 Tensor parity (CPU fp32 reference)
- Per-step `vt`:
  - max_abs <= 1e-4
  - rmse <= 1e-5
- Per-step `xt`:
  - max_abs <= 1e-4
  - rmse <= 1e-5

## 9.2 Audio parity
- Exact sample length match
- SNR >= 35 dB (target)
- Correlation >= 0.98 (target)

## 9.3 Determinism
- Same input + same seed -> same output (within threshold)

## 9.4 Exception thresholds (case-specific)
- Only allowed when auditory output is acceptable and root-cause is tracked.
- Exception thresholds are stored in:
  - `tools/parity/case_thresholds.json`
- Current exceptions:
  - `simple_mode_05`: rmse <= 2e-3, max_abs <= 2.5e-1
  - `text2music_05`: rmse <= 2e-3, max_abs <= 2.5e-1
- All 3-way parity runs must pass with:
  - global thresholds OR the case-specific override.

---

## 10) Risks and mitigations

1) ONNX export operator failures
- Mitigation: split exports and use wrapper-friendly graph paths

2) Tokenizer mismatch across runtimes
- Mitigation: single tokenizer artifact + fixture-based token checks

3) Scheduler mismatch
- Mitigation: build step-level tests before broad integration

4) EP differences (CPU/CUDA/DirectML)
- Mitigation: lock correctness on CPU fp32 first, then move to GPU EP

5) Memory pressure on long durations
- Mitigation: stabilize with short/fixed-length cases first

---

## 11) Issue-ready task IDs

- [x] T-001 baseline case definitions
- [x] T-002 intermediate tensor dump tooling
- [x] T-003 DiT decoder ONNX export script
- [x] T-004 VAE decoder ONNX export script
- [x] T-005 I/O contract documentation
- [x] T-006 Python ORT session manager
- [x] T-007 Python ORT scheduler
- [x] T-008 Python ORT end-to-end pipeline
- [x] T-009 Python parity test runner
- [x] T-010 Rust ORT session manager
- [x] T-011 Rust scheduler
- [x] T-012 Rust end-to-end pipeline
- [x] T-013 Rust vs Python parity runner
- [ ] T-014 CLI + docs completion

---

## 12) Branch strategy

- `feat/onnx-export`
- `feat/python-ort-runtime`
- `feat/parity-tests`
- `feat/rust-ort-runtime`

Each PR must include:
- Scope notes
- Parity report
- Known deviations (if any)

---

## 13) Immediate next actions

1. Finish M0 baseline freeze and fixtures
2. Implement `dit_decoder.onnx` export script
3. Build one-step ORT parity test
4. Extend to full denoising loop parity
5. Add `vae_decoder.onnx` and verify end-to-end audio

This sequence minimizes rework and de-risks the Rust port.

---

## 14) ONNX精度向上（PT基準収束フェーズ）

### 14.1 方針
- 正解系は常に `Python(元実装/PT)` とする。
- 比較は 3 系統で固定:
  - `PT vs Python ONNX`
  - `Python ONNX vs Rust ONNX`
  - `PT vs Rust ONNX`
- 収束順:
1. LM（token/logits step 差分）
2. DiT（`vt_steps/xt_steps`）
3. VAE/audio（最終波形）

### 14.2 実装タスク
- [x] `tools/parity/lm_step_diff.py` を標準運用化（全 simple_mode ケース）
- [x] `tools/parity/dit_step_diff.py` を追加（step 単位で `vt/xt` 比較）
- [x] chat template / tokenizer normalize / mask shape-dtype を `docs/io_contract.md` に固定
- [x] `reports/parity_3way/*` を毎回更新し、fail 要因を `LM/DiT/VAE` で分類

### 14.3 Exit criteria
- `Python ONNX vs Rust ONNX`: 全 10 ケース tensor 差分ゼロ（または既定閾値内）
- `PT vs ONNX`: 既定閾値超過ケースを 0 にするか、既知差分として原因を明示

---

## 15) ONNX量子化・軽量モデル作成

### 15.1 方針
- FP32 モデルを正本として維持し、派生として軽量版を作る。
- サイズ理由のみで機能分割はしない（機能単位分割のみ）。
- 量子化対象は段階導入:
1. DiT
2. LM decode/prefill
3. VAE
- 候補:
  - FP16（重み半精度）
  - INT8 Dynamic（主に LM/MatMul 系）
  - 必要時のみ INT8 Static（calibration 付き）

### 15.2 実装タスク
- [x] `tools/onnx_opt/quantize_all.py` を追加
- [x] 出力配置 `artifacts/onnx_runtime_optimized/{fp16,int8_dynamic,int8_static}/`
- [x] `manifest.json` に元モデル SHA/量子化方式/精度結果を記録
- [x] `run_3way_all_cases.py` に `--onnx-profile {fp32,fp16,int8_dynamic,...}` を追加

### 15.3 受け入れ基準
- 精度劣化上限:
  - tensor: `rmse` 悪化率を FP32 比で管理
  - audio: `SNR/corr` 下限維持
- 速度・メモリ改善が FP32 比で可視化されていること

---

## 16) Rust DLL化（前後処理・中間処理のみ）

### 16.1 スコープ
- DLL は **ORT を内包しない**。
- DLL が担当する処理:
  - 前処理: case 正規化、seed/timestep 生成、token 制約ロジック
  - 中間処理: scheduler step、CFG/制約ロジック、メタデータ FSM
  - 後処理: メタデータ整形、audio post（必要最小）
- 各言語（Python/C#/Node 等）が自前の ONNXRuntime で推論実行し、テンソルを DLL に受け渡す。

### 16.2 公開インターフェース（新規）
- Rust crate を `cdylib` 化し、C ABI を公開:
  - `ace_create_context(config_json)`
  - `ace_free_context(ctx)`
  - `ace_prepare_step_inputs(ctx, state_json, in_tensor_ptr, out_json)`
  - `ace_scheduler_step(ctx, xt_ptr, vt_ptr, dt, out_xt_ptr)`
  - `ace_apply_lm_constraints(ctx, logits_ptr, vocab_size, out_masked_logits_ptr)`
  - `ace_finalize_metadata(ctx, token_ids_ptr, len, out_json)`
- 併せて配布:
  - `include/acestep_runtime.h`
  - `docs/dll_api.md`
  - `examples/ffi/{python,csharp}/`

### 16.3 非機能要件
- ABI 安定化: `v1` を付与、破壊変更は禁止
- メモリ管理責務を明確化（caller free / callee free）
- 文字列は UTF-8、エラーは構造化 JSON で返す

### 16.4 テスト
- [x] Rust unit: scheduler/FSM/token 制約
- [x] FFI integration: Python/C# から同一入力で同一出力
- [x] ORT 非依存確認: DLL リンクに `onnxruntime` を含めない
- [ ] 回帰: 10 ケースで `Python ONNX(ORT host)` と `Rust DLL + ORT host` が一致

### 16.5 リリースパッケージ
- [x] `scripts/dll_release.ps1` を用意
- [x] `docs/dll_api.md` にリリース手順を固定
- [x] `dist/dll_release` + アーカイブ生成（zip/tar）に対応

### 16.6 Exit criteria
- DLL 単体で前後処理・中間処理が再現可能
- ORT 実行はホスト言語側のみで完結
- 10 ケース parity と listening 生成フローが維持される

---

## Important API/Interface changes
- 追加: `tools/parity/dit_step_diff.py`
- 追加: `tools/onnx_opt/quantize_all.py`
- 更新: `tools/parity/run_3way_all_cases.py`（`--onnx-profile`, `--case-thresholds`, `--case-ids`, `--resume`）
- 追加: `tools/parity/case_thresholds.json`
- 追加: `runtime_rust_ort`（または新規 `runtime_rust_dll`）: `cdylib` + C ABI
- 追加: `include/acestep_runtime.h`, `docs/dll_api.md`, `docs/onnx_quantization.md`

---

## Test cases and scenarios
- 必須ケース: `text2music_01..05`, `simple_mode_01..05`
- 比較軸:
  - PT vs PyONNX
  - PyONNX vs RustONNX
  - PT vs RustONNX
- 追加:
  - DLL 経路（各言語 ORT）での同一ケース再現
  - 量子化プロファイルごとの精度・速度比較

---

## Assumptions and defaults
- 基準環境は CPU FP32
- GPU は加速オプション、失敗時は CPU フォールバック
- PT 実装を正解系とする
- DLL は ORT を含まず、ORT は各ホスト言語の Runtime を使う
- 量子化は FP32 正本を保持した派生運用に限定する
