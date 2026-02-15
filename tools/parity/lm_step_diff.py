#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from acestep.constants import DEFAULT_LM_INSPIRED_INSTRUCTION
from acestep.constrained_logits_processor import MetadataConstrainedLogitsProcessor


@dataclass
class LmContract:
    num_layers: int
    prefill_path: str
    decode_path: str


def _load_contract(onnx_dir: Path, variant: str) -> LmContract:
    tag = "0p6" if variant == "0.6B" else "1p7"
    path = onnx_dir / f"io_contract_lm_{tag}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return LmContract(
        num_layers=int(data["num_layers"]),
        prefill_path=str(data["prefill_path"]),
        decode_path=str(data["decode_path"]),
    )


def _to_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy().astype(np.float32, copy=False)


def _metrics(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    da = np.nan_to_num(a.astype(np.float64, copy=False), nan=0.0, posinf=0.0, neginf=0.0).ravel()
    db = np.nan_to_num(b.astype(np.float64, copy=False), nan=0.0, posinf=0.0, neginf=0.0).ravel()
    diff = da - db
    rmse = float(np.sqrt(np.mean(diff * diff)))
    max_abs = float(np.max(np.abs(diff)))
    denom = float(np.linalg.norm(da) * np.linalg.norm(db))
    cos = float(np.dot(da, db) / denom) if denom > 0 else 1.0
    return {"rmse": rmse, "max_abs": max_abs, "cos_sim": cos}


def _build_prompt(query: str, instrumental: bool) -> str:
    inst = "true" if instrumental else "false"
    query_text = query.strip() if query and query.strip() else "NO USER INPUT"
    return (
        "<|im_start|>system\n"
        f"# Instruction\n{DEFAULT_LM_INSPIRED_INSTRUCTION}\n\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{query_text}\n\ninstrumental: {inst}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _build_processor(
    tokenizer,
    vocal_language: str,
    constrained: bool,
) -> Optional[MetadataConstrainedLogitsProcessor]:
    if not constrained:
        return None
    proc = MetadataConstrainedLogitsProcessor(tokenizer=tokenizer, enabled=True, debug=False)
    proc.reset()
    proc.set_generation_phase("understand")
    proc.set_stop_at_reasoning(False)
    proc.set_skip_genres(False)
    proc.set_skip_caption(False)
    proc.set_skip_language(False)
    lang = (vocal_language or "").strip().lower()
    if lang and lang != "unknown":
        proc.set_user_metadata({"language": vocal_language.strip()})
    else:
        proc.set_user_metadata(None)
    return proc


def _run_pt_step(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    past_key_values,
) -> Tuple[torch.Tensor, Any]:
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
    return outputs.logits[:, -1, :], outputs.past_key_values


def _run_onnx_prefill(
    sess: ort.InferenceSession,
    prompt_ids: List[int],
) -> Tuple[np.ndarray, List[np.ndarray]]:
    input_ids = np.asarray([prompt_ids], dtype=np.int64)
    attn = np.ones_like(input_ids, dtype=np.int64)
    out = sess.run(None, {"input_ids": input_ids, "attention_mask": attn})
    logits = out[0][:, -1, :].astype(np.float32, copy=False)
    cache = [x.astype(np.float32, copy=False) for x in out[1:]]
    return logits, cache


def _run_onnx_decode(
    sess: ort.InferenceSession,
    next_token_id: int,
    total_len: int,
    cache: List[np.ndarray],
    num_layers: int,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    input_ids = np.asarray([[next_token_id]], dtype=np.int64)
    attn = np.ones((1, total_len), dtype=np.int64)
    feeds: Dict[str, Any] = {"input_ids": input_ids, "attention_mask": attn}
    for i in range(num_layers):
        feeds[f"past_key_{i}"] = cache[2 * i]
        feeds[f"past_value_{i}"] = cache[2 * i + 1]
    out = sess.run(None, feeds)
    logits = out[0][:, -1, :].astype(np.float32, copy=False)
    next_cache = [x.astype(np.float32, copy=False) for x in out[1:]]
    return logits, next_cache


def main() -> int:
    parser = argparse.ArgumentParser(description="Step-level PT LM vs ONNX LM diff")
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--onnx-dir", type=Path, default=Path("artifacts/onnx_runtime"))
    parser.add_argument("--variant", choices=["0.6B", "1.7B"], default="1.7B")
    parser.add_argument("--lm-model-dir", type=Path, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--constrained", action="store_true")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    case = json.loads(args.case.read_text(encoding="utf-8"))
    query = str(case.get("simple_mode_query", ""))
    meta = case.get("metadata", {}) or {}
    instrumental = bool(meta.get("instrumental", False))
    vocal_language = str(meta.get("vocal_language", "unknown"))
    seed = int(case.get("seed", 42))

    torch.manual_seed(seed)
    np.random.seed(seed)

    model_dir = args.lm_model_dir
    if model_dir is None:
        model_dir = Path("checkpoints") / ("acestep-5Hz-lm-0.6B" if args.variant == "0.6B" else "acestep-5Hz-lm-1.7B")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True).to(args.device).eval()

    prompt = _build_prompt(query, instrumental)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

    pt_proc = _build_processor(tokenizer, vocal_language=vocal_language, constrained=args.constrained)
    onnx_proc = _build_processor(tokenizer, vocal_language=vocal_language, constrained=args.constrained)

    contract = _load_contract(args.onnx_dir, args.variant)
    prefill_sess = ort.InferenceSession(str(args.onnx_dir / contract.prefill_path), providers=["CPUExecutionProvider"])
    decode_sess = ort.InferenceSession(str(args.onnx_dir / contract.decode_path), providers=["CPUExecutionProvider"])

    # PT prefill
    pt_input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=args.device)
    pt_attn = torch.ones_like(pt_input_ids, dtype=torch.long)
    pt_logits, pt_cache = _run_pt_step(model, pt_input_ids, pt_attn, past_key_values=None)
    pt_logits = _to_np(pt_logits)

    # ONNX prefill
    onnx_logits, onnx_cache = _run_onnx_prefill(prefill_sess, prompt_ids)

    eos_id = getattr(tokenizer, "eos_token_id", None)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end_id, list):
        im_end_id = im_end_id[0] if im_end_id else None

    generated_pt: List[int] = []
    generated_onnx: List[int] = []
    step_rows: List[Dict[str, Any]] = []
    first_div: Optional[int] = None

    for step in range(args.max_new_tokens):
        raw_m = _metrics(pt_logits[0], onnx_logits[0])

        pt_proc_logits = pt_logits.copy()
        onnx_proc_logits = onnx_logits.copy()
        if pt_proc is not None:
            ids_t = torch.tensor([prompt_ids + generated_pt], dtype=torch.long)
            scores_t = torch.from_numpy(pt_proc_logits.astype(np.float32, copy=False))
            scores_t = pt_proc(ids_t, scores_t)
            pt_proc_logits = scores_t.detach().cpu().numpy().astype(np.float32, copy=False)
        if onnx_proc is not None:
            ids_t = torch.tensor([prompt_ids + generated_onnx], dtype=torch.long)
            scores_t = torch.from_numpy(onnx_proc_logits.astype(np.float32, copy=False))
            scores_t = onnx_proc(ids_t, scores_t)
            onnx_proc_logits = scores_t.detach().cpu().numpy().astype(np.float32, copy=False)

        proc_m = _metrics(pt_proc_logits[0], onnx_proc_logits[0])

        pt_token = int(np.argmax(pt_proc_logits[0]))
        onnx_token = int(np.argmax(onnx_proc_logits[0]))
        generated_pt.append(pt_token)
        generated_onnx.append(onnx_token)

        if pt_proc is not None:
            pt_proc.update_state(pt_token)
        if onnx_proc is not None:
            onnx_proc.update_state(onnx_token)

        if first_div is None and pt_token != onnx_token:
            first_div = step

        step_rows.append(
            {
                "step": step,
                "pt_token": pt_token,
                "onnx_token": onnx_token,
                "token_match": pt_token == onnx_token,
                "raw_logits": raw_m,
                "proc_logits": proc_m,
            }
        )

        def should_stop(tok: int) -> bool:
            return (eos_id is not None and tok == int(eos_id)) or (pad_id is not None and tok == int(pad_id)) or (
                im_end_id is not None and tok == int(im_end_id)
            )

        # advance PT
        if should_stop(pt_token):
            break
        pt_input_ids = torch.tensor([[pt_token]], dtype=torch.long, device=args.device)
        pt_attn = torch.ones((1, len(prompt_ids) + len(generated_pt)), dtype=torch.long, device=args.device)
        pt_logits, pt_cache = _run_pt_step(model, pt_input_ids, pt_attn, past_key_values=pt_cache)
        pt_logits = _to_np(pt_logits)

        # advance ONNX
        if should_stop(onnx_token):
            break
        onnx_logits, onnx_cache = _run_onnx_decode(
            decode_sess,
            onnx_token,
            len(prompt_ids) + len(generated_onnx),
            onnx_cache,
            contract.num_layers,
        )

    out = {
        "case_id": case.get("case_id"),
        "variant": args.variant,
        "constrained": bool(args.constrained),
        "max_new_tokens": int(args.max_new_tokens),
        "prompt_len": len(prompt_ids),
        "steps": len(step_rows),
        "first_token_divergence_step": first_div,
        "final_pt_token_count": len(generated_pt),
        "final_onnx_token_count": len(generated_onnx),
        "token_exact_match": generated_pt == generated_onnx,
        "step_metrics": step_rows,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
