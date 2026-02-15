#!/usr/bin/env python3
"""Export ACE-Step 5Hz LM to ONNX with KV-cache interfaces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import transformers.masking_utils as masking_utils
from transformers import AutoModelForCausalLM, DynamicCache

from tools.onnx_export.common import (
    ExportResult,
    ensure_real_onnx_module,
    export_with_policy,
    ort_sanity_load,
    update_manifest,
)


LM_VARIANTS = {
    "0.6B": {"checkpoint": "acestep-5Hz-lm-0.6B", "tag": "0p6"},
    "1.7B": {"checkpoint": "acestep-5Hz-lm-1.7B", "tag": "1p7"},
}


class LmWithCache(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.num_layers = int(model.config.num_hidden_layers)

    def forward(self, input_ids, attention_mask, *past_kv):
        if len(past_kv) == 0:
            past = None
        else:
            if len(past_kv) != self.num_layers * 2:
                raise ValueError("Invalid number of past_kv tensors")
            pairs = []
            for i in range(self.num_layers):
                pairs.append((past_kv[2 * i], past_kv[2 * i + 1]))
            past = DynamicCache.from_legacy_cache(tuple(pairs))

        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            past_key_values=past,
            return_dict=True,
        )

        result = [out.logits]
        legacy_cache = out.past_key_values.to_legacy_cache()
        for k, v in legacy_cache:
            result.extend([k, v])
        return tuple(result)


def _build_names(num_layers: int, with_inputs: bool) -> Tuple[List[str], List[str]]:
    output_names = ["logits"]
    for i in range(num_layers):
        output_names += [f"present_key_{i}", f"present_value_{i}"]

    input_names = ["input_ids", "attention_mask"]
    if with_inputs:
        for i in range(num_layers):
            input_names += [f"past_key_{i}", f"past_value_{i}"]
    return input_names, output_names


def _patch_transformers_masking_for_export() -> None:
    """
    Avoid masking_utils vmap path that triggers `invalid unordered_map<K, T> key`.
    """

    def _sdpa_mask_no_vmap(
        batch_size: int,
        cache_position: torch.Tensor,
        kv_length: int,
        kv_offset: int = 0,
        mask_function=masking_utils.causal_mask_function,
        attention_mask: torch.Tensor | None = None,
        **_: object,
    ) -> torch.Tensor:
        padding_mask = masking_utils.prepare_padding_mask(attention_mask, kv_length, kv_offset, _slice=False)
        if padding_mask is not None:
            mask_function = masking_utils.and_masks(mask_function, masking_utils.padding_mask_function(padding_mask))

        kv_arange = torch.arange(kv_length, device=cache_position.device) + kv_offset
        batch_idx = torch.arange(batch_size, device=cache_position.device).view(batch_size, 1, 1, 1)
        head_idx = torch.zeros(1, device=cache_position.device, dtype=torch.long).view(1, 1, 1, 1)
        q_idx = cache_position.view(1, 1, -1, 1)
        kv_idx = kv_arange.view(1, 1, 1, -1)
        return mask_function(batch_idx, head_idx, q_idx, kv_idx)

    masking_utils.sdpa_mask_recent_torch = _sdpa_mask_no_vmap
    masking_utils.sdpa_mask = _sdpa_mask_no_vmap


def _resolve_variant(variant: str) -> tuple[str, str]:
    info = LM_VARIANTS.get(variant)
    if info is None:
        allowed = ", ".join(sorted(LM_VARIANTS))
        raise ValueError(f"Unsupported --variant={variant!r}. Allowed: {allowed}")
    return info["checkpoint"], info["tag"]


def _export_variant(
    project_root: Path,
    out_dir: Path,
    variant: str,
    opset: int,
    external_data_policy: str,
) -> tuple[Dict[str, object], list[ExportResult]]:
    checkpoint_name, tag = _resolve_variant(variant)
    model_dir = project_root / "checkpoints" / checkpoint_name
    if not model_dir.exists():
        raise FileNotFoundError(f"LM model not found: {model_dir}")

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=torch.float32,
        attn_implementation="eager",
    ).eval().cpu()
    model.config._attn_implementation = "eager"
    for module in model.modules():
        cfg = getattr(module, "config", None)
        if cfg is not None and hasattr(cfg, "_attn_implementation"):
            cfg._attn_implementation = "eager"

    wrapper = LmWithCache(model).eval()
    nl = int(model.config.num_hidden_layers)
    entries: list[ExportResult] = []

    prefill_dir = out_dir / f"lm_{tag}" / "prefill"
    decode_dir = out_dir / f"lm_{tag}" / "decode"
    prefill_dir.mkdir(parents=True, exist_ok=True)
    decode_dir.mkdir(parents=True, exist_ok=True)

    prefill_name = f"lm_{tag}/prefill/lm_{tag}_prefill.onnx"
    decode_name = f"lm_{tag}/decode/lm_{tag}_decode.onnx"
    prefill_path = out_dir / prefill_name
    decode_path = out_dir / decode_name

    input_ids = torch.ones(1, 16, dtype=torch.long)
    attention_mask = torch.ones(1, 16, dtype=torch.long)
    prefill_in_names, prefill_out_names = _build_names(nl, with_inputs=False)

    def _export_prefill(use_external_data: bool) -> None:
        torch.onnx.export(
            wrapper,
            (input_ids, attention_mask),
            str(prefill_path),
            input_names=prefill_in_names,
            output_names=prefill_out_names,
            dynamic_axes={
                "input_ids": {0: "B", 1: "T"},
                "attention_mask": {0: "B", 1: "T"},
                "logits": {0: "B", 1: "T"},
                **{f"present_key_{i}": {0: "B", 2: "T_cache"} for i in range(nl)},
                **{f"present_value_{i}": {0: "B", 2: "T_cache"} for i in range(nl)},
            },
            opset_version=opset,
            do_constant_folding=True,
            external_data=use_external_data,
        )

    entries.append(export_with_policy(prefill_path, external_data_policy, _export_prefill))

    decode_input_ids = torch.ones(1, 1, dtype=torch.long)
    decode_attention_mask = torch.ones(1, 17, dtype=torch.long)
    with torch.no_grad():
        prefill_out = wrapper(input_ids, attention_mask)
    past = tuple(prefill_out[1:])
    decode_in_names, decode_out_names = _build_names(nl, with_inputs=True)

    dyn_axes = {
        "input_ids": {0: "B", 1: "T_new"},
        "attention_mask": {0: "B", 1: "T_total"},
        "logits": {0: "B", 1: "T_new"},
    }
    for i in range(nl):
        dyn_axes[f"past_key_{i}"] = {0: "B", 2: "T_cache"}
        dyn_axes[f"past_value_{i}"] = {0: "B", 2: "T_cache"}
        dyn_axes[f"present_key_{i}"] = {0: "B", 2: "T_cache_next"}
        dyn_axes[f"present_value_{i}"] = {0: "B", 2: "T_cache_next"}

    def _export_decode(use_external_data: bool) -> None:
        torch.onnx.export(
            wrapper,
            (decode_input_ids, decode_attention_mask, *past),
            str(decode_path),
            input_names=decode_in_names,
            output_names=decode_out_names,
            dynamic_axes=dyn_axes,
            opset_version=opset,
            do_constant_folding=True,
            external_data=use_external_data,
        )

    entries.append(export_with_policy(decode_path, external_data_policy, _export_decode))

    contract = {
        "variant": variant,
        "num_layers": nl,
        "prefill_path": prefill_name,
        "decode_path": decode_name,
        "prefill": {"inputs": entries[0].input_names, "outputs": entries[0].output_names},
        "decode": {"inputs": entries[1].input_names, "outputs": entries[1].output_names},
    }
    return contract, entries


def main() -> int:
    p = argparse.ArgumentParser(description="Export ACE-Step 5Hz LM prefill/decode ONNX")
    p.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/onnx_runtime"))
    p.add_argument("--manifest", type=Path)
    p.add_argument("--variant", choices=sorted(LM_VARIANTS.keys()), default="1.7B")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--provider", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--external-data-policy", choices=["auto", "single", "external"], default="auto")
    p.add_argument("--skip-sanity", action="store_true")
    args = p.parse_args()

    ensure_real_onnx_module()
    _patch_transformers_masking_for_export()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest or (args.out_dir / "manifest.json")

    contract, entries = _export_variant(
        project_root=args.project_root,
        out_dir=args.out_dir,
        variant=args.variant,
        opset=args.opset,
        external_data_policy=args.external_data_policy,
    )

    tag = LM_VARIANTS[args.variant]["tag"]
    contract_path = args.out_dir / f"io_contract_lm_{tag}.json"
    contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    print(f"Wrote {contract_path}")

    update_manifest(manifest_path, args.out_dir, entries, model_group=f"lm_{tag}")
    print(f"Updated manifest: {manifest_path}")

    if not args.skip_sanity:
        for entry in entries:
            ok, message = ort_sanity_load(entry.model_path, provider=args.provider)
            prefix = "OK" if ok else "NG"
            print(f"ORT sanity {prefix}: {entry.model_path.name} ({message})")

    print(f"LM export done ({args.variant}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
