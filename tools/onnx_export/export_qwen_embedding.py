#!/usr/bin/env python3
"""Export Qwen3 embedding model to ONNX."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
import transformers.masking_utils as masking_utils
from transformers import AutoModel

from tools.onnx_export.common import (
    ExportResult,
    ensure_real_onnx_module,
    export_with_policy,
    ort_sanity_load,
    update_manifest,
)


class QwenEmbeddingWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        return outputs[0]


class QwenTokenEmbeddingWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.embed_tokens = model.get_input_embeddings()

    def forward(self, input_ids):
        return self.embed_tokens(input_ids)


def _patch_transformers_masking_for_export() -> None:
    """
    Avoid masking_utils vmap path that can fail in ONNX tracing on Windows.
    """

    def _sdpa_mask_no_vmap(
        batch_size: int,
        cache_position: torch.Tensor,
        kv_length: int,
        kv_offset: int = 0,
        mask_function=masking_utils.causal_mask_function,
        attention_mask: torch.Tensor | None = None,
        local_size: int | None = None,
        allow_is_causal_skip: bool = True,
        **_: object,
    ) -> torch.Tensor | None:
        q_length = cache_position.shape[0]
        padding_mask = masking_utils.prepare_padding_mask(attention_mask, kv_length, kv_offset, _slice=False)
        if allow_is_causal_skip and masking_utils._ignore_causal_mask_sdpa(
            padding_mask, q_length, kv_length, kv_offset, local_size
        ):
            return None

        if padding_mask is not None:
            mask_function = masking_utils.and_masks(mask_function, masking_utils.padding_mask_function(padding_mask))

        kv_arange = torch.arange(kv_length, device=cache_position.device) + kv_offset
        batch_idx = torch.arange(batch_size, device=cache_position.device).view(batch_size, 1, 1, 1)
        head_idx = torch.arange(1, device=cache_position.device).view(1, 1, 1, 1)
        q_idx = cache_position.view(1, 1, -1, 1)
        kv_idx = kv_arange.view(1, 1, 1, -1)
        return mask_function(batch_idx, head_idx, q_idx, kv_idx)

    masking_utils.sdpa_mask_recent_torch = _sdpa_mask_no_vmap
    masking_utils.sdpa_mask = _sdpa_mask_no_vmap


def _load_model(model_dir: Path) -> torch.nn.Module:
    try:
        model = AutoModel.from_pretrained(str(model_dir), torch_dtype=torch.float32)
    except Exception:
        # Fallback for environments requiring custom modeling code.
        model = AutoModel.from_pretrained(str(model_dir), torch_dtype=torch.float32, trust_remote_code=True)

    if hasattr(model, "config") and hasattr(model.config, "_attn_implementation"):
        model.config._attn_implementation = "eager"
    for module in model.modules():
        cfg = getattr(module, "config", None)
        if cfg is not None and hasattr(cfg, "_attn_implementation"):
            cfg._attn_implementation = "eager"
    return model.eval().cpu()


def _export_model(
    project_root: Path,
    out_dir: Path,
    opset: int,
    external_data_policy: str,
) -> tuple[Dict[str, object], list[ExportResult]]:
    model_dir = project_root / "checkpoints" / "Qwen3-Embedding-0.6B"
    if not model_dir.exists():
        raise FileNotFoundError(f"Qwen embedding model not found: {model_dir}")

    model = _load_model(model_dir)
    wrapper = QwenEmbeddingWrapper(model).eval()
    token_wrapper = QwenTokenEmbeddingWrapper(model).eval()
    out_path = out_dir / "qwen3_embedding_0p6.onnx"
    token_out_path = out_dir / "qwen3_token_embedding_0p6.onnx"
    input_ids = torch.ones(1, 16, dtype=torch.long)
    attention_mask = torch.ones(1, 16, dtype=torch.long)

    def _export_fn(use_external_data: bool) -> None:
        torch.onnx.export(
            wrapper,
            (input_ids, attention_mask),
            str(out_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["last_hidden_state"],
            dynamic_axes={
                "input_ids": {0: "B", 1: "T"},
                "attention_mask": {0: "B", 1: "T"},
                "last_hidden_state": {0: "B", 1: "T"},
            },
            opset_version=opset,
            do_constant_folding=True,
            external_data=use_external_data,
        )

    entries: list[ExportResult] = []
    entries.append(export_with_policy(out_path, external_data_policy, _export_fn))

    def _export_token_fn(use_external_data: bool) -> None:
        torch.onnx.export(
            token_wrapper,
            (input_ids,),
            str(token_out_path),
            input_names=["input_ids"],
            output_names=["token_embeddings"],
            dynamic_axes={
                "input_ids": {0: "B", 1: "T"},
                "token_embeddings": {0: "B", 1: "T"},
            },
            opset_version=opset,
            do_constant_folding=True,
            external_data=use_external_data,
        )

    entries.append(export_with_policy(token_out_path, external_data_policy, _export_token_fn))
    contract = {
        "path": out_path.name,
        "inputs": entries[0].input_names,
        "outputs": entries[0].output_names,
        "token_embedding_path": token_out_path.name,
        "token_embedding_inputs": entries[1].input_names,
        "token_embedding_outputs": entries[1].output_names,
    }
    return contract, entries


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Qwen3 Embedding 0.6B to ONNX")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/onnx_runtime"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--provider", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--external-data-policy", choices=["auto", "single", "external"], default="auto")
    parser.add_argument("--skip-sanity", action="store_true")
    args = parser.parse_args()

    ensure_real_onnx_module()
    _patch_transformers_masking_for_export()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest or (args.out_dir / "manifest.json")

    contract, entries = _export_model(
        project_root=args.project_root,
        out_dir=args.out_dir,
        opset=args.opset,
        external_data_policy=args.external_data_policy,
    )

    contract_path = args.out_dir / "io_contract_qwen3_embedding_0p6.json"
    contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    print(f"Wrote {contract_path}")

    update_manifest(manifest_path, args.out_dir, entries, model_group="qwen3_embedding_0p6")
    print(f"Updated manifest: {manifest_path}")

    if not args.skip_sanity:
        for entry in entries:
            ok, message = ort_sanity_load(entry.model_path, provider=args.provider)
            prefix = "OK" if ok else "NG"
            print(f"ORT sanity {prefix}: {entry.model_path.name} ({message})")

    print("Qwen embedding export done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
