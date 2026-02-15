#!/usr/bin/env python3
"""Export ACE-Step core modules to ONNX."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import types
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from acestep.handler import AceStepHandler
from tools.onnx_export.common import (
    ExportResult,
    ensure_real_onnx_module,
    export_with_policy,
    ort_sanity_load,
    update_manifest,
)


class ConditionEncoderWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        text_hidden_states,
        text_attention_mask,
        lyric_hidden_states,
        lyric_attention_mask,
        refer_audio_acoustic_hidden_states_packed,
        refer_audio_order_mask,
        hidden_states,
        attention_mask,
        silence_latent,
        src_latents,
        chunk_masks,
        is_covers,
        precomputed_lm_hints_25hz,
    ):
        encoder_hidden_states, encoder_attention_mask, context_latents = self.model.prepare_condition(
            text_hidden_states=text_hidden_states,
            text_attention_mask=text_attention_mask,
            lyric_hidden_states=lyric_hidden_states,
            lyric_attention_mask=lyric_attention_mask,
            refer_audio_acoustic_hidden_states_packed=refer_audio_acoustic_hidden_states_packed,
            refer_audio_order_mask=refer_audio_order_mask,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            silence_latent=silence_latent,
            src_latents=src_latents,
            chunk_masks=chunk_masks,
            is_covers=is_covers,
            precomputed_lm_hints_25Hz=precomputed_lm_hints_25hz,
            audio_codes=None,
        )
        return encoder_hidden_states, encoder_attention_mask, context_latents


class DitDecoderWrapper(torch.nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(
        self,
        hidden_states,
        timestep,
        timestep_r,
        attention_mask,
        encoder_hidden_states,
        encoder_attention_mask,
        context_latents,
    ):
        out = self.decoder(
            hidden_states=hidden_states,
            timestep=timestep,
            timestep_r=timestep_r,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            context_latents=context_latents,
            use_cache=False,
            past_key_values=None,
        )
        return out[0]


class DitDecoderKvWrapper(torch.nn.Module):
    def __init__(self, decoder, num_layers: int):
        super().__init__()
        self.decoder = decoder
        self.num_layers = int(num_layers)
        cfg = decoder.config
        self.num_kv_heads = int(getattr(cfg, "num_key_value_heads", cfg.num_attention_heads))
        self.head_dim = int(
            getattr(decoder.layers[0].self_attn, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        )

    def forward(
        self,
        hidden_states,
        timestep,
        timestep_r,
        attention_mask,
        encoder_hidden_states,
        encoder_attention_mask,
        context_latents,
        *past_kv,
    ):
        past_layers: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        cache_obj = None
        if len(past_kv) != 0:
            expected = self.num_layers * 4
            if len(past_kv) != expected:
                raise ValueError(f"Invalid number of DiT past_kv tensors: {len(past_kv)} != {expected}")
            for i in range(self.num_layers):
                base = i * 4
                past_layers.append((past_kv[base], past_kv[base + 1], past_kv[base + 2], past_kv[base + 3]))

            # Keep past inputs alive in ONNX graph without changing numerics.
            dummy = hidden_states.new_zeros(())
            for tensor in past_kv:
                dummy = dummy + tensor.sum() * 0.0
            hidden_states = hidden_states + dummy

            # This model only uses cross-attention cache in generation.
            module = importlib.import_module(self.decoder.__class__.__module__)
            EncoderDecoderCache = module.EncoderDecoderCache
            DynamicCache = module.DynamicCache
            cache_obj = EncoderDecoderCache(DynamicCache(), DynamicCache())
            for i, (_, _, p_cross_k, p_cross_v) in enumerate(past_layers):
                if p_cross_k.numel() == 0 or p_cross_v.numel() == 0:
                    continue
                cache_obj.cross_attention_cache.update(p_cross_k, p_cross_v, i)
                cache_obj.is_updated[i] = True

        out = self.decoder(
            hidden_states=hidden_states,
            timestep=timestep,
            timestep_r=timestep_r,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            context_latents=context_latents,
            use_cache=True,
            past_key_values=cache_obj,
        )
        result: List[torch.Tensor] = [out[0]]
        present_cache = out[1] if len(out) > 1 else None

        bsz = hidden_states.shape[0]
        seq_latent = hidden_states.shape[1]
        seq_enc = encoder_hidden_states.shape[1]
        current_self_k = torch.zeros(
            (bsz, self.num_kv_heads, seq_latent, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        current_self_v = torch.zeros(
            (bsz, self.num_kv_heads, seq_latent, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        current_cross_k = torch.zeros(
            (bsz, self.num_kv_heads, seq_enc, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        current_cross_v = torch.zeros(
            (bsz, self.num_kv_heads, seq_enc, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        use_real_cross_cache = (
            present_cache is not None
            and hasattr(present_cache, "cross_attention_cache")
            and len(present_cache.cross_attention_cache) == self.num_layers
        )

        for i in range(self.num_layers):
            if use_real_cross_cache:
                layer = present_cache.cross_attention_cache.layers[i]
                present_cross_k = layer.keys
                present_cross_v = layer.values
                if past_layers:
                    p_self_k, p_self_v, _, _ = past_layers[i]
                    present_self_k = p_self_k
                    present_self_v = p_self_v
                else:
                    present_self_k = current_self_k
                    present_self_v = current_self_v
                result.extend([present_self_k, present_self_v, present_cross_k, present_cross_v])
                continue

            if past_layers:
                p_self_k, p_self_v, p_cross_k, p_cross_v = past_layers[i]
                present_self_k = torch.cat([p_self_k, current_self_k], dim=2)
                present_self_v = torch.cat([p_self_v, current_self_v], dim=2)
                present_cross_k = p_cross_k
                present_cross_v = p_cross_v
            else:
                present_self_k = current_self_k
                present_self_v = current_self_v
                present_cross_k = current_cross_k
                present_cross_v = current_cross_v
            result.extend([present_self_k, present_self_v, present_cross_k, present_cross_v])
        return tuple(result)


class AudioTokenizerWrapper(torch.nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def forward(self, hidden_states):
        quantized, indices = self.tokenizer.tokenize(hidden_states)
        return quantized, indices.to(torch.int64)


class AudioDetokenizerWrapper(torch.nn.Module):
    def __init__(self, detokenizer):
        super().__init__()
        self.detokenizer = detokenizer

    def forward(self, quantized):
        return self.detokenizer(quantized)


class VaeDecoderWrapper(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latents):
        return self.vae.decode(latents).sample


def _build_dit_kv_names(num_layers: int, with_inputs: bool) -> Tuple[List[str], List[str]]:
    output_names = ["vt"]
    for i in range(num_layers):
        output_names += [
            f"present_self_key_{i}",
            f"present_self_value_{i}",
            f"present_cross_key_{i}",
            f"present_cross_value_{i}",
        ]
    input_names = [
        "hidden_states",
        "timestep",
        "timestep_r",
        "attention_mask",
        "encoder_hidden_states",
        "encoder_attention_mask",
        "context_latents",
    ]
    if with_inputs:
        for i in range(num_layers):
            input_names += [
                f"past_self_key_{i}",
                f"past_self_value_{i}",
                f"past_cross_key_{i}",
                f"past_cross_value_{i}",
            ]
    return input_names, output_names


def _init_handler(project_root: Path, model_path: str, provider: str) -> AceStepHandler:
    device = "cuda" if provider == "cuda" and torch.cuda.is_available() else "cpu"
    handler = AceStepHandler()
    status, ok = handler.initialize_service(
        project_root=str(project_root),
        config_path=model_path,
        device=device,
        use_flash_attention=False,
        compile_model=False,
        offload_to_cpu=False,
        offload_dit_to_cpu=False,
    )
    if not ok:
        raise RuntimeError(f"Failed to init handler: {status}")
    return handler


def _sample_inputs(device: torch.device) -> Dict[str, torch.Tensor]:
    b = 1
    t = 250
    t_text = 77
    t_lyric = 128
    t_enc = 256

    text_attention_mask = torch.ones(b, t_text, device=device)
    text_attention_mask[:, -8:] = 0
    lyric_attention_mask = torch.ones(b, t_lyric, device=device)
    lyric_attention_mask[:, -12:] = 0
    attention_mask = torch.ones(b, t, device=device)
    attention_mask[:, -16:] = 0
    encoder_attention_mask = torch.ones(b, t_enc, device=device)
    encoder_attention_mask[:, -32:] = 0

    return {
        "text_hidden_states": torch.randn(b, t_text, 1024, device=device),
        "text_attention_mask": text_attention_mask,
        "lyric_hidden_states": torch.randn(b, t_lyric, 1024, device=device),
        "lyric_attention_mask": lyric_attention_mask,
        "refer_audio_acoustic_hidden_states_packed": torch.randn(1, 750, 64, device=device),
        "refer_audio_order_mask": torch.zeros(1, dtype=torch.long, device=device),
        "hidden_states": torch.randn(b, t, 64, device=device),
        "attention_mask": attention_mask,
        "silence_latent": torch.randn(b, t, 64, device=device),
        "src_latents": torch.randn(b, t, 64, device=device),
        "chunk_masks": torch.ones(b, t, 64, device=device),
        "is_covers": torch.zeros(b, dtype=torch.bool, device=device),
        "precomputed_lm_hints_25hz": torch.zeros(b, t, 64, device=device),
        "dit_hidden_states": torch.randn(b, t, 64, device=device),
        "timestep": torch.full((b,), 1.0, device=device),
        "timestep_r": torch.full((b,), 1.0, device=device),
        "encoder_hidden_states": torch.randn(b, t_enc, 2048, device=device),
        "encoder_attention_mask": encoder_attention_mask,
        "context_latents": torch.randn(b, t, 128, device=device),
        "vae_latents": torch.randn(b, 64, t, device=device),
        "audio_hidden_states": torch.randn(b, t, 64, device=device),
    }


def _patch_timbre_encoder_for_onnx(model: torch.nn.Module) -> None:
    """Patch timbre unpack path to avoid aten::bincount."""
    timbre_encoder = model.encoder.timbre_encoder

    def _unpack_timbre_embeddings_no_bincount(self, timbre_embs_packed, refer_audio_order_mask):
        _, d = timbre_embs_packed.shape
        dtype = timbre_embs_packed.dtype
        bsz = int(refer_audio_order_mask.max().item() + 1)
        order_one_hot = F.one_hot(refer_audio_order_mask, num_classes=bsz).to(dtype)
        counts = order_one_hot.sum(dim=0).to(torch.long)
        max_count = int(counts.max().item())

        cumsum_per_batch = torch.cumsum(order_one_hot, dim=0).to(torch.long) - 1
        positions_in_batch = (cumsum_per_batch * order_one_hot.to(torch.long)).sum(dim=1)
        indices_2d = refer_audio_order_mask.to(torch.long) * max_count + positions_in_batch
        one_hot_2d = F.one_hot(indices_2d, num_classes=bsz * max_count).to(dtype)

        unpacked_flat = one_hot_2d.transpose(0, 1) @ timbre_embs_packed
        timbre_embs_unpack = unpacked_flat.reshape(bsz, max_count, d)
        valid_mask = (one_hot_2d.sum(dim=0) > 0).to(torch.long)
        new_mask = valid_mask.reshape(bsz, max_count)
        return timbre_embs_unpack, new_mask

    timbre_encoder.unpack_timbre_embeddings = types.MethodType(
        _unpack_timbre_embeddings_no_bincount, timbre_encoder
    )


def _patch_pack_sequences_for_onnx(model: torch.nn.Module) -> None:
    """Patch pack_sequences to avoid argsort-based implementation."""
    module = importlib.import_module(model.__class__.__module__)

    def _pack_sequences_no_sort(hidden1, hidden2, mask1, mask2):
        hidden_cat = torch.cat([hidden1, hidden2], dim=1)
        mask_cat = torch.cat([mask1, mask2], dim=1).to(torch.long)
        _, seq_len, hidden_dim = hidden_cat.shape

        lengths = mask_cat.sum(dim=1, keepdim=True)
        ones_rank = torch.cumsum(mask_cat, dim=1) - 1
        zeros_mask = 1 - mask_cat
        zeros_rank = torch.cumsum(zeros_mask, dim=1) - 1
        target_pos = torch.where(mask_cat > 0, ones_rank, zeros_rank + lengths)

        gather_idx = target_pos.unsqueeze(-1).expand(-1, -1, hidden_dim)
        hidden_left = torch.zeros_like(hidden_cat).scatter(1, gather_idx, hidden_cat)
        new_mask = torch.arange(seq_len, device=hidden_cat.device).unsqueeze(0) < lengths
        return hidden_left, new_mask

    module.pack_sequences = _pack_sequences_no_sort


def _parse_layer_set(raw: str) -> set[int]:
    result: set[int] = set()
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        try:
            result.add(int(token))
        except ValueError:
            continue
    return result


def _patch_attention_layers_for_export(model: torch.nn.Module) -> None:
    """Force eager attention path for selected DiT layers during ONNX export.

    This is used to decompose SDPA into explicit ops only for problematic layers.
    """
    raw = os.environ.get("ACESTEP_DIT_SDPA_DECOMP_LAYERS", "12,14,16,18,19,20,21,22,23").strip().lower()
    if raw in {"all", "*"}:
        target_layers = set(range(len(model.decoder.layers)))
    else:
        target_layers = _parse_layer_set(raw)
    if not target_layers:
        print("[export_core] SDPA layer decomposition disabled")
        return

    kind = os.environ.get("ACESTEP_DIT_SDPA_DECOMP_KIND", "both").strip().lower()
    if kind not in {"self", "cross", "both"}:
        kind = "both"

    def _wrap_attention_module(attn_module: torch.nn.Module, layer_idx: int, name: str) -> None:
        original_forward = attn_module.forward

        def _forced_eager_forward(self, *args, **kwargs):
            cfg = getattr(self, "config", None)
            if cfg is None or not hasattr(cfg, "_attn_implementation"):
                return original_forward(*args, **kwargs)
            prev = cfg._attn_implementation
            cfg._attn_implementation = "eager"
            try:
                return original_forward(*args, **kwargs)
            finally:
                cfg._attn_implementation = prev

        attn_module.forward = types.MethodType(_forced_eager_forward, attn_module)
        print(f"[export_core] Force eager decomposition: layer={layer_idx} module={name}")

    for i, layer in enumerate(model.decoder.layers):
        if i not in target_layers:
            continue
        if kind in {"self", "both"}:
            _wrap_attention_module(layer.self_attn, i, "self_attn")
        if kind in {"cross", "both"} and getattr(layer, "use_cross_attention", False):
            _wrap_attention_module(layer.cross_attn, i, "cross_attn")


def _patch_dit_forward_for_onnx(model: torch.nn.Module) -> None:
    """Patch AceStepDiTModel.forward to keep encoder mask dynamic in ONNX."""
    module = importlib.import_module(model.__class__.__module__)
    EncoderDecoderCache = module.EncoderDecoderCache
    DynamicCache = module.DynamicCache
    logger = getattr(module, "logger", None)
    mask_mode = os.environ.get("ACESTEP_DIT_MASK_MODE", "pt_none").strip().lower()
    if mask_mode not in {"pt_none", "use_input"}:
        mask_mode = "pt_none"

    def _forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        timestep_r: torch.Tensor,
        attention_mask: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        context_latents: torch.Tensor,
        use_cache: bool | None = None,
        past_key_values=None,
        cache_position=None,
        position_ids=None,
        output_attentions: bool | None = False,
        return_hidden_states: int | None = None,
        custom_layers_config: dict | None = None,
        enable_early_exit: bool = False,
        **flash_attn_kwargs,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if self.gradient_checkpointing and self.training and use_cache:
            if logger is not None:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
                )
            use_cache = False
        if self.training:
            use_cache = False
        if not self.training and use_cache and past_key_values is None:
            past_key_values = EncoderDecoderCache(DynamicCache(), DynamicCache())

        temb_t, timestep_proj_t = self.time_embed(timestep)
        temb_r, timestep_proj_r = self.time_embed_r(timestep - timestep_r)
        temb = temb_t + temb_r
        timestep_proj = timestep_proj_t + timestep_proj_r

        hidden_states = torch.cat([context_latents, hidden_states], dim=-1)
        original_seq_len = hidden_states.shape[1]
        if hidden_states.shape[1] % self.patch_size != 0:
            pad_length = self.patch_size - (hidden_states.shape[1] % self.patch_size)
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad_length), mode="constant", value=0)

        hidden_states = self.proj_in(hidden_states)
        encoder_hidden_states = self.condition_embedder(encoder_hidden_states)
        if attention_mask is not None and attention_mask.shape[1] != hidden_states.shape[1]:
            if attention_mask.shape[1] % self.patch_size != 0:
                pad_to = self.patch_size - (attention_mask.shape[1] % self.patch_size)
                attention_mask = F.pad(attention_mask, (0, pad_to), mode="constant", value=0.0)
            attention_mask = attention_mask.unflatten(1, (-1, self.patch_size)).amax(dim=-1)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + hidden_states.shape[1], device=hidden_states.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        dtype = hidden_states.dtype
        device = hidden_states.device
        is_flash_attn = self.config._attn_implementation == "flash_attention_2"

        # Keep mask inputs in graph for I/O parity.
        # `pt_none`: preserve PT runtime behavior (mask inputs ignored in mask construction).
        # `use_input`: use incoming 2D masks when constructing 4D masks.
        mask_dummy = hidden_states.new_zeros(())
        if attention_mask is not None:
            mask_dummy = mask_dummy + attention_mask.sum() * 0.0
        if encoder_attention_mask is not None:
            mask_dummy = mask_dummy + encoder_attention_mask.sum() * 0.0
        hidden_states = hidden_states + mask_dummy
        mask_2d = attention_mask if mask_mode == "use_input" else None

        def _build_4d_mask(attn_mask_2d, seq_len_tensor, is_sliding, is_causal):
            indices = torch.arange(seq_len_tensor, device=device)
            diff = indices.unsqueeze(1) - indices.unsqueeze(0)
            valid_mask = torch.ones_like(diff, dtype=torch.bool)
            if is_causal:
                valid_mask = valid_mask & (diff >= 0)
            if is_sliding and self.config.sliding_window is not None:
                if is_causal:
                    valid_mask = valid_mask & (diff <= self.config.sliding_window)
                else:
                    valid_mask = valid_mask & (torch.abs(diff) <= self.config.sliding_window)
            valid_mask = valid_mask.unsqueeze(0).unsqueeze(0)
            if attn_mask_2d is not None:
                padding_mask_4d = attn_mask_2d.view(attn_mask_2d.shape[0], 1, 1, -1).to(torch.bool)
                valid_mask = valid_mask & padding_mask_4d
            min_dtype = torch.finfo(dtype).min
            mask_tensor = torch.full(valid_mask.shape, min_dtype, dtype=dtype, device=device)
            mask_tensor.masked_fill_(valid_mask, 0.0)
            return mask_tensor

        full_attn_mask = None
        sliding_attn_mask = None
        encoder_attention_mask_4d = None
        if is_flash_attn:
            full_attn_mask = None
            sliding_attn_mask = None
            encoder_attention_mask_4d = None
        else:
            seq_len_tensor = torch._shape_as_tensor(hidden_states)[1]
            enc_seq_len_tensor = torch._shape_as_tensor(encoder_hidden_states)[1]
            full_attn_mask = _build_4d_mask(mask_2d, seq_len_tensor, False, False)
            max_len_tensor = torch.maximum(seq_len_tensor, enc_seq_len_tensor)
            encoder_full_mask = _build_4d_mask(None, max_len_tensor, False, False)
            encoder_attention_mask_4d = encoder_full_mask[
                :, :, : hidden_states.shape[1], : encoder_hidden_states.shape[1]
            ]
            if self.config.use_sliding_window:
                sliding_attn_mask = _build_4d_mask(mask_2d, seq_len_tensor, True, False)

        self_attn_mask_mapping = {
            "full_attention": full_attn_mask,
            "sliding_attention": sliding_attn_mask,
            "encoder_attention_mask": encoder_attention_mask_4d,
        }

        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        all_cross_attentions = () if output_attentions else None
        if custom_layers_config is not None and enable_early_exit:
            output_attentions = True
            if all_cross_attentions is None:
                all_cross_attentions = ()

        for index_block, layer_module in enumerate(self.layers):
            layer_outputs = layer_module(
                hidden_states,
                position_embeddings,
                timestep_proj,
                self_attn_mask_mapping[layer_module.attention_type],
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
                encoder_hidden_states,
                self_attn_mask_mapping["encoder_attention_mask"],
                **flash_attn_kwargs,
            )
            hidden_states = layer_outputs[0]
            if output_attentions and self.layers[index_block].use_cross_attention and len(layer_outputs) >= 3:
                all_cross_attentions += (layer_outputs[2],)

        if return_hidden_states:
            return hidden_states

        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
        hidden_states = (self.norm_out(hidden_states) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states[:, :original_seq_len, :]

        outputs = (hidden_states, past_key_values)
        if output_attentions:
            outputs += (all_cross_attentions,)
        return outputs

    model.decoder.forward = types.MethodType(_forward, model.decoder)


def _export_core(
    handler: AceStepHandler,
    out_dir: Path,
    opset: int,
    external_data_policy: str,
) -> tuple[Dict[str, object], list[ExportResult]]:
    device = torch.device("cpu")
    sample = _sample_inputs(device)
    model = handler.model.float().eval().cpu()
    attn_impl = os.environ.get("ACESTEP_DIT_ATTN_IMPL", "eager").strip().lower()
    if attn_impl not in {"eager", "sdpa"}:
        attn_impl = "eager"
    model.config._attn_implementation = attn_impl
    print(f"[export_core] DiT attention implementation for export: {attn_impl}")
    if attn_impl == "sdpa":
        _patch_attention_layers_for_export(model)
    _patch_timbre_encoder_for_onnx(model)
    _patch_pack_sequences_for_onnx(model)
    _patch_dit_forward_for_onnx(model)
    vae = handler.vae.float().eval().cpu()
    num_layers = int(model.config.num_hidden_layers)

    cond = ConditionEncoderWrapper(model).eval()
    dit = DitDecoderWrapper(model.decoder).eval()
    dit_kv = DitDecoderKvWrapper(model.decoder, num_layers).eval()
    tok = AudioTokenizerWrapper(model.tokenizer).eval()
    detok = AudioDetokenizerWrapper(model.detokenizer).eval()
    vae_dec = VaeDecoderWrapper(vae).eval()

    entries: list[ExportResult] = []
    contract: Dict[str, object] = {"inputs": {}, "outputs": {}}

    cond_path = out_dir / "condition_encoder.onnx"

    def _export_cond(use_external_data: bool) -> None:
        torch.onnx.export(
            cond,
            (
                sample["text_hidden_states"],
                sample["text_attention_mask"],
                sample["lyric_hidden_states"],
                sample["lyric_attention_mask"],
                sample["refer_audio_acoustic_hidden_states_packed"],
                sample["refer_audio_order_mask"],
                sample["hidden_states"],
                sample["attention_mask"],
                sample["silence_latent"],
                sample["src_latents"],
                sample["chunk_masks"],
                sample["is_covers"],
                sample["precomputed_lm_hints_25hz"],
            ),
            str(cond_path),
            input_names=[
                "text_hidden_states",
                "text_attention_mask",
                "lyric_hidden_states",
                "lyric_attention_mask",
                "refer_audio_acoustic_hidden_states_packed",
                "refer_audio_order_mask",
                "hidden_states",
                "attention_mask",
                "silence_latent",
                "src_latents",
                "chunk_masks",
                "is_covers",
                "precomputed_lm_hints_25hz",
            ],
            output_names=["encoder_hidden_states", "encoder_attention_mask", "context_latents"],
            dynamic_axes={
                "text_hidden_states": {0: "B", 1: "T_text"},
                "text_attention_mask": {0: "B", 1: "T_text"},
                "lyric_hidden_states": {0: "B", 1: "T_lyric"},
                "lyric_attention_mask": {0: "B", 1: "T_lyric"},
                "hidden_states": {0: "B", 1: "T_latent"},
                "attention_mask": {0: "B", 1: "T_latent"},
                "silence_latent": {0: "B", 1: "T_latent"},
                "src_latents": {0: "B", 1: "T_latent"},
                "chunk_masks": {0: "B", 1: "T_latent"},
                "precomputed_lm_hints_25hz": {0: "B", 1: "T_latent"},
                "encoder_hidden_states": {0: "B", 1: "T_enc"},
                "encoder_attention_mask": {0: "B", 1: "T_enc"},
                "context_latents": {0: "B", 1: "T_latent"},
            },
            opset_version=opset,
            do_constant_folding=False,
            external_data=use_external_data,
        )

    entries.append(export_with_policy(cond_path, external_data_policy, _export_cond))

    dit_path = out_dir / "dit_decoder.onnx"

    def _export_dit(use_external_data: bool) -> None:
        torch.onnx.export(
            dit,
            (
                sample["dit_hidden_states"],
                sample["timestep"],
                sample["timestep_r"],
                sample["attention_mask"],
                sample["encoder_hidden_states"],
                sample["encoder_attention_mask"],
                sample["context_latents"],
            ),
            str(dit_path),
            input_names=[
                "hidden_states",
                "timestep",
                "timestep_r",
                "attention_mask",
                "encoder_hidden_states",
                "encoder_attention_mask",
                "context_latents",
            ],
            output_names=["vt"],
            dynamic_axes={
                "hidden_states": {0: "B", 1: "T_latent"},
                "timestep": {0: "B"},
                "timestep_r": {0: "B"},
                "attention_mask": {0: "B", 1: "T_latent"},
                "encoder_hidden_states": {0: "B", 1: "T_enc"},
                "encoder_attention_mask": {0: "B", 1: "T_enc"},
                "context_latents": {0: "B", 1: "T_latent"},
                "vt": {0: "B", 1: "T_latent"},
            },
            opset_version=opset,
            do_constant_folding=False,
            external_data=use_external_data,
        )

    entries.append(export_with_policy(dit_path, external_data_policy, _export_dit))

    dit_prefill_path = out_dir / "dit_prefill_kv.onnx"
    dit_decode_path = out_dir / "dit_decode_kv.onnx"
    dit_prefill_input_names, dit_prefill_output_names = _build_dit_kv_names(num_layers, with_inputs=False)
    dit_decode_input_names, dit_decode_output_names = _build_dit_kv_names(num_layers, with_inputs=True)

    dit_prefill_dyn = {
        "hidden_states": {0: "B", 1: "T_latent"},
        "timestep": {0: "B"},
        "timestep_r": {0: "B"},
        "attention_mask": {0: "B", 1: "T_latent"},
        "encoder_hidden_states": {0: "B", 1: "T_enc"},
        "encoder_attention_mask": {0: "B", 1: "T_enc"},
        "context_latents": {0: "B", 1: "T_latent"},
        "vt": {0: "B", 1: "T_latent"},
    }
    for i in range(num_layers):
        dit_prefill_dyn[f"present_self_key_{i}"] = {0: "B", 2: "T_cache"}
        dit_prefill_dyn[f"present_self_value_{i}"] = {0: "B", 2: "T_cache"}
        dit_prefill_dyn[f"present_cross_key_{i}"] = {0: "B", 2: "T_cache"}
        dit_prefill_dyn[f"present_cross_value_{i}"] = {0: "B", 2: "T_cache"}

    def _export_dit_prefill_kv(use_external_data: bool) -> None:
        torch.onnx.export(
            dit_kv,
            (
                sample["dit_hidden_states"],
                sample["timestep"],
                sample["timestep_r"],
                sample["attention_mask"],
                sample["encoder_hidden_states"],
                sample["encoder_attention_mask"],
                sample["context_latents"],
            ),
            str(dit_prefill_path),
            input_names=dit_prefill_input_names,
            output_names=dit_prefill_output_names,
            dynamic_axes=dit_prefill_dyn,
            opset_version=opset,
            do_constant_folding=False,
            external_data=use_external_data,
        )

    entries.append(export_with_policy(dit_prefill_path, external_data_policy, _export_dit_prefill_kv))

    with torch.no_grad():
        dit_prefill_out = dit_kv(
            sample["dit_hidden_states"],
            sample["timestep"],
            sample["timestep_r"],
            sample["attention_mask"],
            sample["encoder_hidden_states"],
            sample["encoder_attention_mask"],
            sample["context_latents"],
        )
    dit_past = tuple(dit_prefill_out[1:])

    dit_decode_dyn = {
        "hidden_states": {0: "B", 1: "T_latent"},
        "timestep": {0: "B"},
        "timestep_r": {0: "B"},
        "attention_mask": {0: "B", 1: "T_latent"},
        "encoder_hidden_states": {0: "B", 1: "T_enc"},
        "encoder_attention_mask": {0: "B", 1: "T_enc"},
        "context_latents": {0: "B", 1: "T_latent"},
        "vt": {0: "B", 1: "T_latent"},
    }
    for i in range(num_layers):
        dit_decode_dyn[f"past_self_key_{i}"] = {0: "B", 2: "T_cache"}
        dit_decode_dyn[f"past_self_value_{i}"] = {0: "B", 2: "T_cache"}
        dit_decode_dyn[f"past_cross_key_{i}"] = {0: "B", 2: "T_cache"}
        dit_decode_dyn[f"past_cross_value_{i}"] = {0: "B", 2: "T_cache"}
        dit_decode_dyn[f"present_self_key_{i}"] = {0: "B", 2: "T_cache_next"}
        dit_decode_dyn[f"present_self_value_{i}"] = {0: "B", 2: "T_cache_next"}
        dit_decode_dyn[f"present_cross_key_{i}"] = {0: "B", 2: "T_cache_next"}
        dit_decode_dyn[f"present_cross_value_{i}"] = {0: "B", 2: "T_cache_next"}

    def _export_dit_decode_kv(use_external_data: bool) -> None:
        torch.onnx.export(
            dit_kv,
            (
                sample["dit_hidden_states"],
                sample["timestep"],
                sample["timestep_r"],
                sample["attention_mask"],
                sample["encoder_hidden_states"],
                sample["encoder_attention_mask"],
                sample["context_latents"],
                *dit_past,
            ),
            str(dit_decode_path),
            input_names=dit_decode_input_names,
            output_names=dit_decode_output_names,
            dynamic_axes=dit_decode_dyn,
            opset_version=opset,
            do_constant_folding=False,
            external_data=use_external_data,
        )

    entries.append(export_with_policy(dit_decode_path, external_data_policy, _export_dit_decode_kv))

    tok_path = out_dir / "audio_tokenizer.onnx"

    def _export_audio_tokenizer(use_external_data: bool) -> None:
        torch.onnx.export(
            tok,
            (sample["audio_hidden_states"],),
            str(tok_path),
            input_names=["hidden_states"],
            output_names=["quantized", "indices"],
            dynamic_axes={
                "hidden_states": {0: "B", 1: "T_25hz"},
                "quantized": {0: "B", 1: "T_5hz"},
                "indices": {0: "B", 1: "T_5hz"},
            },
            opset_version=opset,
            do_constant_folding=False,
            external_data=use_external_data,
        )

    entries.append(export_with_policy(tok_path, external_data_policy, _export_audio_tokenizer))

    with torch.no_grad():
        detok_quantized, _ = tok(sample["audio_hidden_states"])

    detok_path = out_dir / "audio_detokenizer.onnx"

    def _export_audio_detokenizer(use_external_data: bool) -> None:
        torch.onnx.export(
            detok,
            (detok_quantized,),
            str(detok_path),
            input_names=["quantized"],
            output_names=["lm_hints_25hz"],
            dynamic_axes={
                "quantized": {0: "B", 1: "T_5hz"},
                "lm_hints_25hz": {0: "B", 1: "T_25hz"},
            },
            opset_version=opset,
            do_constant_folding=False,
            external_data=use_external_data,
        )

    entries.append(export_with_policy(detok_path, external_data_policy, _export_audio_detokenizer))

    vae_path = out_dir / "vae_decoder.onnx"

    def _export_vae_decoder(use_external_data: bool) -> None:
        torch.onnx.export(
            vae_dec,
            (sample["vae_latents"],),
            str(vae_path),
            input_names=["latents"],
            output_names=["audio"],
            dynamic_axes={
                "latents": {0: "B", 2: "T_latent"},
                "audio": {0: "B", 2: "T_audio"},
            },
            opset_version=opset,
            do_constant_folding=False,
            external_data=use_external_data,
        )

    entries.append(export_with_policy(vae_path, external_data_policy, _export_vae_decoder))

    contract["inputs"] = {
        "condition_encoder": entries[0].input_names,
        "dit_decoder": entries[1].input_names,
        "dit_prefill_kv": entries[2].input_names,
        "dit_decode_kv": entries[3].input_names,
        "audio_tokenizer": entries[4].input_names,
        "audio_detokenizer": entries[5].input_names,
        "vae_decoder": entries[6].input_names,
    }
    contract["outputs"] = {
        "condition_encoder": entries[0].output_names,
        "dit_decoder": entries[1].output_names,
        "dit_prefill_kv": entries[2].output_names,
        "dit_decode_kv": entries[3].output_names,
        "audio_tokenizer": entries[4].output_names,
        "audio_detokenizer": entries[5].output_names,
        "vae_decoder": entries[6].output_names,
    }
    contract["num_layers"] = num_layers
    contract["dit_prefill_kv_path"] = "dit_prefill_kv.onnx"
    contract["dit_decode_kv_path"] = "dit_decode_kv.onnx"
    return contract, entries


def main() -> int:
    parser = argparse.ArgumentParser(description="Export ACE-Step core models to ONNX")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--model-path", type=str, default="acestep-v15-turbo")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/onnx_runtime"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--provider", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--external-data-policy", choices=["auto", "single", "external"], default="auto")
    parser.add_argument("--skip-sanity", action="store_true")
    args = parser.parse_args()
    ensure_real_onnx_module()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest or (args.out_dir / "manifest.json")
    handler = _init_handler(args.project_root, args.model_path, args.provider)
    contract, entries = _export_core(handler, args.out_dir, args.opset, args.external_data_policy)

    contract_path = args.out_dir / "io_contract_core.json"
    contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    print(f"Wrote {contract_path}")

    update_manifest(manifest_path, args.out_dir, entries, model_group="core")
    print(f"Updated manifest: {manifest_path}")

    if not args.skip_sanity:
        for entry in entries:
            ok, message = ort_sanity_load(entry.model_path, provider=args.provider)
            prefix = "OK" if ok else "NG"
            print(f"ORT sanity {prefix}: {entry.model_path.name} ({message})")

    print("Core export done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
