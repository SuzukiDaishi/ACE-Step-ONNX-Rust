#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch

from acestep.handler import AceStepHandler
from runtime_py_ort.case_schema import CaseSpec
from runtime_py_ort.scheduler import resolve_timesteps
from tools.onnx_export.common import export_with_policy
from tools.onnx_export.export_core import _patch_dit_forward_for_onnx


def _pad_for_patch2(
    hidden_states: np.ndarray,
    attention_mask: np.ndarray,
    context_latents: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Pad latent-length axis to patch_size=2 so probe ONNX matches runtime inputs."""
    orig_len = int(hidden_states.shape[1])
    pad_len = (-orig_len) % 2
    if pad_len == 0:
        return hidden_states, attention_mask, context_latents, orig_len
    hs = np.pad(hidden_states, ((0, 0), (0, pad_len), (0, 0)), mode="constant")
    attn = np.pad(attention_mask, ((0, 0), (0, pad_len)), mode="constant")
    ctx = np.pad(context_latents, ((0, 0), (0, pad_len), (0, 0)), mode="constant")
    return hs, attn, ctx, orig_len


def _metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    da = a.astype(np.float64, copy=False)
    db = b.astype(np.float64, copy=False)
    diff = da - db
    rmse = float(np.sqrt(np.mean(diff * diff)))
    max_abs = float(np.max(np.abs(diff)))
    denom = float(np.linalg.norm(da.ravel()) * np.linalg.norm(db.ravel()))
    cos_sim = float(np.dot(da.ravel(), db.ravel()) / denom) if denom > 0 else 1.0
    return {"rmse": rmse, "max_abs": max_abs, "cos_sim": cos_sim}


class DitDecoderProbeWrapper(torch.nn.Module):
    def __init__(self, decoder: torch.nn.Module):
        super().__init__()
        self.decoder = decoder
        self.num_layers = len(decoder.layers)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        timestep_r: torch.Tensor,
        attention_mask: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        context_latents: torch.Tensor,
    ):
        layer_outputs: list[torch.Tensor | None] = [None] * self.num_layers
        handles = []
        for idx, layer in enumerate(self.decoder.layers):
            def _hook(_m, _i, out, *, i=idx):
                layer_outputs[i] = out[0]

            handles.append(layer.register_forward_hook(_hook))
        try:
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
        finally:
            for h in handles:
                h.remove()
        vt = out[0]
        missing = [i for i, x in enumerate(layer_outputs) if x is None]
        if missing:
            raise RuntimeError(f"Probe hooks missing layer outputs: {missing}")
        return (vt, *layer_outputs)


def _init_handler(project_root: Path) -> AceStepHandler:
    handler = AceStepHandler()
    status, ok = handler.initialize_service(
        project_root=str(project_root),
        config_path="acestep-v15-turbo",
        device="cpu",
        use_flash_attention=False,
        compile_model=False,
        offload_to_cpu=False,
        offload_dit_to_cpu=False,
    )
    if not ok:
        raise RuntimeError(f"Failed to initialize AceStepHandler: {status}")
    return handler


def _export_probe_if_needed(
    handler: AceStepHandler,
    probe_path: Path,
    *,
    force: bool,
    external_data_policy: str,
    sample_hidden_states: np.ndarray,
    sample_attention_mask: np.ndarray,
    sample_encoder_hidden_states: np.ndarray,
    sample_encoder_attention_mask: np.ndarray,
    sample_context_latents: np.ndarray,
) -> tuple[list[str], list[str]]:
    if probe_path.exists() and not force:
        sess = ort.InferenceSession(str(probe_path), providers=["CPUExecutionProvider"])
        in_names = [x.name for x in sess.get_inputs()]
        out_names = [x.name for x in sess.get_outputs()]
        return in_names, out_names

    model = handler.model.float().eval().cpu()
    model.config._attn_implementation = "eager"
    _patch_dit_forward_for_onnx(model)

    probe = DitDecoderProbeWrapper(model.decoder).eval()
    num_layers = probe.num_layers
    output_names = ["vt"] + [f"layer_{i:02d}" for i in range(num_layers)]
    input_names = [
        "hidden_states",
        "timestep",
        "timestep_r",
        "attention_mask",
        "encoder_hidden_states",
        "encoder_attention_mask",
        "context_latents",
    ]
    dynamic_axes: dict[str, dict[int, str]] = {
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
        dynamic_axes[f"layer_{i:02d}"] = {0: "B", 1: "T_latent_or_patched"}

    sample_t = np.array([1.0], dtype=np.float32)
    sample_tuple = (
        torch.from_numpy(sample_hidden_states.astype(np.float32)),
        torch.from_numpy(sample_t),
        torch.from_numpy(sample_t),
        torch.from_numpy(sample_attention_mask.astype(np.float32)),
        torch.from_numpy(sample_encoder_hidden_states.astype(np.float32)),
        torch.from_numpy(sample_encoder_attention_mask.astype(np.float32)),
        torch.from_numpy(sample_context_latents.astype(np.float32)),
    )

    probe_path.parent.mkdir(parents=True, exist_ok=True)

    def _export(use_external_data: bool) -> None:
        torch.onnx.export(
            probe,
            sample_tuple,
            str(probe_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=17,
            do_constant_folding=False,
            external_data=use_external_data,
        )

    export_with_policy(probe_path, external_data_policy, _export)
    return input_names, output_names


def _run_pt_step(
    decoder: torch.nn.Module,
    *,
    hidden_states: np.ndarray,
    t_value: float,
    attention_mask: np.ndarray,
    encoder_hidden_states: np.ndarray,
    encoder_attention_mask: np.ndarray,
    context_latents: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray]]:
    device = next(decoder.parameters()).device
    dtype = next(decoder.parameters()).dtype
    hs = torch.from_numpy(hidden_states).to(device=device, dtype=dtype)
    t = torch.full((hs.shape[0],), float(t_value), device=device, dtype=dtype)
    attn = torch.from_numpy(attention_mask).to(device=device, dtype=dtype)
    enc = torch.from_numpy(encoder_hidden_states).to(device=device, dtype=dtype)
    enc_attn = torch.from_numpy(encoder_attention_mask).to(device=device, dtype=dtype)
    ctx = torch.from_numpy(context_latents).to(device=device, dtype=dtype)

    layer_outputs: list[torch.Tensor | None] = [None] * len(decoder.layers)
    handles = []
    for idx, layer in enumerate(decoder.layers):
        def _hook(_m, _i, out, *, i=idx):
            layer_outputs[i] = out[0].detach().to(torch.float32).cpu()

        handles.append(layer.register_forward_hook(_hook))

    with torch.no_grad():
        out = decoder(
            hidden_states=hs,
            timestep=t,
            timestep_r=t,
            attention_mask=attn,
            encoder_hidden_states=enc,
            encoder_attention_mask=enc_attn,
            context_latents=ctx,
            use_cache=False,
            past_key_values=None,
        )
    for h in handles:
        h.remove()

    vt = out[0].detach().to(torch.float32).cpu().numpy()
    layers = []
    for i, l in enumerate(layer_outputs):
        if l is None:
            raise RuntimeError(f"PT hook did not capture layer {i}")
        layers.append(l.numpy())
    return vt, layers


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare PT vs ONNX DiT per-layer tensors")
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--baseline-npz", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--onnx-dir", type=Path, default=Path("artifacts/onnx_runtime"))
    parser.add_argument("--probe-onnx", type=Path, default=None)
    parser.add_argument("--reexport-probe", action="store_true")
    parser.add_argument("--external-data-policy", choices=["auto", "single", "external"], default="auto")
    parser.add_argument("--provider", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--pt-attn-impl", choices=["sdpa", "eager"], default="sdpa")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--rmse-threshold", type=float, default=5e-4)
    parser.add_argument("--max-abs-threshold", type=float, default=1e-1)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    case = CaseSpec.from_path(args.case)
    arr = np.load(str(args.baseline_npz), allow_pickle=False)
    xt_steps = arr["xt_steps"].astype(np.float32)
    latent_masks = arr["latent_masks"].astype(np.float32) if "latent_masks" in arr else np.ones(
        (xt_steps.shape[1], xt_steps.shape[2]), dtype=np.float32
    )
    encoder_hidden_states = arr["encoder_hidden_states"].astype(np.float32)
    encoder_attention_mask = arr["encoder_attention_mask"].astype(np.float32)
    context_latents = arr["context_latents"].astype(np.float32)

    onnx_dir = (project_root / args.onnx_dir).resolve()
    probe_path = args.probe_onnx.resolve() if args.probe_onnx else onnx_dir / "dit_decoder_probe.onnx"

    handler = _init_handler(project_root)
    decoder = handler.model.decoder.eval()
    if hasattr(handler.model, "config"):
        handler.model.config._attn_implementation = args.pt_attn_impl

    input_names, output_names = _export_probe_if_needed(
        handler,
        probe_path,
        force=bool(args.reexport_probe),
        external_data_policy=args.external_data_policy,
        sample_hidden_states=_pad_for_patch2(xt_steps[0], latent_masks, context_latents)[0],
        sample_attention_mask=_pad_for_patch2(xt_steps[0], latent_masks, context_latents)[1],
        sample_encoder_hidden_states=encoder_hidden_states,
        sample_encoder_attention_mask=encoder_attention_mask,
        sample_context_latents=_pad_for_patch2(xt_steps[0], latent_masks, context_latents)[2],
    )

    providers = ["CPUExecutionProvider"]
    if args.provider == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(str(probe_path), providers=providers)

    timesteps = resolve_timesteps(case.shift, None, max_steps=max(1, int(case.inference_steps)))
    max_steps = min(len(timesteps), int(args.max_steps), xt_steps.shape[0])

    layer_count = len(output_names) - 1
    step_summaries: list[dict[str, Any]] = []
    global_worst = {
        "step": None,
        "layer": None,
        "rmse": -1.0,
        "max_abs": -1.0,
    }

    for step in range(max_steps):
        t = float(timesteps[step])
        hs_raw = xt_steps[step]
        hs, attn_mask, ctx_latents, orig_len = _pad_for_patch2(hs_raw, latent_masks, context_latents)
        feeds = {
            "hidden_states": hs,
            "timestep": np.full((hs.shape[0],), t, dtype=np.float32),
            "timestep_r": np.full((hs.shape[0],), t, dtype=np.float32),
            "attention_mask": attn_mask,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": encoder_attention_mask,
            "context_latents": ctx_latents,
        }
        feed_filtered = {k: v for k, v in feeds.items() if k in set(input_names)}

        onnx_out = sess.run(output_names, feed_filtered)
        onnx_vt = onnx_out[0].astype(np.float32, copy=False)[:, :orig_len, :]
        onnx_layers = [x.astype(np.float32, copy=False) for x in onnx_out[1:]]

        pt_vt, pt_layers = _run_pt_step(
            decoder,
            hidden_states=hs,
            t_value=t,
            attention_mask=attn_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            context_latents=ctx_latents,
        )
        pt_vt = pt_vt[:, :orig_len, :]

        vt_metrics = _metrics(pt_vt, onnx_vt)
        per_layer: list[dict[str, Any]] = []
        worst_layer_idx = 0
        worst_max_abs = -1.0
        for i in range(layer_count):
            m = _metrics(pt_layers[i], onnx_layers[i])
            row = {
                "layer": i,
                "name": f"layer_{i:02d}",
                **m,
                "fail": m["rmse"] > args.rmse_threshold or m["max_abs"] > args.max_abs_threshold,
            }
            per_layer.append(row)
            if m["max_abs"] > worst_max_abs:
                worst_max_abs = m["max_abs"]
                worst_layer_idx = i

        worst = per_layer[worst_layer_idx]
        if worst["max_abs"] > float(global_worst["max_abs"]):
            global_worst = {
                "step": step,
                "layer": worst_layer_idx,
                "rmse": worst["rmse"],
                "max_abs": worst["max_abs"],
            }

        step_summaries.append(
            {
                "step": step,
                "timestep": t,
                "vt": vt_metrics,
                "worst_layer": worst,
                "layers": per_layer,
            }
        )

        del onnx_out, onnx_vt, onnx_layers, pt_vt, pt_layers
        gc.collect()

    first_divergence = None
    for row in step_summaries:
        vt = row["vt"]
        if vt["rmse"] > args.rmse_threshold or vt["max_abs"] > args.max_abs_threshold:
            first_divergence = row["step"]
            break

    result = {
        "case_id": case.case_id,
        "case_path": str(args.case),
        "baseline_npz": str(args.baseline_npz),
        "probe_onnx": str(probe_path),
        "thresholds": {
            "rmse": args.rmse_threshold,
            "max_abs": args.max_abs_threshold,
        },
        "first_vt_divergence_step": first_divergence,
        "global_worst_layer": global_worst,
        "steps": step_summaries,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote: {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
