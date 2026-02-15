#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    orig_len = int(hidden_states.shape[1])
    pad_len = (-orig_len) % 2
    if pad_len == 0:
        return hidden_states, attention_mask, context_latents, orig_len, orig_len
    hs = np.pad(hidden_states, ((0, 0), (0, pad_len), (0, 0)), mode="constant")
    attn = np.pad(attention_mask, ((0, 0), (0, pad_len)), mode="constant")
    ctx = np.pad(context_latents, ((0, 0), (0, pad_len), (0, 0)), mode="constant")
    return hs, attn, ctx, orig_len, int(hs.shape[1])


def _metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    if a.shape != b.shape:
        min_len = min(a.shape[1], b.shape[1])
        a = a[:, :min_len, :]
        b = b[:, :min_len, :]
    da = a.astype(np.float64, copy=False)
    db = b.astype(np.float64, copy=False)
    diff = da - db
    rmse = float(np.sqrt(np.mean(diff * diff)))
    max_abs = float(np.max(np.abs(diff)))
    denom = float(np.linalg.norm(da.ravel()) * np.linalg.norm(db.ravel()))
    cos_sim = float(np.dot(da.ravel(), db.ravel()) / denom) if denom > 0 else 1.0
    return {"rmse": rmse, "max_abs": max_abs, "cos_sim": cos_sim}


class DitDecoderSubLayerProbeWrapper(torch.nn.Module):
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
        self_out: list[torch.Tensor | None] = [None] * self.num_layers
        cross_out: list[torch.Tensor | None] = [None] * self.num_layers
        mlp_out: list[torch.Tensor | None] = [None] * self.num_layers
        handles = []

        for idx, layer in enumerate(self.decoder.layers):
            def _hook_self(_m, _i, out, *, i=idx):
                val = out[0] if isinstance(out, (tuple, list)) else out
                self_out[i] = val

            def _hook_cross(_m, _i, out, *, i=idx):
                val = out[0] if isinstance(out, (tuple, list)) else out
                cross_out[i] = val

            def _hook_mlp(_m, _i, out, *, i=idx):
                mlp_out[i] = out

            handles.append(layer.self_attn.register_forward_hook(_hook_self))
            if getattr(layer, "use_cross_attention", False):
                handles.append(layer.cross_attn.register_forward_hook(_hook_cross))
            handles.append(layer.mlp.register_forward_hook(_hook_mlp))

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
        fill = hidden_states
        for i in range(self.num_layers):
            if self_out[i] is None:
                self_out[i] = torch.zeros_like(fill)
            if cross_out[i] is None:
                cross_out[i] = torch.zeros_like(fill)
            if mlp_out[i] is None:
                mlp_out[i] = torch.zeros_like(fill)

        outputs = [vt]
        for i in range(self.num_layers):
            outputs.append(self_out[i])
            outputs.append(cross_out[i])
            outputs.append(mlp_out[i])
        return tuple(outputs)


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

    probe = DitDecoderSubLayerProbeWrapper(model.decoder).eval()
    num_layers = probe.num_layers
    output_names = ["vt"]
    for i in range(num_layers):
        output_names += [f"self_{i:02d}", f"cross_{i:02d}", f"mlp_{i:02d}"]
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
        dynamic_axes[f"self_{i:02d}"] = {0: "B", 1: "T_latent_or_patched"}
        dynamic_axes[f"cross_{i:02d}"] = {0: "B", 1: "T_latent_or_patched"}
        dynamic_axes[f"mlp_{i:02d}"] = {0: "B", 1: "T_latent_or_patched"}

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


def _parse_steps(raw: str | None, max_steps: int) -> list[int]:
    if not raw:
        return list(range(max_steps))
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return [i for i in out if 0 <= i < max_steps]


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare PT vs ONNX DiT sub-layer tensors")
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--baseline-npz", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--onnx-dir", type=Path, default=Path("artifacts/onnx_runtime"))
    parser.add_argument("--probe-onnx", type=Path, default=None)
    parser.add_argument("--reexport-probe", action="store_true")
    parser.add_argument("--external-data-policy", choices=["auto", "single", "external"], default="auto")
    parser.add_argument("--provider", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--steps", type=str, default="")
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

    handler = _init_handler(project_root)
    handler.model.config._attn_implementation = args.pt_attn_impl
    decoder = handler.model.decoder.float().eval().cpu()

    sample_hidden_states = xt_steps[0].astype(np.float32)
    sample_attention_mask = latent_masks.astype(np.float32)
    sample_encoder_hidden_states = encoder_hidden_states.astype(np.float32)
    sample_encoder_attention_mask = encoder_attention_mask.astype(np.float32)
    sample_context_latents = context_latents.astype(np.float32)

    probe_path = (
        args.probe_onnx
        if args.probe_onnx is not None
        else (args.onnx_dir / "dit_decoder_sublayer_probe.onnx")
    )
    in_names, out_names = _export_probe_if_needed(
        handler,
        probe_path,
        force=args.reexport_probe,
        external_data_policy=args.external_data_policy,
        sample_hidden_states=sample_hidden_states,
        sample_attention_mask=sample_attention_mask,
        sample_encoder_hidden_states=sample_encoder_hidden_states,
        sample_encoder_attention_mask=sample_encoder_attention_mask,
        sample_context_latents=sample_context_latents,
    )

    sess = ort.InferenceSession(
        str(probe_path),
        providers=["CPUExecutionProvider"] if args.provider == "cpu" else ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    timesteps = resolve_timesteps(case.shift, None, max_steps=max(1, int(case.inference_steps)))
    step_indices = _parse_steps(args.steps, len(timesteps))
    steps_out: list[dict[str, Any]] = []
    global_worst = {"step": None, "name": "", "rmse": 0.0, "max_abs": 0.0}

    for step_idx in step_indices:
        t = float(timesteps[step_idx])
        xt = xt_steps[step_idx].astype(np.float32)
        hs, attn, ctx, orig_len, padded_len = _pad_for_patch2(xt, latent_masks, context_latents)
        t_vec = np.full((hs.shape[0],), t, dtype=np.float32)
        feeds = {
            "hidden_states": hs,
            "timestep": t_vec,
            "timestep_r": t_vec,
            "attention_mask": attn,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": encoder_attention_mask,
            "context_latents": ctx,
        }

        # PT sublayer outputs
        layer_outputs: dict[str, np.ndarray] = {}
        with torch.no_grad():
            out = decoder(
                hidden_states=torch.from_numpy(xt),
                timestep=torch.full((xt.shape[0],), t),
                timestep_r=torch.full((xt.shape[0],), t),
                attention_mask=torch.from_numpy(latent_masks),
                encoder_hidden_states=torch.from_numpy(encoder_hidden_states),
                encoder_attention_mask=torch.from_numpy(encoder_attention_mask),
                context_latents=torch.from_numpy(context_latents),
                use_cache=False,
                past_key_values=None,
            )
        vt_pt = out[0].detach().to(torch.float32).cpu().numpy()

        # ONNX sublayer outputs
        ort_out = sess.run(out_names, feeds)
        ort_map = {name: val for name, val in zip(out_names, ort_out)}
        vt_onnx = ort_map["vt"]
        if vt_onnx.shape[1] != orig_len:
            vt_onnx = vt_onnx[:, :orig_len, :]

        vt_m = _metrics(vt_pt, vt_onnx)
        step_entry: dict[str, Any] = {"step": step_idx, "timestep": t, "vt": vt_m, "sublayers": []}

        for name in out_names:
            if name == "vt":
                continue
            v = ort_map[name]
            if v.shape[1] != padded_len:
                v = v[:, :padded_len, :]
            if name not in layer_outputs:
                layer_outputs[name] = v

        # PT outputs from probe: reuse ONNX outputs shape, compute by hook via export probe
        # We obtain PT values by re-running the probe wrapper to keep ordering identical.
        probe = DitDecoderSubLayerProbeWrapper(decoder).eval()
        with torch.no_grad():
            pt_out = probe(
                hidden_states=torch.from_numpy(xt),
                timestep=torch.full((xt.shape[0],), t),
                timestep_r=torch.full((xt.shape[0],), t),
                attention_mask=torch.from_numpy(latent_masks),
                encoder_hidden_states=torch.from_numpy(encoder_hidden_states),
                encoder_attention_mask=torch.from_numpy(encoder_attention_mask),
                context_latents=torch.from_numpy(context_latents),
            )
        pt_map = {}
        for name, val in zip(out_names, pt_out):
            if name == "vt":
                continue
            arr = val.detach().to(torch.float32).cpu().numpy()
            if arr.shape[1] != padded_len:
                arr = arr[:, :padded_len, :]
            pt_map[name] = arr

        worst = {"name": "", "rmse": 0.0, "max_abs": 0.0}
        for name in out_names:
            if name == "vt":
                continue
            m = _metrics(pt_map[name], layer_outputs[name])
            fail = (m["rmse"] > args.rmse_threshold) or (m["max_abs"] > args.max_abs_threshold)
            step_entry["sublayers"].append({"name": name, **m, "fail": fail})
            if m["max_abs"] > worst["max_abs"]:
                worst = {"name": name, "rmse": m["rmse"], "max_abs": m["max_abs"]}
            if m["max_abs"] > global_worst["max_abs"]:
                global_worst = {"step": step_idx, "name": name, "rmse": m["rmse"], "max_abs": m["max_abs"]}

        step_entry["worst_sublayer"] = worst
        steps_out.append(step_entry)

    out = {
        "case_id": case.case_id,
        "case_path": str(args.case),
        "baseline_npz": str(args.baseline_npz),
        "probe_onnx": str(probe_path),
        "thresholds": {"rmse": args.rmse_threshold, "max_abs": args.max_abs_threshold},
        "global_worst_sublayer": global_worst,
        "steps": steps_out,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote: {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
