#!/usr/bin/env python3
"""Run current ACE-Step pipeline and dump baseline artifacts for one fixture case."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from acestep.handler import AceStepHandler
from acestep.inference import GenerationConfig, GenerationParams, create_sample, generate_music
from acestep.llm_inference import LLMHandler


def _load_case(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Case JSON must be an object")
    return data


def _to_int(v: Any) -> Optional[int]:
    if v is None or v == "":
        return None
    return int(float(v))


def _to_float(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    return float(v)


def _case_to_params(
    case: Dict[str, Any],
    llm: Optional[LLMHandler],
    *,
    simple_mode_temperature: float,
    simple_mode_top_k: Optional[int],
    simple_mode_top_p: Optional[float],
    simple_mode_repetition_penalty: float,
) -> GenerationParams:
    mode = str(case.get("mode", "text2music"))
    md = case.get("metadata", {}) or {}
    thinking = bool(case.get("thinking", False))

    if mode == "text2music":
        return GenerationParams(
            task_type="text2music",
            caption=str(case.get("caption", "")),
            lyrics=str(case.get("lyrics", "")),
            instrumental=bool(md.get("instrumental", False)),
            vocal_language=str(md.get("vocal_language", "unknown")),
            bpm=_to_int(md.get("bpm")),
            keyscale=str(md.get("keyscale", "")),
            timesignature=str(md.get("timesignature", "")),
            duration=_to_float(md.get("duration")) or -1.0,
            thinking=thinking,
            inference_steps=int(case.get("inference_steps", 8)),
            shift=float(case.get("shift", 3.0)),
            infer_method=str(case.get("infer_method", "ode")),
            use_cot_metas=False,
            use_cot_caption=False,
            use_cot_language=False,
        )

    if mode == "simple_mode":
        if llm is None:
            raise RuntimeError("LLM handler is required for simple_mode")
        query = str(case.get("simple_mode_query", "")).strip()
        if not query:
            raise ValueError("simple_mode_query is required for simple_mode")
        sample = create_sample(
            llm_handler=llm,
            query=query,
            instrumental=bool(md.get("instrumental", False)),
            vocal_language=str(md.get("vocal_language", "unknown")),
            temperature=float(simple_mode_temperature),
            top_k=simple_mode_top_k,
            top_p=simple_mode_top_p,
            repetition_penalty=float(simple_mode_repetition_penalty),
            use_constrained_decoding=True,
        )
        if not sample.success:
            raise RuntimeError(sample.error or sample.status_message or "create_sample failed")

        return GenerationParams(
            task_type="text2music",
            caption=sample.caption,
            lyrics=sample.lyrics,
            instrumental=sample.instrumental,
            vocal_language=sample.language or str(md.get("vocal_language", "unknown")),
            bpm=sample.bpm,
            keyscale=sample.keyscale,
            timesignature=sample.timesignature,
            duration=sample.duration if sample.duration is not None else -1.0,
            thinking=thinking,
            inference_steps=int(case.get("inference_steps", 8)),
            shift=float(case.get("shift", 3.0)),
            infer_method=str(case.get("infer_method", "ode")),
            use_cot_metas=False,
            use_cot_caption=False,
            use_cot_language=False,
        )

    raise ValueError(f"Unsupported mode: {mode}")


def _init_handlers(
    project_root: Path,
    case: Dict[str, Any],
    lm_model_path: str,
    device: str,
    use_flash_attention: bool,
) -> Tuple[AceStepHandler, Optional[LLMHandler]]:
    dit = AceStepHandler()
    status, ok = dit.initialize_service(
        project_root=str(project_root),
        config_path="acestep-v15-turbo",
        device=device,
        use_flash_attention=use_flash_attention,
        compile_model=False,
        offload_to_cpu=False,
        offload_dit_to_cpu=False,
    )
    if not ok:
        raise RuntimeError(f"DiT init failed: {status}")

    llm: Optional[LLMHandler] = None
    if str(case.get("mode", "text2music")) == "simple_mode":
        llm = LLMHandler()
        status, ok = llm.initialize(
            checkpoint_dir=str(project_root / "checkpoints"),
            lm_model_path=lm_model_path,
            backend="pt",
            device=device,
            offload_to_cpu=False,
            dtype=dit.dtype,
        )
        if not ok:
            raise RuntimeError(f"LLM init failed: {status}")

    return dit, llm


def _attach_decoder_trace(dit: AceStepHandler) -> Tuple[List[torch.Tensor], List[torch.Tensor], Any]:
    decoder = dit.model.decoder
    original_forward = decoder.forward
    xt_steps: List[torch.Tensor] = []
    vt_steps: List[torch.Tensor] = []

    def traced_forward(*args, **kwargs):
        hidden_states = kwargs.get("hidden_states")
        if hidden_states is None and len(args) > 0 and torch.is_tensor(args[0]):
            hidden_states = args[0]

        out = original_forward(*args, **kwargs)
        vt = out[0] if isinstance(out, (list, tuple)) and len(out) > 0 and torch.is_tensor(out[0]) else None

        if torch.is_tensor(hidden_states) and torch.is_tensor(vt):
            xt_steps.append(hidden_states.detach().cpu().to(torch.float32))
            vt_steps.append(vt.detach().cpu().to(torch.float32))

        return out

    decoder.forward = traced_forward
    return xt_steps, vt_steps, original_forward


def _save_npz(npz_path: Path, result_extra: Dict[str, Any], xt_steps: List[torch.Tensor], vt_steps: List[torch.Tensor]) -> None:
    arrays: Dict[str, np.ndarray] = {}

    for key in [
        "pred_latents",
        "target_latents",
        "src_latents",
        "chunk_masks",
        "latent_masks",
        "encoder_hidden_states",
        "encoder_attention_mask",
        "context_latents",
        "lyric_token_idss",
        "text_hidden_states",
        "text_attention_mask",
        "lyric_hidden_states",
        "lyric_attention_mask",
        "refer_audio_acoustic_hidden_states_packed",
        "refer_audio_order_mask",
        "is_covers",
        "precomputed_lm_hints_25hz",
        "silence_latent",
    ]:
        val = result_extra.get(key)
        if torch.is_tensor(val):
            arrays[key] = val.detach().to(torch.float32).cpu().numpy()

    if xt_steps:
        arrays["xt_steps"] = torch.stack(xt_steps, dim=0).numpy()
    if vt_steps:
        arrays["vt_steps"] = torch.stack(vt_steps, dim=0).numpy()

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(npz_path), **arrays)


def _collect_condition_inputs(
    dit: AceStepHandler,
    batch: Dict[str, Any],
    processed_data: Tuple[Any, ...],
) -> Dict[str, torch.Tensor]:
    (
        _keys,
        _text_inputs,
        src_latents,
        _target_latents,
        text_hidden_states,
        text_attention_mask,
        lyric_hidden_states,
        lyric_attention_mask,
        audio_attention_mask,
        refer_audio_acoustic_hidden_states_packed,
        refer_audio_order_mask,
        chunk_mask,
        _spans,
        is_covers,
        _audio_codes,
        _lyric_token_idss,
        precomputed_lm_hints_25Hz,
        _non_cover_text_hidden_states,
        _non_cover_text_attention_masks,
    ) = processed_data

    # Ensure silence latent is on device for tokenize/detokenize.
    dit._ensure_silence_latent_on_device()
    silence_latent = dit.silence_latent

    if precomputed_lm_hints_25Hz is None:
        with torch.no_grad():
            lm_hints_5Hz, _, _ = dit.model.tokenize(src_latents, silence_latent, audio_attention_mask)
            precomputed_lm_hints_25Hz = dit.model.detokenize(lm_hints_5Hz)
            precomputed_lm_hints_25Hz = precomputed_lm_hints_25Hz[:, :src_latents.shape[1], :]

    return {
        "text_hidden_states": text_hidden_states,
        "text_attention_mask": text_attention_mask,
        "lyric_hidden_states": lyric_hidden_states,
        "lyric_attention_mask": lyric_attention_mask,
        "refer_audio_acoustic_hidden_states_packed": refer_audio_acoustic_hidden_states_packed,
        "refer_audio_order_mask": refer_audio_order_mask,
        "chunk_masks": chunk_mask,
        "is_covers": is_covers,
        "precomputed_lm_hints_25hz": precomputed_lm_hints_25Hz,
        "silence_latent": silence_latent,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Dump baseline tensors and audio for one case")
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, default=Path("fixtures"))
    parser.add_argument("--lm-model-path", type=str, default="acestep-5Hz-lm-1.7B")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--use-flash-attention", action="store_true")
    parser.add_argument("--simple-mode-temperature", type=float, default=0.0)
    parser.add_argument("--simple-mode-top-k", type=int, default=0)
    parser.add_argument("--simple-mode-top-p", type=float, default=1.0)
    parser.add_argument("--simple-mode-repetition-penalty", type=float, default=1.0)
    args = parser.parse_args()

    case = _load_case(args.case)
    case_id = str(case.get("case_id", args.case.stem))
    case_seed = int(case.get("seed", 42))
    random.seed(case_seed)
    np.random.seed(case_seed)
    torch.manual_seed(case_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(case_seed)
        torch.cuda.manual_seed_all(case_seed)

    dit, llm = _init_handlers(
        args.project_root,
        case,
        args.lm_model_path,
        device=str(args.device),
        use_flash_attention=bool(args.use_flash_attention),
    )
    xt_steps, vt_steps, original_forward = _attach_decoder_trace(dit)

    try:
        params = _case_to_params(
            case,
            llm,
            simple_mode_temperature=float(args.simple_mode_temperature),
            simple_mode_top_k=(None if int(args.simple_mode_top_k) <= 0 else int(args.simple_mode_top_k)),
            simple_mode_top_p=(None if float(args.simple_mode_top_p) >= 1.0 else float(args.simple_mode_top_p)),
            simple_mode_repetition_penalty=float(args.simple_mode_repetition_penalty),
        )
        cfg = GenerationConfig(
            batch_size=1,
            use_random_seed=False,
            seeds=[int(case.get("seed", 42))],
            audio_format=str(case.get("audio_format", "wav")),
            allow_lm_batch=False,
        )

        # Precompute condition encoder inputs for Rust parity.
        duration = params.duration if params.duration is not None and params.duration > 0 else 30.0
        target_wavs = dit.create_target_wavs(float(duration)).unsqueeze(0)
        refer_audios = [[torch.zeros(2, int(30 * dit.sample_rate))]]
        batch = dit._prepare_batch(
            captions=[params.caption],
            lyrics=[params.lyrics],
            keys=None,
            target_wavs=target_wavs,
            refer_audios=refer_audios,
            metas=[{
                "bpm": params.bpm,
                "keyscale": params.keyscale,
                "timesignature": params.timesignature,
                "duration": params.duration,
                "instrumental": params.instrumental,
                "vocal_language": params.vocal_language,
            }],
            vocal_languages=[params.vocal_language],
            repainting_start=None,
            repainting_end=None,
            instructions=None,
            audio_code_hints=None,
            audio_cover_strength=1.0,
        )
        processed_data = dit.preprocess_batch(batch)
        condition_inputs = _collect_condition_inputs(dit, batch, processed_data)

        out_dir = args.output_dir / "audio"
        out_dir.mkdir(parents=True, exist_ok=True)

        result = generate_music(
            dit_handler=dit,
            llm_handler=llm,
            params=params,
            config=cfg,
            save_dir=str(out_dir),
        )

        if not result.success:
            raise RuntimeError(result.error or result.status_message or "generation failed")
        if not result.audios:
            raise RuntimeError("No audio output found")

        # Merge condition inputs into extra outputs for NPZ dump.
        result.extra_outputs.update(condition_inputs)

        src_audio_path = Path(result.audios[0].get("path", ""))
        if not src_audio_path.exists():
            raise FileNotFoundError(f"Generated audio not found: {src_audio_path}")

        dst_audio_path = args.output_dir / "audio" / f"{case_id}.wav"
        if src_audio_path.resolve() != dst_audio_path.resolve():
            shutil.copy2(src_audio_path, dst_audio_path)
            # generate_music writes a UUID filename under save_dir; keep only deterministic case-id output.
            if src_audio_path.parent.resolve() == (args.output_dir / "audio").resolve():
                src_audio_path.unlink(missing_ok=True)

        npz_path = args.output_dir / "tensors" / f"{case_id}.npz"
        _save_npz(npz_path, result.extra_outputs, xt_steps, vt_steps)

        summary = {
            "case_id": case_id,
            "case_path": str(args.case),
            "audio_path": str(dst_audio_path),
            "tensor_path": str(npz_path),
            "num_decoder_steps": len(vt_steps),
            "seed": int(case.get("seed", 42)),
            "success": True,
        }
        summary_path = args.output_dir / "tensors" / f"{case_id}.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        print(f"Saved audio : {dst_audio_path}")
        print(f"Saved tensor: {npz_path}")
        print(f"Summary     : {summary_path}")
        return 0

    finally:
        dit.model.decoder.forward = original_forward


if __name__ == "__main__":
    raise SystemExit(main())
