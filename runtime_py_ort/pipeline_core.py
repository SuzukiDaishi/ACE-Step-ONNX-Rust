from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .case_schema import CaseSpec
from .pipeline_embed import EmbeddingPipeline
from .scheduler import ode_step, resolve_timesteps, x0_from_noise
from .session_manager import OrtSessionManager

DEFAULT_DIT_INSTRUCTION = "Fill the audio semantic mask based on the given conditions:"
SFT_GEN_PROMPT = """# Instruction
{}

# Caption
{}

# Metas
{}<|endoftext|>
"""


def _format_instruction(instruction: str) -> str:
    if instruction.endswith(":"):
        return instruction
    return instruction + ":"


def _meta_to_text(meta: object) -> str:
    if isinstance(meta, str):
        return meta
    if isinstance(meta, dict):
        bpm = meta.get("bpm", meta.get("tempo", "N/A"))
        timesig = meta.get("timesignature", meta.get("time_signature", "N/A"))
        keyscale = meta.get("keyscale", meta.get("key", meta.get("scale", "N/A")))
        duration = meta.get("duration", meta.get("length", 30))
        if isinstance(duration, (int, float)):
            duration_str = f"{int(duration)} seconds"
        elif isinstance(duration, str):
            duration_str = duration
        else:
            duration_str = "30 seconds"
        return (
            f"- bpm: {bpm}\n"
            f"- timesignature: {timesig}\n"
            f"- keyscale: {keyscale}\n"
            f"- duration: {duration_str}\n"
        )
    return "- bpm: N/A\n- timesignature: N/A\n- keyscale: N/A\n- duration: 30 seconds\n"


def _format_lyrics(lyrics: str, language: str) -> str:
    return f"# Languages\n{language}\n\n# Lyric\n{lyrics}<|endoftext|>"


class CorePipeline:
    """Core ORT pipeline.

    Parity-first implementation that can consume precomputed baseline tensors.
    Optional Qwen embedding ONNX path can override text/lyric features online.
    """

    def __init__(
        self,
        onnx_dir: Path,
        provider: str = "cpu",
        qwen_tokenizer_path: Path | None = None,
    ):
        self.onnx_dir = Path(onnx_dir)
        self.sessions = OrtSessionManager(self.onnx_dir)
        self.sessions.config.provider = provider
        self.qwen_embed: Optional[EmbeddingPipeline] = None
        self.qwen_tokenizer_path = qwen_tokenizer_path

    def _load_contract(self) -> Dict[str, List[str]]:
        path = self.onnx_dir / "io_contract_core.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing core contract file: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _run_condition_encoder(
        self,
        arr: Dict[str, np.ndarray],
        contract: Dict[str, List[str]],
        src_latents: np.ndarray,
        attention_mask: np.ndarray,
        prefer_precomputed_condition: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if prefer_precomputed_condition and all(
            k in arr for k in ("encoder_hidden_states", "encoder_attention_mask", "context_latents")
        ):
            return (
                arr["encoder_hidden_states"].astype(np.float32),
                arr["encoder_attention_mask"].astype(np.float32),
                arr["context_latents"].astype(np.float32),
            )
        cond_inputs = set(contract.get("inputs", {}).get("condition_encoder", []))
        if not cond_inputs:
            required = ["encoder_hidden_states", "encoder_attention_mask", "context_latents"]
            missing = [k for k in required if k not in arr]
            if missing:
                raise ValueError(f"inputs npz missing keys: {missing}")
            return (
                arr["encoder_hidden_states"].astype(np.float32),
                arr["encoder_attention_mask"].astype(np.float32),
                arr["context_latents"].astype(np.float32),
            )

        fallback_map = {
            "hidden_states": src_latents,
            "attention_mask": attention_mask,
            "silence_latent": arr["silence_latent"].astype(np.float32)
            if "silence_latent" in arr
            else src_latents,
        }
        feeds: Dict[str, np.ndarray] = {}
        for name in cond_inputs:
            if name in arr:
                value = arr[name]
            elif name in fallback_map:
                value = fallback_map[name]
            else:
                raise ValueError(f"inputs npz missing condition_encoder input: {name}")

            if name == "refer_audio_order_mask":
                feeds[name] = value.astype(np.int64, copy=False)
            elif name == "is_covers":
                feeds[name] = value.astype(bool, copy=False)
            else:
                feeds[name] = value.astype(np.float32, copy=False)

        out = self.sessions.run(
            "condition_encoder.onnx",
            feeds,
            output_names=["encoder_hidden_states", "encoder_attention_mask", "context_latents"],
        )
        encoder_hidden_states = out[0].astype(np.float32)
        encoder_attention_mask = out[1].astype(np.float32)
        context_latents = out[2].astype(np.float32)
        return encoder_hidden_states, encoder_attention_mask, context_latents

    def _maybe_build_qwen_embeddings(
        self,
        case: CaseSpec,
        arr: Dict[str, np.ndarray],
        enable: bool,
        text: Optional[str],
        lyrics: Optional[str],
        text_max_tokens: int,
        lyric_max_tokens: int,
    ) -> Dict[str, np.ndarray]:
        if not enable:
            return {}

        text_value = (text if text is not None else case.caption).strip()
        lyric_value = (lyrics if lyrics is not None else case.lyrics).strip()
        meta = case.metadata or {}
        instruction = _format_instruction(DEFAULT_DIT_INSTRUCTION)
        meta_text = _meta_to_text(meta)
        language = str(meta.get("language") or meta.get("vocal_language") or "unknown")
        text_prompt = SFT_GEN_PROMPT.format(instruction, text_value, meta_text)
        lyrics_prompt = _format_lyrics(lyric_value, language)
        overrides: Dict[str, np.ndarray] = {}

        if not text_value and "text_hidden_states" in arr and "text_attention_mask" in arr:
            pass
        else:
            if self.qwen_embed is None:
                self.qwen_embed = EmbeddingPipeline(
                    self.onnx_dir,
                    provider=self.sessions.config.provider,
                    tokenizer_path=self.qwen_tokenizer_path,
                )
            text_hidden, text_mask = self.qwen_embed.encode_text(text_prompt, max_tokens=text_max_tokens)
            overrides["text_hidden_states"] = text_hidden
            overrides["text_attention_mask"] = text_mask

        if not lyric_value and "lyric_hidden_states" in arr and "lyric_attention_mask" in arr:
            pass
        else:
            if self.qwen_embed is None:
                self.qwen_embed = EmbeddingPipeline(
                    self.onnx_dir,
                    provider=self.sessions.config.provider,
                    tokenizer_path=self.qwen_tokenizer_path,
                )
            lyric_hidden, lyric_mask = self.qwen_embed.embed_text(lyrics_prompt, max_tokens=lyric_max_tokens)
            overrides["lyric_hidden_states"] = lyric_hidden
            overrides["lyric_attention_mask"] = lyric_mask

        return overrides

    def generate_from_precomputed(
        self,
        case: CaseSpec,
        inputs_npz: Path,
        out_npz: Path,
        *,
        use_online_qwen_embed: bool = False,
        prefer_precomputed_condition: bool = True,
        text: str | None = None,
        lyrics: str | None = None,
        text_max_tokens: int = 256,
        lyric_max_tokens: int = 2048,
    ) -> Path:
        contract = self._load_contract()
        npz = np.load(str(inputs_npz), allow_pickle=False)
        arr: Dict[str, np.ndarray] = {k: npz[k] for k in npz.files}

        overrides = self._maybe_build_qwen_embeddings(
            case=case,
            arr=arr,
            enable=use_online_qwen_embed,
            text=text,
            lyrics=lyrics,
            text_max_tokens=text_max_tokens,
            lyric_max_tokens=lyric_max_tokens,
        )
        if overrides:
            arr.update(overrides)

        if "src_latents" not in arr:
            raise ValueError("inputs npz missing key: src_latents")
        src_latents = arr["src_latents"].astype(np.float32)
        if "xt_steps" in arr:
            xt = arr["xt_steps"][0].astype(np.float32).copy()
        else:
            xt = src_latents.copy()
        if "latent_masks" in arr:
            attention_mask = arr["latent_masks"].astype(np.float32)
        else:
            attention_mask = np.ones((xt.shape[0], xt.shape[1]), dtype=np.float32)

        encoder_hidden_states, encoder_attention_mask, context_latents = self._run_condition_encoder(
            arr,
            contract,
            src_latents=src_latents,
            attention_mask=attention_mask,
            prefer_precomputed_condition=prefer_precomputed_condition,
        )
        context_latents_orig = context_latents.copy()

        patch_size = 2
        orig_len = int(xt.shape[1])
        pad_len = (-orig_len) % patch_size
        if pad_len:
            context_latents_padded = np.pad(context_latents, ((0, 0), (0, pad_len), (0, 0)), mode="constant")
            attention_mask_padded = np.pad(attention_mask, ((0, 0), (0, pad_len)), mode="constant")
        else:
            context_latents_padded = context_latents
            attention_mask_padded = attention_mask
        inputs_contract = contract.get("inputs", {})
        outputs_contract = contract.get("outputs", {})
        dit_inputs = set(inputs_contract.get("dit_decoder", []))
        dit_prefill_inputs = list(inputs_contract.get("dit_prefill_kv", []))
        dit_decode_inputs = list(inputs_contract.get("dit_decode_kv", []))
        dit_prefill_outputs = list(outputs_contract.get("dit_prefill_kv", []))
        dit_decode_outputs = list(outputs_contract.get("dit_decode_kv", []))
        use_dit_kv = bool(
            dit_prefill_inputs
            and dit_decode_inputs
            and dit_prefill_outputs
            and dit_decode_outputs
            and (self.onnx_dir / "dit_prefill_kv.onnx").exists()
            and (self.onnx_dir / "dit_decode_kv.onnx").exists()
        )
        if os.environ.get("ACESTEP_DISABLE_DIT_KV", "").strip().lower() in {"1", "true", "yes", "on"}:
            use_dit_kv = False
        if not use_dit_kv:
            print("[runtime_py_ort] dit kv models unavailable; fallback to dit_decoder.onnx")

        timesteps = resolve_timesteps(case.shift, None, max_steps=max(1, int(case.inference_steps)))
        xt_steps = []
        vt_steps = []
        cache_map: Dict[str, np.ndarray] = {}

        for i, t in enumerate(timesteps):
            t_vec = np.full((xt.shape[0],), t, dtype=np.float32)
            if pad_len:
                xt_in = np.pad(xt, ((0, 0), (0, pad_len), (0, 0)), mode="constant")
            else:
                xt_in = xt
            base_feeds = {
                "hidden_states": xt_in,
                "timestep": t_vec,
                "timestep_r": t_vec,
                "encoder_hidden_states": encoder_hidden_states,
                "context_latents": context_latents_padded,
            }
            if use_dit_kv:
                if "attention_mask" in set(dit_prefill_inputs + dit_decode_inputs):
                    base_feeds["attention_mask"] = attention_mask_padded
                if "encoder_attention_mask" in set(dit_prefill_inputs + dit_decode_inputs):
                    base_feeds["encoder_attention_mask"] = encoder_attention_mask
                if i == 0:
                    feeds = {name: base_feeds[name] for name in dit_prefill_inputs if name in base_feeds}
                    out_names = dit_prefill_outputs
                    out = self.sessions.run("dit_prefill_kv.onnx", feeds, output_names=out_names)
                else:
                    feeds: Dict[str, np.ndarray] = {}
                    for name in dit_decode_inputs:
                        if name.startswith("past_"):
                            present_name = "present_" + name[len("past_") :]
                            if present_name not in cache_map:
                                raise ValueError(f"missing cache tensor for {present_name}")
                            feeds[name] = cache_map[present_name]
                        elif name in base_feeds:
                            feeds[name] = base_feeds[name]
                    out_names = dit_decode_outputs
                    out = self.sessions.run("dit_decode_kv.onnx", feeds, output_names=out_names)
                out_map = {name: value for name, value in zip(out_names, out)}
                vt = out_map["vt"].astype(np.float32)
                cache_map = {
                    name: value.astype(np.float32)
                    for name, value in out_map.items()
                    if name.startswith("present_")
                }
            else:
                feeds = dict(base_feeds)
                if "attention_mask" in dit_inputs:
                    feeds["attention_mask"] = attention_mask_padded
                if "encoder_attention_mask" in dit_inputs:
                    feeds["encoder_attention_mask"] = encoder_attention_mask
                vt = self.sessions.run("dit_decoder.onnx", feeds, output_names=["vt"])[0].astype(np.float32)
            if pad_len:
                vt = vt[:, :orig_len, :]
            xt_steps.append(xt.copy())
            vt_steps.append(vt.copy())

            if i == len(timesteps) - 1:
                xt = x0_from_noise(xt, vt, float(t))
                break
            xt = ode_step(xt, vt, float(t), float(timesteps[i + 1]))

        out_npz.parent.mkdir(parents=True, exist_ok=True)
        pred_latents = xt
        np.savez_compressed(
            str(out_npz),
            pred_latents=pred_latents,
            xt_steps=np.asarray(xt_steps, dtype=np.float32),
            vt_steps=np.asarray(vt_steps, dtype=np.float32),
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            context_latents=context_latents_orig,
        )
        return out_npz
