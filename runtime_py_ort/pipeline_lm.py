from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from acestep.constants import DEFAULT_LM_INSPIRED_INSTRUCTION
from .constrained_decoding import SamplingConfig, sample_next_token
from .session_manager import OrtSessionManager


@dataclass
class LmGenerateResult:
    token_ids: List[int]
    text: str = ""


@dataclass
class SimpleModeSample:
    token_ids: List[int]
    raw_text: str
    metadata: Dict[str, Any]
    caption: str
    lyrics: str
    instrumental: bool


class LMPipeline:
    def __init__(self, onnx_dir: Path, provider: str = "cpu", lm_model_variant: str = "1.7B"):
        self.onnx_dir = Path(onnx_dir)
        self.sessions = OrtSessionManager(self.onnx_dir)
        self.sessions.config.provider = provider
        variant_tag = "0p6" if str(lm_model_variant) == "0.6B" else "1p7"
        self.prefill_model = f"lm_{variant_tag}_prefill.onnx"
        self.decoder_model = f"lm_{variant_tag}_decode.onnx"
        self.lm_model_variant = str(lm_model_variant)
        self._tokenizer = None

        contract_candidates = [
            self.onnx_dir / f"io_contract_lm_{variant_tag}.json",
            self.onnx_dir / "io_contract_lm.json",
        ]
        for contract_path in contract_candidates:
            if not contract_path.exists():
                continue
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            self.prefill_model = str(contract.get("prefill_path", self.prefill_model))
            self.decoder_model = str(contract.get("decode_path", contract.get("decoder_path", self.decoder_model)))
            break

    @staticmethod
    def _extract_lyrics_from_output(output_text: str) -> str:
        import re

        match = re.search(r"</think>", output_text)
        if not match:
            return ""
        after_think = output_text[match.end() :].strip()
        if not after_think:
            return ""
        after_think = re.sub(r"^#\s*Lyri[c|cs]?\s*\n", "", after_think, flags=re.IGNORECASE)
        after_think = re.sub(r"<\|im_end\|>\s*$", "", after_think)
        return after_think.strip()

    @staticmethod
    def _parse_lm_output(output_text: str) -> tuple[Dict[str, Any], str]:
        import re

        metadata: Dict[str, Any] = {}
        audio_codes = ""

        code_matches = re.findall(r"<\|audio_code_\d+\|>", output_text)
        if code_matches:
            audio_codes = "".join(code_matches)

        reasoning_text = None
        for pattern in [r"<think>(.*?)</think>", r"<reasoning>(.*?)</reasoning>"]:
            match = re.search(pattern, output_text, re.DOTALL)
            if match:
                reasoning_text = match.group(1).strip()
                break
        if not reasoning_text:
            reasoning_text = output_text.split("<|audio_code_")[0].strip() if "<|audio_code_" in output_text else output_text.strip()

        if reasoning_text:
            from acestep.constrained_logits_processor import MetadataConstrainedLogitsProcessor

            lines = reasoning_text.split("\n")
            current_key = None
            current_value_lines: List[str] = []

            def save_current_field() -> None:
                nonlocal current_key, current_value_lines
                if current_key and current_value_lines:
                    value = "\n".join(current_value_lines)
                    if current_key == "bpm":
                        try:
                            metadata["bpm"] = int(value.strip())
                        except Exception:
                            metadata["bpm"] = value.strip()
                    elif current_key == "caption":
                        metadata["caption"] = MetadataConstrainedLogitsProcessor.postprocess_caption(value)
                    elif current_key == "duration":
                        try:
                            metadata["duration"] = int(value.strip())
                        except Exception:
                            metadata["duration"] = value.strip()
                    elif current_key in {"genres", "keyscale", "language", "timesignature"}:
                        metadata[current_key] = value.strip()
                current_key = None
                current_value_lines = []

            for line in lines:
                if line.strip().startswith("<"):
                    continue
                if line and not line[0].isspace() and ":" in line:
                    save_current_field()
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        current_key = parts[0].strip().lower()
                        first_value = parts[1]
                        if first_value.strip():
                            current_value_lines.append(first_value)
                elif (line.startswith(" ") or line.startswith("\t")) and current_key:
                    current_value_lines.append(line)
            save_current_field()

        return metadata, audio_codes

    def _load_tokenizer(self, tokenizer_path: Optional[Path] = None):
        if self._tokenizer is not None:
            return self._tokenizer

        from transformers import AutoTokenizer

        if tokenizer_path is not None:
            target = str(tokenizer_path)
        else:
            model_dir = "acestep-5Hz-lm-0.6B" if self.lm_model_variant == "0.6B" else "acestep-5Hz-lm-1.7B"
            target = str(Path("checkpoints") / model_dir)
        self._tokenizer = AutoTokenizer.from_pretrained(target, use_fast=True)
        return self._tokenizer

    @staticmethod
    def _build_inspiration_prompt(tokenizer, query: str, instrumental: bool) -> str:
        instrumental_str = "true" if instrumental else "false"
        user_content = f"{query}\n\ninstrumental: {instrumental_str}"
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": f"# Instruction\n{DEFAULT_LM_INSPIRED_INSTRUCTION}\n\n"},
                {"role": "user", "content": user_content},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    @staticmethod
    def _build_constrained_processor(
        tokenizer,
        vocal_language: str,
        enabled: bool,
    ):
        if not enabled:
            return None
        from acestep.constrained_logits_processor import MetadataConstrainedLogitsProcessor

        processor = MetadataConstrainedLogitsProcessor(
            tokenizer=tokenizer,
            enabled=True,
            debug=False,
        )
        processor.reset()
        processor.set_generation_phase("understand")
        processor.set_stop_at_reasoning(False)
        processor.set_skip_genres(False)
        processor.set_skip_caption(False)
        processor.set_skip_language(False)

        lang = (vocal_language or "").strip().lower()
        if lang and lang != "unknown":
            processor.set_user_metadata({"language": vocal_language.strip()})
        else:
            processor.set_user_metadata(None)
        return processor

    def generate_tokens(
        self,
        prompt_ids: List[int],
        max_new_tokens: int = 128,
        sampling: SamplingConfig = SamplingConfig(),
        seed: int = 42,
        deterministic: bool = True,
        constrained_processor=None,
        eos_token_id: int | None = None,
        pad_token_id: int | None = None,
    ) -> LmGenerateResult:
        rng = np.random.default_rng(seed)

        input_ids = np.asarray([prompt_ids], dtype=np.int64)
        attn = np.ones_like(input_ids, dtype=np.int64)
        out = self.sessions.run(self.prefill_model, {"input_ids": input_ids, "attention_mask": attn})
        logits = out[0]
        cache = out[1:]

        generated: List[int] = []
        full_ids: List[int] = list(prompt_ids)

        for _ in range(max_new_tokens):
            logits_last = logits[0, -1]
            if constrained_processor is not None:
                import torch

                ids_t = torch.tensor([full_ids], dtype=torch.long)
                scores_t = torch.from_numpy(logits_last.astype(np.float32, copy=False)[None, :])
                scores_t = constrained_processor(ids_t, scores_t)
                logits_last = scores_t.detach().cpu().numpy()[0]

            if deterministic:
                next_token = int(np.argmax(logits_last))
            else:
                next_token = sample_next_token(logits_last, sampling, rng)
            generated.append(next_token)
            full_ids.append(next_token)
            if constrained_processor is not None:
                constrained_processor.update_state(next_token)

            if (eos_token_id is not None and next_token == int(eos_token_id)) or (
                pad_token_id is not None and next_token == int(pad_token_id)
            ):
                break

            decode_ids = np.asarray([[next_token]], dtype=np.int64)
            attn = np.ones((1, len(full_ids)), dtype=np.int64)
            feeds: Dict[str, np.ndarray] = {"input_ids": decode_ids, "attention_mask": attn}
            for i in range(0, len(cache), 2):
                layer = i // 2
                feeds[f"past_key_{layer}"] = cache[i]
                feeds[f"past_value_{layer}"] = cache[i + 1]

            out = self.sessions.run(self.decoder_model, feeds)
            logits = out[0]
            cache = out[1:]

        return LmGenerateResult(token_ids=generated)

    def generate_sample_from_query(
        self,
        query: str,
        *,
        instrumental: bool,
        vocal_language: str = "unknown",
        max_new_tokens: int = 768,
        sampling: SamplingConfig = SamplingConfig(),
        seed: int = 42,
        deterministic: bool = True,
        use_constrained_decoding: bool = True,
        tokenizer_path: Optional[Path] = None,
    ) -> SimpleModeSample:
        query_text = query.strip() if query and query.strip() else "NO USER INPUT"
        tokenizer = self._load_tokenizer(tokenizer_path=tokenizer_path)
        formatted_prompt = self._build_inspiration_prompt(tokenizer, query_text, instrumental=instrumental)
        prompt_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False)

        constrained = self._build_constrained_processor(
            tokenizer=tokenizer,
            vocal_language=vocal_language,
            enabled=use_constrained_decoding,
        )

        lm_out = self.generate_tokens(
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            sampling=sampling,
            seed=seed,
            deterministic=deterministic,
            constrained_processor=constrained,
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
            pad_token_id=getattr(tokenizer, "pad_token_id", None),
        )
        raw_text = tokenizer.decode(lm_out.token_ids, skip_special_tokens=False)
        metadata, _ = self._parse_lm_output(raw_text)
        lyrics = self._extract_lyrics_from_output(raw_text)
        if lyrics:
            metadata["lyrics"] = lyrics
        elif instrumental:
            lyrics = "[Instrumental]"
            metadata["lyrics"] = lyrics

        caption = str(metadata.get("caption", "") or "")
        if not lyrics:
            lyrics = str(metadata.get("lyrics", "") or "")
        metadata["instrumental"] = bool(instrumental)
        return SimpleModeSample(
            token_ids=lm_out.token_ids,
            raw_text=raw_text,
            metadata=metadata,
            caption=caption,
            lyrics=lyrics,
            instrumental=bool(instrumental),
        )
