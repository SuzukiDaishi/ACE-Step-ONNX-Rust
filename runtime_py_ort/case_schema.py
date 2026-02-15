from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass
class CaseSpec:
    case_id: str
    mode: str
    seed: int = 42
    inference_steps: int = 8
    shift: float = 3.0
    infer_method: str = "ode"
    thinking: bool = False
    audio_format: str = "wav"
    caption: str = ""
    lyrics: str = ""
    simple_mode_query: str = ""
    lm_model_variant: str = "1.7B"
    deterministic: bool = True
    lm_sampling: Dict[str, Any] = field(default_factory=lambda: {"temperature": 0.0, "top_k": 0, "top_p": 1.0})
    metadata: Dict[str, Any] = field(default_factory=dict)
    expected: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_path(path: Path) -> "CaseSpec":
        data = json.loads(path.read_text(encoding="utf-8"))
        return CaseSpec(
            case_id=str(data.get("case_id", path.stem)),
            mode=str(data.get("mode", "text2music")),
            seed=int(data.get("seed", 42)),
            inference_steps=int(data.get("inference_steps", 8)),
            shift=float(data.get("shift", 3.0)),
            infer_method=str(data.get("infer_method", "ode")),
            thinking=bool(data.get("thinking", False)),
            audio_format=str(data.get("audio_format", "wav")),
            caption=str(data.get("caption", "")),
            lyrics=str(data.get("lyrics", "")),
            simple_mode_query=str(data.get("simple_mode_query", "")),
            lm_model_variant=str(data.get("lm_model_variant", "1.7B")),
            deterministic=bool(data.get("deterministic", True)),
            lm_sampling=data.get("lm_sampling", {"temperature": 0.0, "top_k": 0, "top_p": 1.0}) or {"temperature": 0.0, "top_k": 0, "top_p": 1.0},
            metadata=data.get("metadata", {}) or {},
            expected=data.get("expected", {}) or {},
        )
