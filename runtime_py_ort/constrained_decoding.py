from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np


@dataclass
class SamplingConfig:
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0


def apply_banned_token_mask(logits: np.ndarray, banned_token_ids: Optional[Iterable[int]] = None) -> np.ndarray:
    out = logits.copy()
    if banned_token_ids is None:
        return out
    for idx in banned_token_ids:
        idx = int(idx)
        if 0 <= idx < out.shape[-1]:
            out[..., idx] = -1e30
    return out


def sample_next_token(logits: np.ndarray, cfg: SamplingConfig, rng: np.random.Generator) -> int:
    x = logits.astype(np.float64)
    t = max(cfg.temperature, 1e-6)
    x = x / t

    if cfg.top_k and cfg.top_k > 0:
        k = int(min(cfg.top_k, x.shape[-1]))
        top_idx = np.argpartition(x, -k)[-k:]
        mask = np.full_like(x, -1e30)
        mask[top_idx] = x[top_idx]
        x = mask

    x = x - np.max(x)
    probs = np.exp(x)
    probs = probs / np.sum(probs)

    if cfg.top_p < 1.0:
        order = np.argsort(probs)[::-1]
        csum = np.cumsum(probs[order])
        keep = order[csum <= cfg.top_p]
        if keep.size == 0:
            keep = order[:1]
        filtered = np.zeros_like(probs)
        filtered[keep] = probs[keep]
        probs = filtered / np.sum(filtered)

    return int(rng.choice(np.arange(probs.size), p=probs))
