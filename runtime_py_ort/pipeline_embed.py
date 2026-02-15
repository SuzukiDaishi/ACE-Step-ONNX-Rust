from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
from tokenizers import Tokenizer

from .session_manager import OrtSessionManager


class EmbeddingPipeline:
    def __init__(self, onnx_dir: Path, provider: str = "cpu", tokenizer_path: Path | None = None):
        self.onnx_dir = Path(onnx_dir)
        self.sessions = OrtSessionManager(self.onnx_dir)
        self.sessions.config.provider = provider
        self.model_name = "qwen3_embedding_0p6.onnx"
        self.token_model_name = "qwen3_token_embedding_0p6.onnx"
        self.tokenizer_path = Path(tokenizer_path) if tokenizer_path else (
            Path("checkpoints") / "Qwen3-Embedding-0.6B" / "tokenizer.json"
        )
        self._tokenizer: Tokenizer | None = None

        contract_path = self.onnx_dir / "io_contract_qwen3_embedding_0p6.json"
        if contract_path.exists():
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            self.model_name = str(contract.get("path", self.model_name))
            self.token_model_name = str(contract.get("token_embedding_path", self.token_model_name))

    def encode(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        outputs = self.sessions.run(
            self.model_name,
            {
                "input_ids": input_ids.astype(np.int64, copy=False),
                "attention_mask": attention_mask.astype(np.int64, copy=False),
            },
            output_names=["last_hidden_state"],
        )
        return outputs[0].astype(np.float32, copy=False)

    def token_embed(self, input_ids: np.ndarray) -> np.ndarray:
        try:
            outputs = self.sessions.run(
                self.token_model_name,
                {"input_ids": input_ids.astype(np.int64, copy=False)},
                output_names=["token_embeddings"],
            )
            return outputs[0].astype(np.float32, copy=False)
        except FileNotFoundError:
            attention_mask = np.ones_like(input_ids, dtype=np.int64)
            return self.encode(input_ids=input_ids, attention_mask=attention_mask)

    def encode_text(self, text: str, max_tokens: int) -> Tuple[np.ndarray, np.ndarray]:
        if self._tokenizer is None:
            if not self.tokenizer_path.exists():
                raise FileNotFoundError(f"Tokenizer JSON not found: {self.tokenizer_path}")
            self._tokenizer = Tokenizer.from_file(str(self.tokenizer_path))

        normalized = text if text.strip() else " "
        ids = self._tokenizer.encode(normalized).ids
        if not ids:
            ids = [0]
        if max_tokens > 0:
            ids = ids[:max_tokens]
        input_ids = np.asarray([ids], dtype=np.int64)
        attention_mask = np.ones_like(input_ids, dtype=np.int64)
        hidden = self.encode(input_ids=input_ids, attention_mask=attention_mask)
        return hidden.astype(np.float32, copy=False), attention_mask.astype(np.float32, copy=False)

    def embed_text(self, text: str, max_tokens: int) -> Tuple[np.ndarray, np.ndarray]:
        if self._tokenizer is None:
            if not self.tokenizer_path.exists():
                raise FileNotFoundError(f"Tokenizer JSON not found: {self.tokenizer_path}")
            self._tokenizer = Tokenizer.from_file(str(self.tokenizer_path))

        normalized = text if text.strip() else " "
        ids = self._tokenizer.encode(normalized).ids
        if not ids:
            ids = [0]
        if max_tokens > 0:
            ids = ids[:max_tokens]
        input_ids = np.asarray([ids], dtype=np.int64)
        attention_mask = np.ones_like(input_ids, dtype=np.int64)
        hidden = self.token_embed(input_ids=input_ids)
        return hidden.astype(np.float32, copy=False), attention_mask.astype(np.float32, copy=False)
