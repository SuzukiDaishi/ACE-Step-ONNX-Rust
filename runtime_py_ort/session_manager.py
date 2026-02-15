from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass
class SessionConfig:
    provider: str = "cpu"
    intra_op_num_threads: int = 0


class OrtSessionManager:
    def __init__(self, onnx_dir: Path, config: Optional[SessionConfig] = None):
        self.onnx_dir = Path(onnx_dir)
        self.config = config or SessionConfig()
        self._ort = None
        self._sessions: Dict[str, object] = {}

    def _load_ort(self):
        if self._ort is None:
            import onnxruntime as ort
            if self.config.provider.lower() == "cuda":
                # Load CUDA/cuDNN DLLs from pip-installed NVIDIA packages on Windows.
                try:
                    ort.preload_dlls()
                except Exception:
                    pass
            self._ort = ort
        return self._ort

    def _providers(self) -> List[str]:
        if self.config.provider.lower() == "cuda":
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def get(self, model_name: str):
        if model_name in self._sessions:
            return self._sessions[model_name]

        ort = self._load_ort()
        opts = ort.SessionOptions()
        if self.config.intra_op_num_threads > 0:
            opts.intra_op_num_threads = self.config.intra_op_num_threads
        if os.environ.get("ACESTEP_ORT_DISABLE_OPT", "").strip().lower() in {"1", "true", "yes", "on"}:
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

        model_path = self.onnx_dir / model_name
        if not model_path.exists():
            raise FileNotFoundError(f"Missing ONNX file: {model_path}")

        providers = self._providers()
        try:
            sess = ort.InferenceSession(str(model_path), sess_options=opts, providers=providers)
        except Exception:
            if self.config.provider.lower() != "cuda":
                raise
            sess = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self._sessions[model_name] = sess
        return sess

    def run(self, model_name: str, feeds: Dict[str, object], output_names: Optional[Iterable[str]] = None):
        sess = self.get(model_name)
        input_names = {x.name for x in sess.get_inputs()}
        filtered_feeds = {k: v for k, v in feeds.items() if k in input_names}
        missing = [name for name in input_names if name not in filtered_feeds]
        if missing:
            raise ValueError(f"Missing required inputs for {model_name}: {missing}")
        return sess.run(list(output_names) if output_names else None, filtered_feeds)
