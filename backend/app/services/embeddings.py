"""FastEmbed ONNX embedding service — singleton, warm at startup."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import numpy as np
from fastembed import TextEmbedding

if TYPE_CHECKING:
    from app.config import Settings


class EmbeddingService:
    _instance: EmbeddingService | None = None
    _lock = threading.Lock()

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name, threads=1)
        self._dim: int | None = None
        self._cache: dict[str, np.ndarray] = {}

    @classmethod
    def get_instance(cls, settings: Settings) -> EmbeddingService:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(settings.embedding_model)
        return cls._instance

    @property
    def dimension(self) -> int:
        if self._dim is None:
            vec = next(self._model.embed(["warmup"]))
            self._dim = len(vec)
        return self._dim

    def embed_query(self, text: str) -> np.ndarray:
        clean = text.strip()
        if clean in self._cache:
            return self._cache[clean].copy()
        prefix = "query: " if "bge" in self.model_name.lower() else ""
        vec = next(self._model.embed([f"{prefix}{clean}"]))
        arr = np.array(vec, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr /= norm
        if len(self._cache) > 2000:
            self._cache.clear()
        self._cache[clean] = arr
        return arr

    def embed_passages(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        prefixed = []
        for t in texts:
            prefix = "passage: " if "bge" in self.model_name.lower() else ""
            prefixed.append(f"{prefix}{t.strip()}")
        vectors = list(self._model.embed(prefixed, batch_size=batch_size))
        arr = np.array(vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

    def warmup(self, n: int = 5) -> None:
        for i in range(n):
            self.embed_query(f"warmup query number {i}")
