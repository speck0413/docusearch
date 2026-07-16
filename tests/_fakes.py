"""Shared test doubles."""

from __future__ import annotations

import hashlib

import numpy as np


class FakeProvider:
    """Deterministic hash-based embedding provider (no torch): same text -> same vector."""

    def __init__(self, model_id: str = "fake-v1", dim: int = 8) -> None:
        self._id = model_id
        self._dim = dim

    @property
    def model_id(self) -> str:
        return self._id

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vec = np.frombuffer(digest[: self._dim], dtype=np.uint8).astype(np.float32)
            out[i] = vec / (np.linalg.norm(vec) or 1.0)
        return out
