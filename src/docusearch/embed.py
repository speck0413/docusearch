"""Embeddings & model provenance (§8, R-EMB-*).

Local embedding via sentence-transformers is the default; no API key is needed for the
core workflow (R-EMB-1). Every vector this module produces is tagged with the model id
that made it (R-EMB-2), and vectors are L2-normalized so cosine similarity is a dot
product. ``embed.model: none`` means no provider at all — BM25-only (R-CFG-4).

API-based providers are deferred (R-EMB-7): ``ApiProvider`` is a stub. The
client/server ``RemoteServerProvider`` + ``auto`` negotiation arrive with Phase 3.

Public surface:
    EmbedProvider (Protocol)      -- model_id, dim, embed(texts) -> (n, dim) float32
    LocalProvider                 -- sentence-transformers, lazy-loaded
    ApiProvider                   -- STUB ONLY (R-EMB-7)
    make_provider(embed_config)   -> EmbedProvider | None   (None == BM25-only)
    to_blob(vec) / from_blob(bytes)  -- float32 vector <-> DB BLOB (provenance store)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from .config import EmbedConfig


class EmbedError(Exception):
    """Embedding provenance problem (e.g. re-indexing with a different model)."""


@runtime_checkable
class EmbedProvider(Protocol):
    """Anything that turns texts into L2-normalized float32 vectors, tagged by model."""

    @property
    def model_id(self) -> str: ...

    @property
    def dim(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


class LocalProvider:
    """Embeds with a local sentence-transformers model. The model loads lazily on first
    use (keeps idle RSS low, R-PERF-4) and stays cached for the process."""

    def __init__(
        self,
        model_id: str,
        *,
        device: str = "cpu",
        batch_size: int = 128,
        trust_remote_code: bool = False,
    ) -> None:
        self._model_id = model_id
        self._device = device
        self._batch_size = batch_size
        self._trust_remote_code = trust_remote_code
        self._model: Any = None
        self._dim: int | None = None

    def _ensure_loaded(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self._model_id,
                device=self._device,
                trust_remote_code=self._trust_remote_code,
            )
            # method was renamed across sentence-transformers versions; prefer the new one
            if hasattr(self._model, "get_embedding_dimension"):
                self._dim = int(self._model.get_embedding_dimension())
            else:
                self._dim = int(self._model.get_sentence_embedding_dimension())

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dim(self) -> int:
        self._ensure_loaded()
        assert self._dim is not None
        return self._dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an ``(n, dim)`` float32 array of L2-normalized vectors."""
        self._ensure_loaded()
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)


class ApiProvider:  # STUB ONLY — R-EMB-7 (verify provider availability before wiring)
    """Placeholder for API-based embedding providers (OpenAI/Copilot/Anthropic-partner).

    Deferred by R-EMB-7: the interface exists so callers can be written against it, but
    no implementation ships until a provider + key story is verified.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError("API embedding providers are deferred (R-EMB-7).")


def make_provider(embed_config: EmbedConfig) -> EmbedProvider | None:
    """Build the configured provider, or None for ``embed.model: none`` (BM25-only)."""
    model = embed_config.model
    if model == "none":
        return None
    if model == "auto":
        # 'auto' negotiates a model with a server (§8) — a client/server feature (Phase 3).
        raise NotImplementedError("embed.model 'auto' negotiation lands in Phase 3.")
    device = embed_config.device
    if device == "auto":
        device = "cpu"  # deterministic default + low RSS; set device: cuda to opt into GPU
    return LocalProvider(
        model,
        device=device,
        batch_size=embed_config.batch_size,
        trust_remote_code=embed_config.trust_remote_code,
    )


def to_blob(vec: np.ndarray) -> bytes:
    """Serialize a vector to a float32 BLOB for the embeddings table."""
    return np.asarray(vec, dtype=np.float32).tobytes()


def from_blob(blob: bytes) -> np.ndarray:
    """Deserialize a float32 BLOB back into a 1-D vector."""
    return np.frombuffer(blob, dtype=np.float32)
