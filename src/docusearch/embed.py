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
    RemoteServerProvider          -- POST text to a server's /v1/embed (client, R-EMB-4)
    choose_auto_strategy(...)     -- `auto` negotiation: embed local vs send text (R-EMB-5)
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


class RemoteServerProvider:
    """Embeds by POSTing text to a docusearch server's ``/v1/embed`` (R-EMB-4).

    The client 'auto' fallback: when a client can't (or shouldn't) load the model
    locally, it asks the server to embed. model_id/dim are learned from ``/v1/embed-info``.
    """

    def __init__(self, server_url: str, *, http_client: Any = None, timeout: float = 30.0) -> None:
        self._url = server_url.rstrip("/")
        self._client = http_client
        self._timeout = timeout
        self._model_id: str | None = None
        self._dim: int | None = None

    def _http(self) -> Any:
        if self._client is None:
            import httpx

            self._client = httpx.Client(base_url=self._url, timeout=self._timeout)
        return self._client

    def _ensure_info(self) -> None:
        if self._model_id is None:
            info = self._http().get("/v1/embed-info").json()
            self._model_id = str(info["model"])
            self._dim = int(info["dim"])

    @property
    def model_id(self) -> str:
        self._ensure_info()
        assert self._model_id is not None
        return self._model_id

    @property
    def dim(self) -> int:
        self._ensure_info()
        assert self._dim is not None
        return self._dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        resp = self._http().post("/v1/embed", json={"texts": list(texts)})
        if resp.status_code == 409:
            raise EmbedError("server has no embedding model (embed.model: none) to embed with")
        resp.raise_for_status()
        return np.asarray(resp.json()["vectors"], dtype=np.float32)


def choose_auto_strategy(server_info: dict[str, Any], auto_max_mb: int) -> str:
    """`auto` negotiation (R-EMB-5): 'local' if the server's model is small enough to run
    locally (<= auto_max_mb), else 'text' (send text and let the server embed)."""
    if str(server_info.get("model", "none")) == "none":
        return "text"
    return "local" if int(server_info.get("approx_mb", 1 << 30)) <= auto_max_mb else "text"


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
