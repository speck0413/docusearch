"""Catalog — the small fluent facade that ties config + store + ingest + search together.

This is the public entry point (R-ARCH-2). It stays deliberately thin: it opens the
store, delegates to the pipeline and search modules, and hands back plain result
objects. Heavy logic lives in ``ingest.py`` and ``search.py``.

    from docusearch import Catalog
    cat = Catalog.from_config("docusearch.yaml")   # creates the file if missing (R-CFG-2)
    result = cat.ingest()                          # -> IngestResult
    hits = cat.search("SPI timing configuration")  # -> list[SearchHit]
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import overload

from . import embed, ingest, search
from .config import DEFAULT_CONFIG_PATH, Config, load
from .embed import EmbedProvider
from .ingest import IngestResult
from .search import SearchHit
from .store import Store


class Catalog:
    """A configured document catalog: ingest sources and search the index."""

    def __init__(self, config: Config) -> None:
        self.config = config

    @classmethod
    def from_config(cls, path: Path | str = DEFAULT_CONFIG_PATH) -> Catalog:
        """Load (creating a template if missing) the config at ``path`` (R-CFG-2)."""
        return cls(load(Path(path)))

    @property
    def db_path(self) -> str:
        return self.config.paths.db_path

    def ingest(self, *, force: bool = False) -> IngestResult:
        """Ingest every configured source into the index (§7). Returns audit counts."""
        with Store.open(self.db_path) as store:
            return ingest.run_ingest(self.config, store, force=force)

    def _provider(self) -> EmbedProvider | None:
        """The embedding provider for queries, or None (BM25-only) when not applicable."""
        if self.config.embed.model == "none" or self.config.search.bm25_only:
            return None
        try:
            return embed.make_provider(self.config.embed)
        except NotImplementedError:  # e.g. embed.model: auto (Phase 3) -> fall back to BM25
            return None

    @overload
    def search(
        self, query: str, *, top_k: int | None = ..., prefix: bool = ...
    ) -> list[SearchHit]: ...

    @overload
    def search(
        self, query: list[str], *, top_k: int | None = ..., prefix: bool = ...
    ) -> list[list[SearchHit]]: ...

    def search(
        self,
        query: str | list[str],
        *,
        top_k: int | None = None,
        prefix: bool = False,
    ) -> list[SearchHit] | list[list[SearchHit]]:
        """Search the index — hybrid when embeddings exist, else BM25 (R-SRCH-1/2/3/4).

        Accepts one query or a batch. Roles come from ``DOCUSEARCH_ROLES`` (R-SRCH-4).
        ``top_k`` defaults to the configured value.
        """
        k = top_k if top_k is not None else self.config.search.top_k_default
        roles = search.roles_from_env()
        provider = self._provider()
        with Store.open(self.db_path) as store:
            vector_index = None
            if provider is not None and store.count_embeddings() > 0:
                index_model = store.get_meta("embed_model")
                if index_model is not None and index_model != provider.model_id:
                    # Never compare vectors from two different models (R-EMB-3). Fall back
                    # to BM25 (loudly) rather than silently mixing embedding spaces.
                    warnings.warn(
                        f"index was embedded with {index_model!r} but embed.model is "
                        f"{provider.model_id!r}; using BM25 only. Re-ingest to enable hybrid.",
                        stacklevel=2,
                    )
                    provider = None
                else:
                    ann_path = (
                        Path(self.db_path).with_suffix(".hnsw")
                        if self.db_path != ":memory:"
                        else "__no_ann__"
                    )
                    vector_index = search.VectorIndex.load(store, provider.dim, ann_path)
            return search.search(
                store,
                query,
                top_k=k,
                provider=provider,
                vector_index=vector_index,
                rrf_k=self.config.search.rrf_k,
                prefix=prefix,
                roles=roles,
                bm25_only=self.config.search.bm25_only,
            )

    def audit(self) -> str:
        """Render the current index audit (counts + anomalies)."""
        with Store.open(self.db_path) as store:
            return ingest.render_store_audit(store)
