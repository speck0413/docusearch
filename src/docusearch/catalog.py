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

from pathlib import Path

from . import ingest, search
from .config import DEFAULT_CONFIG_PATH, Config, load
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

    def search(
        self, query: str, *, top_k: int | None = None, prefix: bool = False
    ) -> list[SearchHit]:
        """BM25 search the index (R-SRCH-1). ``top_k`` defaults to the configured value."""
        k = top_k if top_k is not None else self.config.search.top_k_default
        with Store.open(self.db_path) as store:
            return search.bm25_search(store, query, top_k=k, prefix=prefix)

    def audit(self) -> str:
        """Render the current index audit (counts + anomalies)."""
        with Store.open(self.db_path) as store:
            return ingest.render_store_audit(store)
