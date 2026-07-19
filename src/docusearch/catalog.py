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
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import overload

from . import embed, enrich, ingest, search, vision
from .config import DEFAULT_CONFIG_PATH, Config, ConfigError, load
from .embed import EmbedProvider
from .ingest import IngestResult, ProgressFn
from .search import FederatedMember, FederatedSearch, SearchHit
from .store import Store


@dataclass
class SummaryResult:
    """Counts for one AI-summary enrichment run (§17)."""

    pending: int = 0
    summarized: int = 0
    skipped: int = 0  # empty document text — nothing to summarize
    failed: int = 0
    errors: list[tuple[int, str]] = field(default_factory=list)  # (doc_id, message)


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

    def ingest(
        self,
        *,
        force: bool = False,
        reembed: bool = False,
        progress: ingest.ProgressFn | None = None,
    ) -> IngestResult:
        """Ingest every configured source into the index (§7). Returns audit counts.

        ``reembed`` drops existing vectors first (switch models / heal a mixed index);
        ``progress`` receives (phase, done, total) callbacks for a live progress bar.
        """
        with Store.open(self.db_path) as store:
            return ingest.run_ingest(
                self.config, store, force=force, reembed=reembed, progress=progress
            )

    def prune_missing(self, *, apply: bool = False) -> int:
        """Remove documents whose source file no longer exists (e.g. after the source folder
        was moved or renamed, which orphans the path-keyed originals). Returns the count;
        with ``apply=False`` it only counts (dry run)."""
        with Store.open(self.db_path) as store:
            missing = [i for i, p in store.all_document_paths() if not Path(p).exists()]
            if apply and missing:
                for doc_id in missing:
                    store.delete_document(doc_id)
                self._refresh_sidecar(store)
        return len(missing)

    def remove_source(self, name: str) -> int:
        """Purge everything ingested under source label ``name`` — documents and their
        chunks, embeddings, relations, and images — and refresh the ANN sidecar. Returns
        the number of documents removed (0 if the label isn't present)."""
        with Store.open(self.db_path) as store:
            doc_ids = store.document_ids_for_source(name)
            for doc_id in doc_ids:
                store.delete_document(doc_id)  # cascades chunks/embeddings/relations/images
            self._refresh_sidecar(store)
        return len(doc_ids)

    def enrich_vision(
        self,
        *,
        limit: int | None = None,
        by_size: bool = False,
        progress: vision.ProgressFn | None = None,
    ) -> vision.VisionResult:
        """Enrich retained images with cloud OCR + description (enrich.vision_images).

        Sends each not-yet-enriched image to the configured vision model once, persists the
        result, adds a searchable enrichment chunk, then embeds the new chunks (reusing the
        ingest embed path, R-REUSE-2) so hybrid search finds them and refreshes the ANN
        sidecar. Refuses with actionable guidance when vision is off."""
        provider = vision.make_vision_provider(self.config.enrich)
        if provider is None:
            raise ConfigError(
                "enrich.vision_images is off — set `enrich.vision_images: true` in your "
                "config to run image vision (it calls a paid cloud API)."
            )
        with Store.open(self.db_path) as store:
            result = vision.enrich_images(
                store,
                provider,
                staging_dir=self.config.paths.staging_dir,
                limit=limit,
                by_size=by_size,
                progress=progress,
            )
            embed_provider = self._provider()
            if embed_provider is not None and store.chunks_without_embeddings():
                ingest._embed_chunks(store, embed_provider, self.config.embed.batch_size)
                self._refresh_sidecar(store)
        return result

    def enrich_summaries(
        self,
        *,
        model: str = "claude-opus-4-8",
        runner: enrich.Runner | None = None,
        limit: int | None = None,
        progress: ProgressFn | None = None,
    ) -> SummaryResult:
        """Generate a searchable AI summary per document (§17 optional AI summaries; off by
        default). Summarizes each not-yet-summarized document once, persists it as an
        ``enrichment`` chunk (locator ``summary``), then embeds the new chunks and refreshes the
        ANN — determinism by persistence (R-SRCH-5). Idempotent: a re-run skips summarized docs.
        Refuses with actionable guidance when ``enrich.ai_summaries`` is off."""
        if not self.config.enrich.ai_summaries:
            raise ConfigError(
                "enrich.ai_summaries is off — set `enrich.ai_summaries: true` in your config to "
                "generate AI summaries (each doc is sent once to the `claude` CLI)."
            )
        result = SummaryResult()
        with Store.open(self.db_path) as store:
            pending = store.documents_needing_summary(limit or 0)
            result.pending = len(pending)
            made_chunk = False
            for done, (doc_id, _title) in enumerate(pending, 1):
                text = store.document_ingest_text(doc_id)
                if not text.strip():
                    result.skipped += 1
                else:
                    try:
                        summary = enrich.summarize_document(text, model=model, runner=runner)
                    except enrich.EnrichError as exc:
                        result.failed += 1
                        result.errors.append((doc_id, str(exc)))
                    else:
                        store.add_enrichment_chunk(doc_id, summary, "summary")
                        result.summarized += 1
                        made_chunk = True
                if progress is not None:
                    progress("summaries", done, len(pending))
            embed_provider = self._provider()
            if made_chunk and embed_provider is not None and store.chunks_without_embeddings():
                ingest._embed_chunks(store, embed_provider, self.config.embed.batch_size)
                self._refresh_sidecar(store)
        return result

    def _refresh_sidecar(self, store: Store) -> None:
        """Rebuild or remove the on-disk ANN index after the vector set changed."""
        if self.db_path == ":memory:":
            return
        ann_path = Path(self.db_path).with_suffix(".hnsw")
        if store.count_embeddings() == 0:
            store.clear_embeddings()  # drop now-stale embed_model/dim provenance too
            ann_path.unlink(missing_ok=True)
            return
        model = store.existing_embedding_model()
        if self.config.index.ann and model is not None:
            search.VectorIndex.build(
                store,
                model[1],
                ann_path,
                m=self.config.index.ann_m,
                ef_construction=self.config.index.ann_ef_construction,
            )

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

    def check_discrepancies(self, *, persist: bool = False) -> enrich.DiscrepancyReport:
        """Scan for duplicate active documents + high-similarity conflict candidates (§17). Uses the
        vector index for conflict detection when the store is embedded (BM25-only ⇒ dupes only).
        ``persist`` writes the findings as filterable ``discrepancy`` flags."""
        provider = self._provider()
        with Store.open(self.db_path) as store:
            vector_index = None
            if provider is not None and store.count_embeddings() > 0:
                index_model = store.get_meta("embed_model")
                if index_model is None or index_model == provider.model_id:
                    ann_path = (
                        Path(self.db_path).with_suffix(".hnsw")
                        if self.db_path != ":memory:"
                        else "__no_ann__"
                    )
                    vector_index = search.VectorIndex.load(store, provider.dim, ann_path)
            report = enrich.scan_discrepancies(store, vector_index=vector_index)
            if persist:
                enrich.persist_discrepancies(store, report)
            return report


def _open_member(name: str, member_config: Config) -> tuple[FederatedMember, Store]:
    """Open one federation member's store and build its search machinery (hybrid if the store holds
    embeddings for the member's configured model, else BM25), mirroring ``Catalog.search``."""
    store = Store.open(member_config.paths.db_path)
    provider = embed.make_provider(member_config.embed)
    vector_index = None
    if provider is not None and store.count_embeddings() > 0:
        index_model = store.get_meta("embed_model")
        if index_model is not None and index_model != provider.model_id:
            warnings.warn(
                f"federation member {name!r}: index embedded with {index_model!r} but embed.model "
                f"is {provider.model_id!r}; using BM25 for this member.",
                stacklevel=2,
            )
            provider = None
        else:
            ann = Path(member_config.paths.db_path).with_suffix(".hnsw")
            vector_index = search.VectorIndex.load(store, provider.dim, ann)
    return FederatedMember(store, provider, vector_index, name=name), store


@contextmanager
def open_federation(
    config: Config, *, only: list[str] | None = None
) -> Iterator[FederatedSearch]:
    """Open the member stores named in ``config.federation`` and yield a ``FederatedSearch`` over
    them (R-TEST-3, §4f), closing all stores on exit. Each member is loaded from its own config
    file, so members may use different embedding models. ``only`` restricts to those member names
    (the server passes the caller's access-permitted subset; omit for all, e.g. the local CLI).
    Scope a query further with ``stores=[...]`` on ``.search``. Member ``config`` paths are used as
    given (absolute, or relative to the current working directory)."""
    if not config.federation:
        raise ConfigError("this config has no 'federation:' members to search")
    wanted = None if only is None else set(only)
    members: list[FederatedMember] = []
    stores: list[Store] = []
    try:
        for member in config.federation:
            if wanted is not None and member.name not in wanted:
                continue
            member_config = load(Path(member.config))
            federated_member, store = _open_member(member.name, member_config)
            members.append(federated_member)
            stores.append(store)
        yield FederatedSearch(members)
    finally:
        for store in stores:
            store.close()
