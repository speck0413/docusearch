"""Search: BM25, vector ANN, and hybrid RRF fusion (§9, R-SRCH-*).

BM25 over FTS5 is always available and good enough alone (R-SRCH-1). When embeddings
exist, a vector index (hnswlib cosine ANN, numpy brute-force fallback) is fused with
BM25 via Reciprocal Rank Fusion (R-SRCH-2). Batch queries (R-SRCH-3) and cooperative
role filtering (R-SRCH-4) are supported. Ranking is deterministic: identical index +
query ⇒ identical ranked results, tie-broken on (doc id, chunk id) (R-SRCH-5).

Public surface:
    SearchHit                                     # one ranked result (§9 shape)
    sanitize_query(text, *, prefix=False) -> str
    VectorIndex                                   # build/load; query(vec, k)
    roles_from_env() -> set[str] | None           # DOCUSEARCH_ROLES cooperative filter
    bm25_search(store, query, *, top_k, prefix, roles) -> list[SearchHit]
    hybrid_search(store, query, provider, vector_index, *, top_k, rrf_k, prefix, roles)
    search(store, queries, *, ...) -> list[SearchHit] | list[list[SearchHit]]   # batch
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .store import Store

if TYPE_CHECKING:
    from .embed import EmbedProvider

_TOKEN = re.compile(r"\w+", re.UNICODE)
_MAX_TERMS = 64
_SNIPPET_CHARS = 180


@dataclass
class SearchHit:
    """One ranked result — also the REST/MCP JSON shape (§9)."""

    doc_id: int
    chunk_id: int
    title: str
    path: str
    fmt: str
    locator: str
    kind: str
    snippet: str
    score: float
    citation: str
    audience: list[str] = field(default_factory=list)
    images: list[str] = field(default_factory=list)
    embed_model_used: str = "none"
    search_mode: str = "bm25"


def sanitize_query(text: str, *, prefix: bool = False) -> str:
    """Turn arbitrary user text into a safe FTS5 MATCH string.

    Each word becomes a quoted literal term (neutralizing FTS operators and punctuation)
    joined with **OR**, so a natural-language query retrieves documents matching *any*
    term, ranked by BM25 (docs matching more/rarer terms rank higher) — matching all
    terms (AND) is too strict for real questions. With ``prefix=True`` each term also
    matches by prefix (``"term"*``). Duplicate terms are dropped and the count is capped
    (``_MAX_TERMS``) so a long/repetitive query cannot cost O(n^2) in FTS.
    """
    seen: set[str] = set()
    terms: list[str] = []
    for term in _TOKEN.findall(text.lower()):
        if term not in seen:
            seen.add(term)
            terms.append(term)
            if len(terms) >= _MAX_TERMS:
                break
    if not terms:
        return ""
    suffix = "*" if prefix else ""
    return " OR ".join(f'"{term}"{suffix}' for term in terms)


def roles_from_env() -> set[str] | None:
    """Caller roles from ``DOCUSEARCH_ROLES`` (comma-separated); None = no filtering."""
    raw = os.environ.get("DOCUSEARCH_ROLES", "").strip()
    if not raw:
        return None
    return {role.strip() for role in raw.split(",") if role.strip()}


def _audience(value: Any) -> list[str]:
    try:
        parsed = json.loads(value) if value else []
    except (ValueError, TypeError):
        return []
    return [str(x) for x in parsed] if isinstance(parsed, list) else []


def _snippet(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= _SNIPPET_CHARS else text[:_SNIPPET_CHARS] + " …"


def _filter_roles(hits: list[SearchHit], roles: set[str] | None) -> list[SearchHit]:
    """Drop results whose document audience ∩ caller roles = ∅ (R-SRCH-4, cooperative)."""
    if roles is None:
        return hits
    return [hit for hit in hits if set(hit.audience) & roles]


def _pool_size(top_k: int, roles: set[str] | None) -> int:
    # over-fetch when role-filtering so the filter runs before the cutoff (§9)
    return top_k if roles is None else max(top_k * 5, 50)


# ----------------------------------------------------------------- vector index


class VectorIndex:
    """Cosine nearest-neighbour over chunk embeddings: hnswlib when a saved index exists,
    else an exact numpy brute-force fallback (both deterministic)."""

    def __init__(
        self,
        dim: int,
        *,
        hnsw: Any = None,
        matrix: np.ndarray | None = None,
        ids: list[int] | None = None,
    ) -> None:
        self.dim = dim
        self._hnsw = hnsw
        self._matrix = matrix
        self._ids = ids or []

    @staticmethod
    def _stack(rows: list[tuple[int, bytes]], dim: int) -> tuple[np.ndarray, list[int]]:
        if not rows:
            return np.zeros((0, dim), dtype=np.float32), []
        vectors = []
        for cid, blob in rows:
            vec = np.frombuffer(blob, dtype=np.float32)
            if vec.shape[0] != dim:  # a mixed-model index (e.g. interrupted model swap)
                raise ValueError(
                    f"chunk {cid} has a {vec.shape[0]}-dim vector but the index dimension "
                    f"is {dim}: the embeddings table mixes models. Heal it with "
                    f"`docusearch ingest --reembed` (or use a fresh db_path)."
                )
            vectors.append(vec)
        return np.stack(vectors).astype(np.float32), [cid for cid, _ in rows]

    @classmethod
    def build(
        cls,
        store: Store,
        dim: int,
        path: Path | str,
        *,
        m: int = 16,
        ef_construction: int = 200,
    ) -> VectorIndex:
        """Build a fresh hnswlib index from all stored embeddings and save it to ``path``."""
        import hnswlib

        rows = store.all_embeddings()
        matrix, ids = cls._stack(rows, dim)
        index = hnswlib.Index(space="cosine", dim=dim)
        index.set_num_threads(1)  # deterministic build (R-SRCH-5)
        index.init_index(
            max_elements=max(len(ids), 1), ef_construction=ef_construction, M=m, random_seed=100
        )
        if ids:
            index.add_items(matrix, ids)
        index.save_index(str(path))
        return cls(dim, hnsw=index)

    @classmethod
    def load(cls, store: Store, dim: int, path: Path | str, *, ef: int = 64) -> VectorIndex:
        """Load the saved hnswlib index if present, else fall back to numpy brute force."""
        rows = store.all_embeddings()
        if Path(path).exists():
            try:
                import hnswlib

                index = hnswlib.Index(space="cosine", dim=dim)
                index.load_index(str(path), max_elements=max(len(rows), 1))
                index.set_num_threads(1)
                index.set_ef(max(ef, 16))
                return cls(dim, hnsw=index)
            except Exception:  # noqa: BLE001 - fall back to numpy on any load failure
                pass
        matrix, ids = cls._stack(rows, dim)
        return cls(dim, matrix=matrix, ids=ids)

    def query(self, vector: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        """Return up to ``top_k`` (chunk_id, cosine_similarity) pairs, best first."""
        vec = np.asarray(vector, dtype=np.float32)
        if self._hnsw is not None:
            count = self._hnsw.get_current_count()
            if count == 0:
                return []
            k = min(top_k, count)
            self._hnsw.set_ef(max(k, 16))
            labels, distances = self._hnsw.knn_query(vec.reshape(1, -1), k=k)
            return [
                (int(lbl), 1.0 - float(dist))
                for lbl, dist in zip(labels[0], distances[0], strict=False)
            ]
        if self._matrix is None or not self._ids:
            return []
        sims = self._matrix @ vec
        order = np.argsort(-sims, kind="stable")[:top_k]
        return [(self._ids[i], float(sims[i])) for i in order]


# ----------------------------------------------------------------- ranking


def _rrf(ranked_lists: Sequence[Sequence[int]], k: int) -> dict[int, float]:
    """Reciprocal Rank Fusion: sum of 1/(k + rank) across lists (§9, R-SRCH-2)."""
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return scores


def _bm25_hit(row: Any) -> SearchHit:
    doc_id, chunk_id = int(row["doc_id"]), int(row["chunk_id"])
    return SearchHit(
        doc_id=doc_id,
        chunk_id=chunk_id,
        title=str(row["title"] or ""),
        path=str(row["path"] or ""),
        fmt=str(row["fmt"] or ""),
        locator=str(row["locator"] or ""),
        kind=str(row["kind"] or ""),
        snippet=str(row["snippet"] or ""),
        score=round(-float(row["bm25"]), 6),  # higher = better
        citation=f"D:{doc_id}#{chunk_id}",
        audience=_audience(row["audience"]),
        search_mode="bm25",
    )


def bm25_search(
    store: Store,
    query: str,
    *,
    top_k: int = 10,
    prefix: bool = False,
    roles: set[str] | None = None,
) -> list[SearchHit]:
    """Rank chunks for ``query`` by BM25, best first (R-SRCH-1), role-filtered."""
    match = sanitize_query(query, prefix=prefix)
    if not match:
        return []
    hits = [_bm25_hit(row) for row in store.bm25(match, _pool_size(top_k, roles))]
    return _filter_roles(hits, roles)[:top_k]


def hybrid_search(
    store: Store,
    query: str,
    provider: EmbedProvider,
    vector_index: VectorIndex,
    *,
    top_k: int = 10,
    rrf_k: int = 60,
    prefix: bool = False,
    roles: set[str] | None = None,
) -> list[SearchHit]:
    """Fuse BM25 and vector rankings with RRF (R-SRCH-2), role-filtered, deterministic."""
    pool = _pool_size(top_k, roles)
    match = sanitize_query(query, prefix=prefix)
    bm25_ids = [int(row["chunk_id"]) for row in store.bm25(match, pool)] if match else []
    query_vec = provider.embed([query])[0]
    vec_ids = [cid for cid, _ in vector_index.query(query_vec, pool)]

    scores = _rrf([bm25_ids, vec_ids], rrf_k)
    hydrated = store.hydrate_chunks(list(scores))
    ranked = sorted(
        (
            (score, int(hydrated[cid]["doc_id"]), cid)
            for cid, score in scores.items()
            if cid in hydrated
        ),
        key=lambda t: (-t[0], t[1], t[2]),
    )
    hits: list[SearchHit] = []
    for score, doc_id, chunk_id in ranked:
        row = hydrated[chunk_id]
        hits.append(
            SearchHit(
                doc_id=doc_id,
                chunk_id=chunk_id,
                title=str(row["title"] or ""),
                path=str(row["path"] or ""),
                fmt=str(row["fmt"] or ""),
                locator=str(row["locator"] or ""),
                kind=str(row["kind"] or ""),
                snippet=_snippet(str(row["text"] or "")),
                score=round(score, 6),
                citation=f"D:{doc_id}#{chunk_id}",
                audience=_audience(row["audience"]),
                embed_model_used=provider.model_id,
                search_mode="hybrid",
            )
        )
    return _filter_roles(hits, roles)[:top_k]


def vector_search(
    store: Store,
    query_vector: np.ndarray,
    vector_index: VectorIndex,
    *,
    top_k: int = 10,
    roles: set[str] | None = None,
) -> list[SearchHit]:
    """Rank chunks by cosine similarity to a PRE-COMPUTED query vector (client sent it).

    Used by the query-vectors path (§8): the server does no embedding here — it trusts
    the vector after verifying the model tag matched (the caller enforces that, R-EMB-3).
    """
    ranked = vector_index.query(query_vector, _pool_size(top_k, roles))
    hydrated = store.hydrate_chunks([cid for cid, _ in ranked])
    hits: list[SearchHit] = []
    for chunk_id, sim in ranked:
        row = hydrated.get(chunk_id)
        if row is None:
            continue
        doc_id = int(row["doc_id"])
        hits.append(
            SearchHit(
                doc_id=doc_id,
                chunk_id=chunk_id,
                title=str(row["title"] or ""),
                path=str(row["path"] or ""),
                fmt=str(row["fmt"] or ""),
                locator=str(row["locator"] or ""),
                kind=str(row["kind"] or ""),
                snippet=_snippet(str(row["text"] or "")),
                score=round(float(sim), 6),
                citation=f"D:{doc_id}#{chunk_id}",
                audience=_audience(row["audience"]),
                search_mode="vector",
            )
        )
    return _filter_roles(hits, roles)[:top_k]


def search(
    store: Store,
    queries: str | Sequence[str],
    *,
    top_k: int = 10,
    provider: EmbedProvider | None = None,
    vector_index: VectorIndex | None = None,
    rrf_k: int = 60,
    prefix: bool = False,
    roles: set[str] | None = None,
    bm25_only: bool = False,
) -> list[SearchHit] | list[list[SearchHit]]:
    """Search one query or a batch (R-SRCH-3). Hybrid when a provider + index are given
    and ``bm25_only`` is False, else BM25. A single string returns one list; a sequence
    returns a list of per-query lists."""
    single = isinstance(queries, str)
    query_list: list[str] = [queries] if isinstance(queries, str) else list(queries)
    use_hybrid = provider is not None and vector_index is not None and not bm25_only
    results: list[list[SearchHit]] = []
    for query in query_list:
        if use_hybrid:
            assert provider is not None and vector_index is not None
            results.append(
                hybrid_search(
                    store,
                    query,
                    provider,
                    vector_index,
                    top_k=top_k,
                    rrf_k=rrf_k,
                    prefix=prefix,
                    roles=roles,
                )
            )
        else:
            results.append(bm25_search(store, query, top_k=top_k, prefix=prefix, roles=roles))
    return results[0] if single else results
