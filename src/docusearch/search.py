"""Search: BM25 today, hybrid (vector + RRF) later (§9, R-SRCH-*).

BM25 over the FTS5 index is always available and good enough to run the whole system
alone (R-SRCH-1). A sanitizer quotes arbitrary user text so raw strings — including FTS
operators and punctuation — never crash MATCH (§9). Ranking is deterministic: identical
index + query ⇒ identical ranked results, tie-broken on (doc id, chunk id) (R-SRCH-5).

Public surface (grows in Phase 2 with hybrid/batch/role-filter):
    SearchHit                                             # one ranked result (§9 shape)
    sanitize_query(text, *, prefix=False) -> str          # safe FTS5 MATCH string
    bm25_search(store, query, *, top_k, prefix) -> list[SearchHit]
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .store import Store

_TOKEN = re.compile(r"\w+", re.UNICODE)


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
    images: list[str] = field(default_factory=list)


_MAX_TERMS = 64


def sanitize_query(text: str, *, prefix: bool = False) -> str:
    """Turn arbitrary user text into a safe FTS5 MATCH string.

    Each word becomes a quoted literal term (neutralizing FTS operators and punctuation)
    and terms are implicitly AND-ed. With ``prefix=True`` each term also matches by
    prefix (``"term"*``) — used by the partial/misspelled-needle suite (§15.2).

    Duplicate terms are dropped and the term count is capped (``_MAX_TERMS``) so a
    pathologically long or repetitive query cannot cost O(n^2) in FTS (red-team Finding 2).
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
    return " ".join(f'"{term}"{suffix}' for term in terms)


def bm25_search(
    store: Store,
    query: str,
    *,
    top_k: int = 10,
    prefix: bool = False,
) -> list[SearchHit]:
    """Rank chunks for ``query`` by BM25, best first (R-SRCH-1)."""
    match = sanitize_query(query, prefix=prefix)
    if not match:
        return []
    hits: list[SearchHit] = []
    for row in store.bm25(match, top_k):
        doc_id = int(row["doc_id"])
        chunk_id = int(row["chunk_id"])
        hits.append(
            SearchHit(
                doc_id=doc_id,
                chunk_id=chunk_id,
                title=str(row["title"] or ""),
                path=str(row["path"] or ""),
                fmt=str(row["fmt"] or ""),
                locator=str(row["locator"] or ""),
                kind=str(row["kind"] or ""),
                snippet=str(row["snippet"] or ""),
                score=round(-float(row["bm25"]), 6),  # higher = better (bm25 is lower-better)
                citation=f"D:{doc_id}#{chunk_id}",
            )
        )
    return hits
