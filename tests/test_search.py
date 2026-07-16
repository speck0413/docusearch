"""BM25 search: ranking, sanitizer safety, determinism (R-SRCH-1, R-SRCH-5, §9)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from docusearch import search
from docusearch.store import Store


@pytest.fixture
def store() -> Iterator[Store]:
    with Store.open(":memory:") as s:
        d1 = s.add_document(path="/spi.html", title="SPI", fmt="html")
        s.add_chunk(
            document_id=d1,
            ord=0,
            text="SPI timing configuration for the peripheral bus interface",
            kind="body",
            locator="Interfaces > SPI",
        )
        d2 = s.add_document(path="/uart.html", title="UART", fmt="html")
        s.add_chunk(document_id=d2, ord=0, text="UART baud rate timing", kind="body")
        d3 = s.add_document(path="/needle.html", title="N", fmt="html")
        s.add_chunk(document_id=d3, ord=0, text="the nonce ZQX7734 lives here", kind="body")
        yield s


def test_bm25_finds_and_ranks(store: Store) -> None:
    hits = search.bm25_search(store, "SPI timing", top_k=10)
    assert hits
    # the chunk containing both terms should rank first
    assert hits[0].title == "SPI"
    assert "timing" in hits[0].snippet.lower()


def test_top_k_limits_results(store: Store) -> None:
    hits = search.bm25_search(store, "timing", top_k=1)
    assert len(hits) == 1


def test_sanitizer_neutralizes_operators_and_injection(store: Store) -> None:
    # raw FTS operators / punctuation must never crash MATCH (§9)
    for q in ["SPI OR timing", '"; drop table chunks; --', "timing AND NOT bus", "NEAR("]:
        hits = search.bm25_search(store, q, top_k=10)
        assert isinstance(hits, list)  # no exception


def test_empty_query_returns_empty(store: Store) -> None:
    assert search.bm25_search(store, "   ", top_k=10) == []
    assert search.bm25_search(store, "!!!", top_k=10) == []


def test_prefix_matches_partial_nonce(store: Store) -> None:
    assert not search.bm25_search(store, "ZQX773", top_k=10)  # exact term miss
    hits = search.bm25_search(store, "ZQX773", top_k=10, prefix=True)
    assert hits and hits[0].title == "N"


def test_hit_carries_citation_and_locator(store: Store) -> None:
    hits = search.bm25_search(store, "SPI", top_k=10)
    hit = hits[0]
    assert hit.citation == f"D:{hit.doc_id}#{hit.chunk_id}"
    assert hit.locator == "Interfaces > SPI"


def test_deterministic_order_including_ties() -> None:
    with Store.open(":memory:") as s:
        for i in range(5):
            d = s.add_document(path=f"/d{i}.html", title=f"T{i}", fmt="html")
            s.add_chunk(document_id=d, ord=0, text="identical timing text", kind="body")
        first = [(h.doc_id, h.chunk_id) for h in search.bm25_search(s, "timing", top_k=10)]
        second = [(h.doc_id, h.chunk_id) for h in search.bm25_search(s, "timing", top_k=10)]
        assert first == second
        # tie-break is ascending on (doc_id, chunk_id)
        assert first == sorted(first)
