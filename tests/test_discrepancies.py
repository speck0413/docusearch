"""Discrepancy scan (§17 Phase 5): duplicate ACTIVE documents + high-similarity conflict candidates,
persisted as filterable flags rows (kind=discrepancy)."""

from __future__ import annotations

import numpy as np

from docusearch import enrich
from docusearch.search import VectorIndex
from docusearch.store import Store


def _doc(store: Store, path: str, content_hash: str) -> int:
    return store.add_document(
        path=path,
        source="s",
        source_version="1",
        title=path,
        content_hash=content_hash,
        content_type="documentation",
        fmt="html",
        audience=["eng"],
        mtime=0.0,
        status="active",
    )


def test_scan_finds_duplicate_active_documents() -> None:
    with Store.open(":memory:") as store:
        d1 = _doc(store, "a.html", "HASH_SAME")
        d2 = _doc(store, "b.html", "HASH_SAME")  # byte-identical to a.html
        _doc(store, "c.html", "HASH_OTHER")  # unique — not a dup
        report = enrich.scan_discrepancies(store)
        assert len(report.duplicate_actives) == 1
        grp = report.duplicate_actives[0]
        assert grp.content_hash == "HASH_SAME"
        assert {did for did, _ in grp.docs} == {d1, d2}
        assert report.conflict_candidates == []  # no embeddings → no conflict scan


def _unit(vec: list[float]) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    return arr / np.linalg.norm(arr)


def test_scan_finds_cross_doc_near_duplicate_conflicts() -> None:
    with Store.open(":memory:") as store:
        da = _doc(store, "a.html", "HA")
        db = _doc(store, "b.html", "HB")
        # chunk 1 (doc A) and chunk 2 (doc B) are near-identical; chunk 3 (doc A) is unrelated.
        c1 = store.add_chunk(document_id=da, ord=0, text="strobe timing setup for the bus")
        c2 = store.add_chunk(document_id=db, ord=0, text="strobe timing setup for the bus, v2")
        c3 = store.add_chunk(document_id=da, ord=1, text="unrelated calibration content")
        vecs = {
            c1: _unit([1.0, 0.02, 0.0]),
            c2: _unit([0.985, 0.05, 0.0]),  # ~0.999-ish? tuned into the band below
            c3: _unit([0.0, 0.0, 1.0]),
        }
        dim = 3
        store.add_embeddings(
            [(cid, "fake-v1", dim, v.astype(np.float32).tobytes()) for cid, v in vecs.items()]
        )
        matrix = np.stack([vecs[c1], vecs[c2], vecs[c3]])
        index = VectorIndex(dim, matrix=matrix, ids=[c1, c2, c3])

        report = enrich.scan_discrepancies(store, vector_index=index, sim_lo=0.90, sim_hi=0.9999)
        assert len(report.conflict_candidates) == 1
        pair = report.conflict_candidates[0]
        assert {pair.chunk_a, pair.chunk_b} == {c1, c2}
        assert {pair.doc_a, pair.doc_b} == {da, db}  # cross-document
        assert 0.90 <= pair.similarity < 0.9999
        # c3 is not similar to anything; same-doc pairs excluded
        assert c3 not in {pair.chunk_a, pair.chunk_b}


def test_persist_discrepancies_writes_flags() -> None:
    with Store.open(":memory:") as store:
        d1 = _doc(store, "a.html", "H")
        d2 = _doc(store, "b.html", "H")
        report = enrich.scan_discrepancies(store)
        n = enrich.persist_discrepancies(store, report)
        assert n == 2  # one flag per duplicate doc
        assert store.count_flags("discrepancy") == 2
        flags = store.flags_for_document(d1)
        assert any(f["rule_id"] == "duplicate-active" for f in flags)
        assert str(d2) in " ".join(str(f["note"]) for f in flags)
        # a re-scan replaces prior findings rather than doubling them
        enrich.persist_discrepancies(store, enrich.scan_discrepancies(store))
        assert store.count_flags("discrepancy") == 2
