"""N-hop related_documents over the resolved relations graph (R-ING-5, §17 Phase 5).

A recursive-CTE walk: out (docs this one links to), in (docs that link to it), both, up to N hops,
returning each reachable doc's SHORTEST hop count. Cycle-safe via the depth cap.
"""

from __future__ import annotations

from docusearch.store import Store


def _doc(store: Store, path: str) -> int:
    return store.add_document(
        path=path,
        source="s",
        source_version="1",
        title=path.upper(),
        content_hash=path,
        content_type="documentation",
        fmt="html",
        audience=["eng"],
        mtime=0.0,
        status="active",
    )


def _link(store: Store, src: int, dst: int, dst_path: str, link_type: str = "ref") -> None:
    rid = store.add_relation(src_doc=src, dst_raw=dst_path, link_type=link_type)
    store.set_relation_dst(rid, dst)


def test_related_documents_out_in_and_nhop() -> None:
    # Graph:  A -> B -> C   and   D -> A
    with Store.open(":memory:") as store:
        a, b, c, d = (_doc(store, x) for x in ("a", "b", "c", "d"))
        _link(store, a, b, "b", "see-also")
        _link(store, b, c, "c")
        _link(store, d, a, "a")

        out1 = store.related_documents(a, "out", depth=1)
        assert [(r["doc_id"], r["hops"]) for r in out1] == [(b, 1)]
        assert out1[0]["link_type"] == "see-also" and out1[0]["direction"] == "out"

        out2 = store.related_documents(a, "out", depth=2)
        assert [(r["doc_id"], r["hops"]) for r in out2] == [(b, 1), (c, 2)]
        assert out2[1]["link_type"] == ""  # link_type only for direct (1-hop) neighbours

        in1 = store.related_documents(a, "in", depth=1)
        assert [(r["doc_id"], r["direction"]) for r in in1] == [(d, "in")]

        both2 = store.related_documents(a, "both", depth=2)
        assert {(r["doc_id"], r["hops"], r["direction"]) for r in both2} == {
            (b, 1, "out"),
            (c, 2, "out"),
            (d, 1, "in"),
        }


def test_related_documents_is_cycle_safe_and_excludes_self() -> None:
    # A <-> B cycle must not spin, and A must never list itself.
    with Store.open(":memory:") as store:
        a, b = _doc(store, "a"), _doc(store, "b")
        _link(store, a, b, "b")
        _link(store, b, a, "a")
        res = store.related_documents(a, "both", depth=5)
        assert {r["doc_id"] for r in res} == {b}  # only B, and no infinite loop
        assert all(r["doc_id"] != a for r in res)
