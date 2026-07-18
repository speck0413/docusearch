"""Phase 4e — Federated suite (R-TEST-3, §4f): one query fans out across ≥3 stores, merges with
RRF, and dedupes by content_hash — same call shape as single-store search."""

from __future__ import annotations

from pathlib import Path

from docusearch import config
from docusearch.catalog import Catalog
from docusearch.search import FederatedSearch
from docusearch.store import Store


def _store_from_docs(tmp: Path, name: str, docs: dict[str, str]) -> Path:
    """Ingest ``{filename: html}`` into its own store; return the db path."""
    corpus = tmp / f"{name}-corpus"
    corpus.mkdir(parents=True)
    for fn, html in docs.items():
        (corpus / fn).write_text(html, encoding="utf-8")
    cfg_path = tmp / f"{name}.yaml"
    cfg_path.write_text(
        f'paths:\n  staging_dir: "{(tmp / name / "s").as_posix()}"\n'
        f'  db_path: "{(tmp / name / "c.db").as_posix()}"\n  tmp_dir: "{(tmp / name / "t").as_posix()}"\n'
        f'sources:\n  - name: {name}\n    location: "{corpus.as_posix()}"\n'
        '    include: ["**/*.html"]\n    min_content_chars: 5\n'
        'embed:\n  model: "none"\n',
        encoding="utf-8",
    )
    cfg = config.load(cfg_path)
    Catalog(cfg).ingest()
    return cfg.paths.db_path


def _doc(title: str, body: str) -> str:
    return f"<body><h1>{title}</h1><p>{body}</p></body>"


def test_federated_fans_out_across_three_stores(tmp_path: Path) -> None:
    # A needle lives in a different store each — one federated query must find all three.
    a = _store_from_docs(tmp_path, "a", {"a.html": _doc("Alpha", "needle ZEBRA07 in store A prose.")})
    b = _store_from_docs(tmp_path, "b", {"b.html": _doc("Bravo", "needle QUOKKA13 in store B prose.")})
    c = _store_from_docs(tmp_path, "c", {"c.html": _doc("Charlie", "needle NARWHAL21 in store C prose.")})
    with Store.open(a) as sa, Store.open(b) as sb, Store.open(c) as sc:
        fed = FederatedSearch([sa, sb, sc])
        for needle in ("ZEBRA07", "QUOKKA13", "NARWHAL21"):
            hits = fed.search(needle, top_k=10)
            assert hits, f"{needle} not found via federation"
            assert needle in (hits[0].snippet + hits[0].title)


def test_federated_dedupes_by_content_hash(tmp_path: Path) -> None:
    # The SAME document is ingested into two stores (identical bytes -> same content_hash). A
    # federated query must return it ONCE, not once per store.
    shared = _doc("Shared", "the DUPLICATE9 calibration note appears in two federated stores.")
    a = _store_from_docs(tmp_path, "a", {"dup.html": shared, "a2.html": _doc("A2", "filler alpha.")})
    b = _store_from_docs(tmp_path, "b", {"dup.html": shared, "b2.html": _doc("B2", "filler bravo.")})
    with Store.open(a) as sa, Store.open(b) as sb:
        fed = FederatedSearch([sa, sb])
        hits = fed.search("DUPLICATE9 calibration note", top_k=10)
        dup_hits = [h for h in hits if "DUPLICATE9" in h.snippet]
        assert len(dup_hits) == 1, f"expected the shared doc once, got {len(dup_hits)}"
        assert hits[0].search_mode == "federated"


def test_federated_matches_single_combined_store(tmp_path: Path) -> None:
    # Splitting a corpus across 3 stores and federating must surface the same top document as
    # ingesting the whole corpus into one store (ranking parity for a discriminating query).
    docs = {
        "timeit.html": _doc("timeit", "measure how long a small piece of code takes to run"),
        "json.html": _doc("json", "serialize a python object to text and load it back"),
        "socket.html": _doc("socket", "open a network connection and send bytes over TCP"),
        "re.html": _doc("re", "find every place a pattern appears inside a string"),
        "os.html": _doc("os", "work with files and directories on the operating system"),
        "math.html": _doc("math", "trigonometric and logarithmic functions for floats"),
    }
    # single combined store
    single = _store_from_docs(tmp_path, "single", docs)
    # split across three stores (2 docs each)
    items = list(docs.items())
    a = _store_from_docs(tmp_path, "s1", dict(items[0:2]))
    b = _store_from_docs(tmp_path, "s2", dict(items[2:4]))
    c = _store_from_docs(tmp_path, "s3", dict(items[4:6]))
    query = "connection send bytes over the network"
    with Store.open(single) as ss, Store.open(a) as sa, Store.open(b) as sb, Store.open(c) as sc:
        from docusearch.search import bm25_search

        single_top = bm25_search(ss, query, top_k=1)[0]
        fed_top = FederatedSearch([sa, sb, sc]).search(query, top_k=1)[0]
        assert fed_top.path.rsplit("/", 1)[-1] == single_top.path.rsplit("/", 1)[-1]
