"""Hybrid search mechanics: ANN, RRF fusion, batch, role filter, determinism.

Uses the deterministic FakeProvider (no torch). Retrieval *quality* (paraphrase recall)
is validated with the real model in the Phase-2 suites; here we pin the plumbing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from docusearch import config as cfg
from docusearch import ingest, search
from docusearch.store import Store

from ._fakes import FakeProvider


def _config(tmp: Path, root: Path, *, ann: bool = True, bm25_only: bool = False) -> cfg.Config:
    path = tmp / "docusearch.yaml"
    path.write_text(
        f'paths:\n  staging_dir: "{(tmp / "s").as_posix()}"\n'
        f'  db_path: "{(tmp / "catalog.db").as_posix()}"\n'
        f'  tmp_dir: "{(tmp / "t").as_posix()}"\n'
        f'sources:\n  - name: d\n    location: "{root.as_posix()}"\n'
        "    min_content_chars: 5\n    audience: [engineering]\n"
        f'embed:\n  model: "none"\n  batch_size: 4\n'
        f"index:\n  ann: {'true' if ann else 'false'}\n"
        f"search:\n  bm25_only: {'true' if bm25_only else 'false'}\n",
        encoding="utf-8",
    )
    return cfg.load(path)


def _corpus(root: Path, n: int = 8) -> None:
    root.mkdir()
    for i in range(n):
        (root / f"doc{i}.html").write_text(
            f"<body><h1>Doc {i}</h1><p>content about topic {i} concerning timing bus number {i}</p></body>",
            encoding="utf-8",
        )


def _ingested(tmp: Path, *, ann: bool = True) -> tuple[cfg.Config, search.VectorIndex]:
    root = tmp / "docs"
    _corpus(root)
    config = _config(tmp, root, ann=ann)
    provider = FakeProvider()
    with Store.open(config.paths.db_path) as store:
        ingest.run_ingest(config, store, provider=provider)
    with Store.open(config.paths.db_path) as store:
        vi = search.VectorIndex.load(
            store, provider.dim, Path(config.paths.db_path).with_suffix(".hnsw")
        )
    return config, vi


def test_hybrid_returns_hybrid_mode_and_finds_bm25_match(tmp_path: Path) -> None:
    config, vi = _ingested(tmp_path)
    provider = FakeProvider()
    with Store.open(config.paths.db_path) as store:
        hits = search.hybrid_search(store, "topic 3", provider, vi, top_k=5, rrf_k=60)
        assert hits
        assert all(h.search_mode == "hybrid" for h in hits)
        assert all(h.embed_model_used == "fake-v1" for h in hits)
        # the doc literally about "topic 3" (BM25 side of the fusion) is retrieved
        assert any("doc3.html" in h.path for h in hits)


def test_ann_sidecar_is_built_and_used(tmp_path: Path) -> None:
    config, _ = _ingested(tmp_path, ann=True)
    assert Path(config.paths.db_path).with_suffix(".hnsw").exists()
    with Store.open(config.paths.db_path) as store:
        vi = search.VectorIndex.load(store, 8, Path(config.paths.db_path).with_suffix(".hnsw"))
        assert vi._hnsw is not None  # loaded the hnswlib index, not the numpy fallback


def test_numpy_fallback_when_no_sidecar(tmp_path: Path) -> None:
    config, _ = _ingested(tmp_path, ann=False)  # no .hnsw built
    assert not Path(config.paths.db_path).with_suffix(".hnsw").exists()
    with Store.open(config.paths.db_path) as store:
        vi = search.VectorIndex.load(store, 8, Path(config.paths.db_path).with_suffix(".hnsw"))
        assert vi._hnsw is None and vi._matrix is not None
        q = FakeProvider().embed(["content about topic 3"])[0]
        hits = vi.query(q, 3)
        assert len(hits) == 3 and hits[0][1] >= hits[-1][1]  # sorted by similarity desc


def test_batch_search_returns_per_query_lists(tmp_path: Path) -> None:
    config, vi = _ingested(tmp_path)
    provider = FakeProvider()
    with Store.open(config.paths.db_path) as store:
        out = search.search(
            store, ["topic 1", "topic 2"], provider=provider, vector_index=vi, top_k=3
        )
        assert isinstance(out, list) and len(out) == 2
        assert all(isinstance(lst, list) for lst in out)


def test_role_filter_removes_out_of_audience(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    (root / "eng.html").write_text("<body><p>engineering timing content here</p></body>", "utf-8")
    (root / "fin.html").write_text("<body><p>finance timing content here</p></body>", "utf-8")
    path = tmp_path / "docusearch.yaml"
    path.write_text(
        f'paths:\n  db_path: "{(tmp_path / "c.db").as_posix()}"\n'
        f'  staging_dir: "{(tmp_path / "s").as_posix()}"\n  tmp_dir: "{(tmp_path / "t").as_posix()}"\n'
        "sources:\n"
        f'  - name: eng\n    location: "{root.as_posix()}"\n    include: ["eng.html"]\n'
        "    min_content_chars: 5\n    audience: [engineering]\n"
        f'  - name: fin\n    location: "{root.as_posix()}"\n    include: ["fin.html"]\n'
        "    min_content_chars: 5\n    audience: [finance]\n"
        'embed:\n  model: "none"\n',
        encoding="utf-8",
    )
    config = cfg.load(path)
    with Store.open(config.paths.db_path) as store:
        ingest.run_ingest(config, store)
        eng_only = search.bm25_search(store, "timing content", top_k=10, roles={"engineering"})
        assert eng_only and all("eng.html" in h.path for h in eng_only)
        unfiltered = search.bm25_search(store, "timing content", top_k=10, roles=None)
        assert len(unfiltered) == 2  # no filtering sees both


def test_hybrid_is_deterministic_across_independent_ingests(tmp_path: Path) -> None:
    a = tmp_path / "a"
    a.mkdir()
    b = tmp_path / "b"
    b.mkdir()
    provider = FakeProvider()
    sigs = []
    for base in (a, b):
        config, vi = _ingested(base)
        with Store.open(config.paths.db_path) as store:
            hits = search.hybrid_search(store, "topic 5 timing", provider, vi, top_k=5)
            sigs.append([(h.doc_id, h.chunk_id, round(h.score, 6)) for h in hits])
    assert sigs[0] == sigs[1]  # byte-identical ranked hybrid results (R-SRCH-5)


def test_rrf_fusion_math() -> None:
    fused = search._rrf([[10, 20, 30], [30, 40]], k=60)
    # 30 appears in both lists (rank 3 in first, rank 1 in second)
    assert fused[30] == pytest.approx(1 / 63 + 1 / 61)
    assert fused[10] == pytest.approx(1 / 61)


@pytest.mark.model
@pytest.mark.filterwarnings("ignore")
def test_catalog_hybrid_end_to_end_real_model(tmp_path: Path) -> None:
    from docusearch import Catalog

    model = "sentence-transformers/all-MiniLM-L6-v2"
    root = tmp_path / "docs"
    root.mkdir()
    docs = {
        "spi.html": "the serial peripheral interface bus transfers data using a shared clock signal",
        "uart.html": "universal asynchronous receiver transmitter sends bytes over a serial line",
        "watchdog.html": "the watchdog timer resets the microcontroller when software hangs",
    }
    for name, text in docs.items():
        (root / name).write_text(f"<body><h1>{name}</h1><p>{text}</p></body>", encoding="utf-8")
    path = tmp_path / "docusearch.yaml"
    path.write_text(
        f'paths:\n  db_path: "{(tmp_path / "c.db").as_posix()}"\n'
        f'  staging_dir: "{(tmp_path / "s").as_posix()}"\n  tmp_dir: "{(tmp_path / "t").as_posix()}"\n'
        f'sources:\n  - name: d\n    location: "{root.as_posix()}"\n'
        "    min_content_chars: 5\n    audience: [engineering]\n"
        f'embed:\n  model: "{model}"\n  device: cpu\n  batch_size: 8\n',
        encoding="utf-8",
    )
    cat = Catalog(cfg.load(path))
    result = cat.ingest()
    assert result.embedded == result.chunks > 0
    hits = cat.search("how does the SPI clock work", top_k=3)
    assert hits
    assert hits[0].search_mode == "hybrid"
    assert hits[0].embed_model_used == model
    assert "spi.html" in hits[0].path  # lexical + semantic fusion surfaces the SPI doc
