"""Phase-2 harness logic: obtuse (auto-QA + negatives) and compare (§15.3–15.4)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from docusearch import config as cfg
from docusearch import ingest
from docusearch.store import Store

_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, filename: str):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(name, _ROOT / "harness" / filename)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ob = _load("harness_obtuse", "obtuse.py")
cmp = _load("harness_compare", "compare.py")


def test_auto_qa_yaml_is_well_formed() -> None:
    qa = ob.load_qa(_ROOT / "harness" / "auto_qa.yaml")
    assert len(qa) >= 10
    for entry in qa:
        assert {"id", "question", "expect_docs"} <= set(entry)
        assert isinstance(entry["expect_docs"], list) and entry["expect_docs"]


def _bm25_corpus(tmp: Path) -> cfg.Config:
    root = tmp / "docs" / "library"
    root.mkdir(parents=True)
    (root / "timeit.html").write_text(
        "<body><h1>timeit</h1><p>measure how long a piece of code takes to run with a timer</p></body>",
        encoding="utf-8",
    )
    (root / "json.html").write_text(
        "<body><h1>json</h1><p>serialize a python object to text and load it back from a string</p></body>",
        encoding="utf-8",
    )
    path = tmp / "docusearch.yaml"
    path.write_text(
        f'paths:\n  db_path: "{(tmp / "c.db").as_posix()}"\n'
        f'  staging_dir: "{(tmp / "s").as_posix()}"\n  tmp_dir: "{(tmp / "t").as_posix()}"\n'
        f'sources:\n  - name: d\n    location: "{(tmp / "docs").as_posix()}"\n    min_content_chars: 5\n'
        'embed:\n  model: "none"\n',
        encoding="utf-8",
    )
    return cfg.load(path)


def test_evaluate_qa_bm25_and_negatives(tmp_path: Path) -> None:
    config = _bm25_corpus(tmp_path)
    with Store.open(config.paths.db_path) as store:
        ingest.run_ingest(config, store)
        qa = [
            {
                "id": "q1",
                "question": "measure how long code takes to run",
                "expect_docs": ["timeit.html"],
            },
            {"id": "q2", "question": "serialize object to text", "expect_docs": ["json.html"]},
        ]
        total, hybrid_r, bm25_r, misses = ob.evaluate_qa(store, qa)
        assert total == 2
        assert hybrid_r is None  # no embeddings -> hybrid not evaluated
        assert bm25_r >= 0.5
        neg_total, neg_fp = ob.run_negatives(store, seed=1, count=10)
        assert neg_total == 10 and neg_fp == 0  # fabricated topics -> no lexical hits


def test_absent_topics_deterministic_and_unique() -> None:
    assert ob.absent_topics(1, 5) == ob.absent_topics(1, 5)
    assert len(set(ob.absent_topics(1, 20))) == 20


def test_obtuse_report_renders() -> None:
    report = ob.QAReport(
        total=12,
        hybrid_recall_at10=0.9,
        bm25_recall_at10=0.5,
        negatives_total=20,
        negative_false_positives=0,
    )
    text = ob.render_report(report)
    assert "recall@10" in text and "PASS" in text and "negatives" in text.lower()


def test_compare_overlap_and_render() -> None:
    c = cmp.QueryComparison("q", ["a/1.html", "a/2.html"], ["a/1.html", "a/3.html"])
    assert c.overlap_at_k == 0.5  # 1 shared / max(2, 2)
    text = cmp.render_compare([c])
    assert "BM25" in text and "hybrid" in text and "overlap" in text
