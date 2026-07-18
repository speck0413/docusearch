"""Needle harness: generation determinism + retrieval (BM25 + hybrid paraphrase) (§15.2)."""

from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path

import pytest

from docusearch import config as cfg
from docusearch import embed, search
from docusearch.catalog import Catalog
from docusearch.store import Store

_ROOT = Path(__file__).resolve().parents[1]
MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _load_needles():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(
        "harness_needles", _ROOT / "harness" / "needles.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


nd = _load_needles()


def _haystack_config(tmp: Path, needles, *, model: str) -> cfg.Config:  # type: ignore[no-untyped-def]
    filler = tmp / "filler"
    filler.mkdir()
    for i in range(8):
        (filler / f"f{i}.html").write_text(
            "<html><body><h1>Doc</h1><p>ordinary filler text about timing, interfaces, "
            "and configuration of the peripheral bus.</p></body></html>",
            encoding="utf-8",
        )
    haystack = tmp / "hay"
    nd.build_haystack(filler, haystack, needles)
    path = tmp / "docusearch.yaml"
    path.write_text(
        f'paths:\n  staging_dir: "{(tmp / "s").as_posix()}"\n'
        f'  db_path: "{(tmp / "catalog.db").as_posix()}"\n'
        f'  tmp_dir: "{(tmp / "t").as_posix()}"\n'
        f'sources:\n  - name: hay\n    location: "{haystack.as_posix()}"\n'
        '    include: ["**/*.html"]\n    min_content_chars: 50\n'
        f'embed:\n  model: "{model}"\n  device: cpu\n',
        encoding="utf-8",
    )
    return cfg.load(path)


def test_generation_is_deterministic_and_unique() -> None:
    a = nd.generate_needles(seed=3, count=12)
    b = nd.generate_needles(seed=3, count=12)
    assert [n.nonce for n in a] == [n.nonce for n in b]
    assert len({n.nonce for n in a}) == 12
    assert len({n.subject for n in a}) == 12  # unique subjects for discriminative paraphrases


def test_placements_distributed() -> None:
    counts = Counter(n.placement for n in nd.generate_needles(seed=1, count=60))
    assert counts["prose"] == 20
    assert counts["code"] == counts["table"] == counts["image"] == counts["deep"] == 10


def test_needle_bm25_exact_and_partial(tmp_path: Path) -> None:
    needles = nd.generate_needles(seed=5, count=12)
    config = _haystack_config(tmp_path, needles, model="none")
    Catalog(config).ingest()
    with Store.open(config.paths.db_path) as store:
        report = nd.evaluate(store, needles)
    assert report.exact_top1_rate == 1.0
    assert report.partial_recall_at10 >= 0.80
    assert report.hybrid_paraphrase_recall_at5 is None  # no embeddings -> not evaluated


@pytest.mark.model
@pytest.mark.filterwarnings("ignore")
def test_needle_paraphrase_hybrid(tmp_path: Path) -> None:
    needles = nd.generate_needles(seed=3, count=10)
    config = _haystack_config(tmp_path, needles, model=MODEL)
    Catalog(config).ingest()
    with Store.open(config.paths.db_path) as store:
        provider = embed.make_provider(config.embed)
        vi = search.VectorIndex.load(
            store, provider.dim, Path(config.paths.db_path).with_suffix(".hnsw")
        )
        report = nd.evaluate(store, needles, provider=provider, vector_index=vi)
    assert report.exact_top1_rate == 1.0
    assert report.hybrid_paraphrase_recall_at5 is not None
    assert report.hybrid_paraphrase_recall_at5 >= 0.90


def test_needles_survive_pdf_conversion(tmp_path: Path) -> None:
    # §15.4 needles-through-conversion: the SAME 60 needles, but routed HTML -> PDF -> ingest.
    # A needle lost here is a converter/extractor defect (every placement — prose/code/table/
    # image-alt/deep — must still be recoverable by exact nonce from the PDF text layer).
    needles = nd.generate_needles(seed=5, count=60)
    filler = tmp_path / "filler"
    filler.mkdir()
    for i in range(4):
        (filler / f"f{i}.html").write_text(
            "<html><body><h1>Doc</h1><p>ordinary filler text about timing and interfaces of "
            "the peripheral bus, long enough to clear the content filter.</p></body></html>",
            encoding="utf-8",
        )
    pdf_dir = nd.build_pdf_haystack(filler, tmp_path / "work", needles)
    assert (pdf_dir / "needle_000.pdf").is_file()  # needle files converted alongside filler

    config = nd.conversion_config(tmp_path, pdf_dir, model="none", min_chars=5)
    Catalog(config).ingest()
    with Store.open(config.paths.db_path) as store:
        report = nd.evaluate(store, needles)
    assert report.exact_top1_rate == 1.0  # every placement's nonce survived the round trip
    assert report.partial_recall_at10 >= 0.80
    # every placement fully recovered
    for placement in ("prose", "code", "table", "image", "deep"):
        hit, tot = report.per_placement[placement]
        assert hit == tot, f"{placement}: only {hit}/{tot} needles survived PDF conversion"


def test_render_report_mentions_gates() -> None:
    report = nd.NeedleReport(
        seed=1,
        count=60,
        exact_top1_rate=1.0,
        partial_recall_at10=0.85,
        hybrid_paraphrase_recall_at5=0.92,
        bm25_paraphrase_recall_at5=0.70,
    )
    text = nd.render_report(report)
    assert "top-1" in text and "recall@10" in text and "recall@5" in text and "PASS" in text
