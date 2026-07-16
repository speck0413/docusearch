"""Needle harness: generation determinism + retrieval on a small haystack (§15.2)."""

from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path

from docusearch import Catalog
from docusearch import config as cfg

_ROOT = Path(__file__).resolve().parents[1]


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


def test_generation_is_deterministic_and_unique() -> None:
    a = nd.generate_needles(seed=3, count=12)
    b = nd.generate_needles(seed=3, count=12)
    assert [n.nonce for n in a] == [n.nonce for n in b]  # same seed -> same nonces
    assert len({n.nonce for n in a}) == 12  # unique


def test_placements_distributed() -> None:
    counts = Counter(n.placement for n in nd.generate_needles(seed=1, count=60))
    assert counts["prose"] == 20
    assert counts["code"] == counts["table"] == counts["image"] == counts["deep"] == 10


def test_needle_retrieval_on_small_haystack(tmp_path: Path) -> None:
    filler = tmp_path / "filler"
    filler.mkdir()
    for i in range(8):
        (filler / f"f{i}.html").write_text(
            "<html><body><h1>Doc</h1><p>ordinary filler text about timing, "
            "interfaces, and configuration of the peripheral bus.</p></body></html>",
            encoding="utf-8",
        )
    needles = nd.generate_needles(seed=5, count=12)
    haystack = tmp_path / "hay"
    nd.build_haystack(filler, haystack, needles)

    config_path = tmp_path / "docusearch.yaml"
    config_path.write_text(
        f'paths:\n  staging_dir: "{(tmp_path / "staging").as_posix()}"\n'
        f'  db_path: "{(tmp_path / "catalog.db").as_posix()}"\n'
        f'  tmp_dir: "{(tmp_path / "tmp").as_posix()}"\n'
        f'sources:\n  - name: hay\n    location: "{haystack.as_posix()}"\n'
        '    include: ["**/*.html"]\n    min_content_chars: 50\n'
        'embed:\n  model: "none"\n',
        encoding="utf-8",
    )
    catalog = Catalog(cfg.load(config_path))
    catalog.ingest()
    report = nd.evaluate(catalog, needles)

    assert report.exact_top1_rate == 1.0  # every unique nonce -> top-1
    assert report.partial_recall_at10 >= 0.80


def test_render_report_mentions_gates() -> None:
    report = nd.NeedleReport(seed=1, count=60, exact_top1_rate=1.0, partial_recall_at10=0.9)
    text = nd.render_report(report)
    assert "top-1" in text and "recall@10" in text and "PASS" in text
