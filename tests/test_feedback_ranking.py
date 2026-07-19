"""Feedback-aware re-ranking (Phase 8 / #63): source tier (internal > vendor) + a user's feedback
(feedback > internal > vendor) re-order otherwise-equal relevance hits."""

from __future__ import annotations

from pathlib import Path

from docusearch import config as cfg
from docusearch import ingest
from docusearch.server import Service
from docusearch.store import Store

_DOC = "widget calibration procedure alpha bravo charlie\n"  # identical text → BM25 tie


def _service(tmp_path: Path) -> tuple[Service, int, int]:
    ven, intr = tmp_path / "vendor", tmp_path / "internal"
    ven.mkdir()
    intr.mkdir()
    (ven / "v.md").write_text(_DOC, encoding="utf-8")
    (intr / "i.md").write_text(_DOC, encoding="utf-8")
    path = tmp_path / "d.yaml"
    path.write_text(
        f'paths:\n  staging_dir: "{(tmp_path / "s").as_posix()}"\n'
        f'  db_path: "{(tmp_path / "c.db").as_posix()}"\n  tmp_dir: "{(tmp_path / "t").as_posix()}"\n'
        f'sources:\n  - name: vendor\n    location: "{ven.as_posix()}"\n'
        '    include: ["*.md"]\n    min_content_chars: 1\n    tier: "vendor"\n'
        f'  - name: internal\n    location: "{intr.as_posix()}"\n'
        '    include: ["*.md"]\n    min_content_chars: 1\n    tier: "internal"\n'
        'embed:\n  model: "none"\n', encoding="utf-8")
    config = cfg.load(path)
    with Store.open(config.paths.db_path) as store:
        ingest.run_ingest(config, store)
        vid = store.document_ids_for_source("vendor")[0]
        iid = store.document_ids_for_source("internal")[0]
    return Service(config), vid, iid


def _order(svc: Service, user: str | None = None) -> list[int]:
    results, _m, _mode = svc.search(["widget calibration alpha"], user=user)
    return [h.doc_id for h in results[0]]


def test_internal_outranks_vendor_on_a_tie(tmp_path: Path) -> None:
    svc, vid, iid = _service(tmp_path)
    order = _order(svc)
    assert order.index(iid) < order.index(vid)  # internal tier boost breaks the BM25 tie


def test_positive_feedback_lifts_vendor_above_internal(tmp_path: Path) -> None:
    svc, vid, iid = _service(tmp_path)
    # a global +1 on the vendor doc: feedback_weight (0.03) > internal_boost (0.02) → vendor wins
    svc.submit_feedback(user="alice", text="this vendor page is the right answer",
                        doc_id=vid, rating=1, make_global=True)
    order = _order(svc)
    assert order.index(vid) < order.index(iid)  # feedback > internal


def test_private_feedback_only_affects_its_author(tmp_path: Path) -> None:
    svc, vid, iid = _service(tmp_path)
    svc.submit_feedback(user="alice", text="prefer vendor", doc_id=vid, rating=1)  # private
    assert _order(svc, user="alice").index(vid) < _order(svc, user="alice").index(iid)
    # bob doesn't see alice's private feedback → internal (tier) still wins for him
    assert _order(svc, user="bob").index(iid) < _order(svc, user="bob").index(vid)
