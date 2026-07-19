"""Feedback as first-class, ranking-eligible data (Phase 8 / #63): per-user by default, promotable to
global, stored in the DB with a rating + target."""

from __future__ import annotations

from pathlib import Path

from docusearch import config as cfg
from docusearch.server import Service
from docusearch.store import Store


def _service(tmp_path: Path) -> Service:
    src = tmp_path / "seed"
    src.mkdir()
    path = tmp_path / "d.yaml"
    path.write_text(
        f'paths:\n  staging_dir: "{(tmp_path / "s").as_posix()}"\n'
        f'  db_path: "{(tmp_path / "c.db").as_posix()}"\n  tmp_dir: "{(tmp_path / "t").as_posix()}"\n'
        f'sources:\n  - name: seed\n    location: "{src.as_posix()}"\n    min_content_chars: 1\n'
        'embed:\n  model: "none"\n', encoding="utf-8")
    return Service(cfg.load(path))


def test_feedback_private_by_default_and_global_opt_in(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    a = svc.submit_feedback(user="alice", text="doc 5 is wrong", doc_id=5, rating=-1)
    svc.submit_feedback(user="alice", text="doc 7 is the canonical answer", doc_id=7, rating=1,
                        make_global=True)
    assert a["scope"] == "user" and a["recorded"] is True

    with Store.open(svc.config.paths.db_path) as store:
        # bob sees only global feedback; alice sees her private + global
        bob = store.feedback_entries(author="bob")
        alice = store.feedback_entries(author="alice")
        assert {r["doc_id"] for r in bob} == {7}                # only the global one
        assert {r["doc_id"] for r in alice} == {5, 7}           # her private + the global
        # ranking signal: alice's view nets doc 5 down, doc 7 up; bob's view only doc 7 up
        assert store.feedback_scores(author="alice") == {5: -1, 7: 1}
        assert store.feedback_scores(author="bob") == {7: 1}


def test_feedback_scores_aggregate_and_cleanup(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    for r in (1, 1, -1):  # net +1 on doc 3, all global
        svc.submit_feedback(user="carol", text="note", doc_id=3, rating=r, make_global=True)
    with Store.open(svc.config.paths.db_path) as store:
        assert store.feedback_scores() == {3: 1} and store.count_feedback() == 3
        store.delete_document(3)  # deleting the target removes its feedback
        assert store.count_feedback() == 0


def test_cross_tier_discrepancy_annotation() -> None:
    """A conflict pair spanning internal vs vendor is flagged cross-tier, internal is authoritative."""
    from types import SimpleNamespace

    from docusearch.server import _annotate_conflict_tiers

    meta = {1: ("vendor/a.md", "vendor"), 2: ("internal/b.md", "internal"), 3: ("vendor/c.md", "vendor")}
    tier_of = {"vendor": "vendor", "internal": "internal"}
    conflicts = [
        SimpleNamespace(chunk_a=10, chunk_b=20, doc_a=1, doc_b=2, similarity=0.95),  # cross-tier
        SimpleNamespace(chunk_a=30, chunk_b=40, doc_a=1, doc_b=3, similarity=0.95),  # both vendor
    ]
    rows, cross = _annotate_conflict_tiers(conflicts, meta, tier_of)
    assert cross == 1
    a, b = rows
    assert a["cross_tier"] is True and a["authoritative_doc"] == 2  # internal wins over vendor
    assert a["doc_a_tier"] == "vendor" and a["doc_b_tier"] == "internal"
    assert b["cross_tier"] is False and b["authoritative_doc"] is None
