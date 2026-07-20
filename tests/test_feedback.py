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


def test_feedback_scores_one_vote_per_account_and_cleanup(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    # a single account cannot STACK votes — its latest rating counts once (red-team #H2). carol
    # votes +1, +1, -1 on doc 3 → she nets -1 (her latest), not +1 (the old, exploitable sum).
    for r in (1, 1, -1):
        svc.submit_feedback(user="carol", text="note", doc_id=3, rating=r, make_global=True)
    # a DIFFERENT account still contributes independently
    svc.submit_feedback(user="dave", text="agree", doc_id=3, rating=1, make_global=True)
    with Store.open(svc.config.paths.db_path) as store:
        assert store.feedback_scores() == {3: 0}       # carol -1 + dave +1
        assert store.count_feedback() == 4             # all 4 rows still stored (dedup is at score time)
        store.delete_document(3)                        # deleting the target removes its feedback
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


def test_phase8_redteam_regressions(tmp_path: Path) -> None:
    # H2: an out-of-range rating is clamped to -1/0/+1, and one account counts at most once per doc
    svc = _service(tmp_path)
    svc.submit_feedback(user="a", text="x", doc_id=1, rating=-10_000)          # clamped
    for _ in range(9):
        svc.submit_feedback(user="a", text="x", doc_id=1, rating=-1)           # spam
    with Store.open(svc.config.paths.db_path) as store:
        assert store.feedback_scores(author="a") == {1: -1}                    # bounded, not -19
        # L2: a malformed rating row (bypassing add_feedback) must not crash ranking, just be skipped
        store._conn.execute(  # noqa: SLF001
            "INSERT INTO feedback(ts, author, scope, doc_id, chunk_id, rating, text) "
            "VALUES ('t','b','global',1,NULL,'oops','bad')")
        assert store.feedback_scores(author="a") == {1: -1}                    # no crash, bad row ignored

    # M1: an absurd doc_id is a clean ValueError, not an uncaught OverflowError
    import pytest
    with pytest.raises(ValueError, match="out of range"):
        svc.submit_feedback(user="a", text="x", doc_id=2**100, rating=1)


def test_feedback_write_gated_on_private_store(tmp_path: Path) -> None:
    # H1: a caller who may not read a private store may not write feedback into it
    src = tmp_path / "seed"
    src.mkdir()
    path = tmp_path / "d.yaml"
    path.write_text(
        f'paths:\n  staging_dir: "{(tmp_path / "s").as_posix()}"\n'
        f'  db_path: "{(tmp_path / "c.db").as_posix()}"\n  tmp_dir: "{(tmp_path / "t").as_posix()}"\n'
        'access:\n  visibility: "private"\n  allowed_users: ["owner"]\n'
        f'sources:\n  - name: seed\n    location: "{src.as_posix()}"\n    min_content_chars: 1\n'
        'embed:\n  model: "none"\n', encoding="utf-8")
    svc = Service(cfg.load(path))
    import pytest
    with pytest.raises(PermissionError):
        svc.submit_feedback(user="intruder", text="x", doc_id=1, rating=1)
    assert svc.submit_feedback(user="owner", text="ok", doc_id=1, rating=1)["recorded"] is True


def test_tier_typo_is_a_config_error(tmp_path: Path) -> None:
    # M2: a mistyped tier must fail config load (closed enum), not silently score as vendor
    import pytest
    src = tmp_path / "seed"
    src.mkdir()
    path = tmp_path / "d.yaml"
    path.write_text(
        f'paths:\n  staging_dir: "{(tmp_path / "s").as_posix()}"\n'
        f'  db_path: "{(tmp_path / "c.db").as_posix()}"\n  tmp_dir: "{(tmp_path / "t").as_posix()}"\n'
        f'sources:\n  - name: seed\n    location: "{src.as_posix()}"\n    min_content_chars: 1\n'
        '    tier: "gold"\nembed:\n  model: "none"\n', encoding="utf-8")
    with pytest.raises(cfg.ConfigError):
        cfg.load(path)
