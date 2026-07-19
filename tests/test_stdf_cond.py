"""STDF DTR condition engine (R-STDF-1): COND:/COND_OFF datalog rules → sticky conditions on
following tests. Pure state-machine tests (no STDF file needed)."""

from __future__ import annotations

from docusearch import stdf


def test_cond_sets_and_trims_pairs() -> None:
    t = stdf.ConditionTracker()
    t.apply("COND:  corner = slow ,temp= 125C ")  # messy spacing on purpose
    assert t.snapshot() == {"corner": "slow", "temp": "125C"}  # keys + values trimmed


def test_cond_change_updates_only_that_key() -> None:
    t = stdf.ConditionTracker()
    t.apply("COND: corner=slow, temp=125C")
    t.apply("COND: corner=fast")  # re-set one key
    assert t.snapshot() == {"corner": "fast", "temp": "125C"}


def test_cond_off_per_key() -> None:
    t = stdf.ConditionTracker()
    t.apply("COND: corner=slow, temp=125C")
    t.apply("COND_OFF: corner")
    assert t.snapshot() == {"temp": "125C"}  # only corner cleared


def test_cond_off_clear_all_two_spellings() -> None:
    for clear_stmt in ("COND_OFF", "COND_OFF: *"):
        t = stdf.ConditionTracker()
        t.apply("COND: corner=slow, temp=125C")
        t.apply(clear_stmt)
        assert t.snapshot() == {}, clear_stmt


def test_non_matching_dtr_is_ignored() -> None:
    t = stdf.ConditionTracker()
    t.apply("COND: corner=slow")
    t.apply("just some datalog text, not a condition")
    t.apply("COND_OFF")  # clears
    assert t.snapshot() == {}
    t.apply("COND: x=1")
    t.apply("COND_OFFICER: note")  # must NOT be read as COND_OFF
    assert t.snapshot() == {"x": "1"}


def test_reset_clears_for_per_part_scope() -> None:
    t = stdf.ConditionTracker()
    t.apply("COND: corner=slow")
    t.reset()  # part boundary when scope=part
    assert t.snapshot() == {}


def test_snapshot_is_a_copy() -> None:
    t = stdf.ConditionTracker()
    t.apply("COND: a=1")
    snap = t.snapshot()
    t.apply("COND: b=2")
    assert snap == {"a": "1"}  # earlier snapshot not mutated by later applies
