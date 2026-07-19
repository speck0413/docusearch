"""STDF single-log analytics (Phase 6 / GATE 6): the data functions behind the agent's `stdf_*`
tools — audit (two-file compare), site-to-site, trend across runs — built on the general
:mod:`docusearch.analytics` plot/stats engine. Numeric results come from :func:`stdf.parse_stdf_tests`.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from . import analytics
from .stdf import StdfRun, StdfTest


def tests_by_num(run: StdfRun) -> dict[int, list[StdfTest]]:
    out: dict[int, list[StdfTest]] = defaultdict(list)
    for t in run.tests:
        out[t.test_num].append(t)
    return dict(out)


def results_for(run: StdfRun, test_num: int) -> list[float]:
    return [t.result for t in run.tests if t.test_num == test_num and t.result is not None]


def part_yield(run: StdfRun) -> tuple[int, int]:
    """(passing parts, total parts) — a part passes when all of its tests pass."""
    parts: dict[str, bool] = {}
    for t in run.tests:
        parts.setdefault(t.part_id, True)
        if not t.passed:
            parts[t.part_id] = False
    return sum(1 for ok in parts.values() if ok), len(parts)


def site_groups(run: StdfRun, test_num: int) -> dict[int, list[float]]:
    """A test's results grouped by site — the input to a site-to-site box/Q-Q comparison."""
    groups: dict[int, list[float]] = defaultdict(list)
    for t in run.tests:
        if t.test_num == test_num and t.result is not None:
            groups[t.site].append(t.result)
    return dict(sorted(groups.items()))


def trend_points(runs: list[tuple[str, StdfRun]], test_num: int, stat: str = "mean") -> list[tuple[str, float]]:
    """A test's ``stat`` (mean/median/std/min/max) across an ordered list of ``(label, run)`` — the
    long-run drift series. Runs with no data for the test are skipped."""
    points: list[tuple[str, float]] = []
    for label, run in runs:
        s = analytics.summary_stats(results_for(run, test_num))
        if s.get("n", 0):
            points.append((label, s[stat]))
    return points


@dataclass
class TestDelta:
    test_num: int
    test_txt: str
    a: dict[str, float]
    b: dict[str, float]
    mean_delta: float | None


@dataclass
class AuditReport:
    matched: list[TestDelta] = field(default_factory=list)
    added: list[int] = field(default_factory=list)  # test_nums only in B
    removed: list[int] = field(default_factory=list)  # test_nums only in A
    yield_a: tuple[int, int] = (0, 0)
    yield_b: tuple[int, int] = (0, 0)
    conditions_only_a: list[str] = field(default_factory=list)
    conditions_only_b: list[str] = field(default_factory=list)


def _condition_set(run: StdfRun) -> set[str]:
    seen: set[str] = set()
    for t in run.tests:
        for k, v in t.conditions.items():
            seen.add(f"{k}={v}")
    return seen


def audit_runs(run_a: StdfRun, run_b: StdfRun) -> AuditReport:
    """Compare two STDF runs (R-STDF-2): **test alignment** (matched / added / removed by test
    number), **yield** for each, **condition diff**, and per-matched-test distribution stats +
    mean delta — the top level of a drill-down audit the agent renders."""
    a_by, b_by = tests_by_num(run_a), tests_by_num(run_b)
    rep = AuditReport(
        added=sorted(set(b_by) - set(a_by)),
        removed=sorted(set(a_by) - set(b_by)),
        yield_a=part_yield(run_a),
        yield_b=part_yield(run_b),
    )
    ca, cb = _condition_set(run_a), _condition_set(run_b)
    rep.conditions_only_a = sorted(ca - cb)
    rep.conditions_only_b = sorted(cb - ca)
    for num in sorted(set(a_by) & set(b_by)):
        sa = analytics.summary_stats(results_for(run_a, num))
        sb = analytics.summary_stats(results_for(run_b, num))
        delta = (sb["mean"] - sa["mean"]) if (sa.get("n") and sb.get("n")) else None
        rep.matched.append(
            TestDelta(num, a_by[num][0].test_txt or b_by[num][0].test_txt, sa, sb, delta)
        )
    return rep
