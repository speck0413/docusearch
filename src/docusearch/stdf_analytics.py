"""STDF single-log analytics (Phase 6 / GATE 6): the data functions behind the agent's `stdf_*`
tools — audit (two-file compare), site-to-site, trend across runs — built on the general
:mod:`docusearch.analytics` plot/stats engine. Numeric results come from :func:`stdf.parse_stdf_tests`.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from html import escape

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


# ------------------------------------------------------- drill-down HTML report builders


def _test_txt(run: StdfRun, test_num: int) -> str:
    return next((t.test_txt for t in run.tests if t.test_num == test_num), f"test {test_num}")


def _stats_line(s: dict[str, float]) -> str:
    if not s.get("n"):
        return "no data"
    return (
        f"n={int(s['n'])} · mean={s['mean']:.4g} · median={s['median']:.4g} · "
        f"std={s['std']:.3g} · min={s['min']:.4g} · max={s['max']:.4g}"
    )


def plot_test_html(
    run: StdfRun, test_num: int, *, kind: str = "histogram", backend: str = "matplotlib"
) -> str:
    """A single test's distribution plot + summary stats."""
    vals = results_for(run, test_num)
    txt = _test_txt(run, test_num)
    plot = analytics.render_plot(
        kind, y=vals, title=f"{txt} (test {test_num})", xlabel=txt, ylabel="count", backend=backend
    )
    return (
        f'<section class="stdf-plot"><h3>{escape(txt)} — test {test_num}</h3>{plot}'
        f"<p class='stats'>{_stats_line(analytics.summary_stats(vals))}</p></section>"
    )


def site_compare_html(run: StdfRun, test_num: int, *, backend: str = "matplotlib") -> str:
    """Site-to-site box comparison of one test."""
    groups = site_groups(run, test_num)
    series = [(f"site {site}", vals) for site, vals in groups.items()]
    txt = _test_txt(run, test_num)
    plot = analytics.render_plot(
        "whisker", series=series, title=f"{txt} by site", ylabel=txt, backend=backend
    )
    rows = "".join(
        f"<li>site {site}: {_stats_line(analytics.summary_stats(v))}</li>"
        for site, v in groups.items()
    )
    return f'<section class="stdf-sites"><h3>{escape(txt)} — site-to-site</h3>{plot}<ul>{rows}</ul></section>'


def trend_html(
    runs: list[tuple[str, StdfRun]], test_num: int, *, stat: str = "mean", backend: str = "matplotlib"
) -> str:
    """Long-run trend of a test's ``stat`` across ordered runs."""
    pts = trend_points(runs, test_num, stat)
    txt = runs[0][1] and _test_txt(runs[0][1], test_num)
    plot = analytics.render_plot(
        "linear", x=list(range(len(pts))), y=[p[1] for p in pts],
        title=f"{txt} {stat} trend", xlabel="run", ylabel=f"{txt} {stat}", backend=backend,
    )
    rows = "".join(f"<li>{escape(label)}: {value:.4g}</li>" for label, value in pts)
    return f'<section class="stdf-trend"><h3>{escape(str(txt))} — {stat} trend</h3>{plot}<ul>{rows}</ul></section>'


def audit_report_html(
    run_a: StdfRun, run_b: StdfRun, *, backend: str = "matplotlib",
    label_a: str = "A", label_b: str = "B",
) -> str:
    """A **drill-down** audit report: top-level summary (yields, added/removed tests, condition
    diff) → collapsible per-test Q-Q + stat deltas, so the user clicks from the yield delta down to
    the individual test (R-STDF-2)."""
    rep = audit_runs(run_a, run_b)
    ya_p, ya_t = rep.yield_a
    yb_p, yb_t = rep.yield_b
    ya = 100 * ya_p / ya_t if ya_t else 0.0
    yb = 100 * yb_p / yb_t if yb_t else 0.0
    parts = [
        '<section class="stdf-audit"><h2>STDF audit</h2>',
        f"<p><strong>Yield</strong>: {label_a} {ya:.1f}% ({ya_p}/{ya_t}) → "
        f"{label_b} {yb:.1f}% ({yb_p}/{yb_t}) &nbsp; <strong>Δ {yb - ya:+.1f}%</strong></p>",
        f"<p><strong>Tests</strong>: {len(rep.matched)} matched · "
        f"{len(rep.added)} added ({rep.added}) · {len(rep.removed)} removed ({rep.removed})</p>",
        f"<p><strong>Conditions only in {label_a}</strong>: {rep.conditions_only_a or '—'}<br>"
        f"<strong>Conditions only in {label_b}</strong>: {rep.conditions_only_b or '—'}</p>",
        "<h3>Per-test (drill down)</h3>",
    ]
    for d in rep.matched:
        qq = analytics.render_plot(
            "qq",
            series=[(label_a, results_for(run_a, d.test_num)), (label_b, results_for(run_b, d.test_num))],
            title=f"{d.test_txt} Q-Q ({label_a} vs {label_b})", xlabel=label_a, ylabel=label_b,
            backend=backend,
        )
        flag = " ⚠️" if d.mean_delta is not None and abs(d.mean_delta) > 0 else ""
        dmean = f"{d.mean_delta:+.4g}" if d.mean_delta is not None else "n/a"
        parts.append(
            f"<details><summary>test {d.test_num} — {escape(d.test_txt)} "
            f"(Δmean {dmean}){flag}</summary>"
            f"<p>{label_a}: {_stats_line(d.a)}<br>{label_b}: {_stats_line(d.b)}</p>{qq}</details>"
        )
    parts.append("</section>")
    return "".join(parts)
