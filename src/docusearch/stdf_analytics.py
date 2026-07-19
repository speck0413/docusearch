"""STDF single-log analytics (Phase 6 / GATE 6): the data functions behind the agent's `stdf_*`
tools — audit (two-file compare), site-to-site, trend across runs — built on the general
:mod:`docusearch.analytics` plot/stats engine. Numeric results come from :func:`stdf.parse_stdf_tests`.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from html import escape

from . import analytics
from .stdf import StdfPart, StdfRun, StdfTest

DEFAULT_PART_KEY = ("lot", "sublot", "wafer", "x", "y")


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


# ------------------------------------------------------- part traceability + insertion yield


def parse_part_key(spec: str) -> tuple[str, ...]:
    """Turn a config ``part_key`` string (``"lot,sublot,wafer,x,y"``) into field tuple."""
    fields = tuple(f.strip() for f in spec.split(",") if f.strip())
    return fields or DEFAULT_PART_KEY


def insertion_yield(
    parts: Sequence[StdfPart], *, part_key: Sequence[str] = DEFAULT_PART_KEY
) -> list[dict[str, object]]:
    """First-pass and final yield **per insertion** (R-STDF-2). Within an insertion, a part touched
    down more than once (same key → multiple PRRs) counts once: **first-pass** uses its first
    touchdown, **final** uses its last (post-retest). Insertions keep first-seen order (WS1, WS1-RT,
    WS2, FT …)."""
    order: list[str] = []
    grouped: dict[str, dict[tuple[str, ...], list[StdfPart]]] = {}
    for p in parts:
        if p.insertion not in grouped:
            grouped[p.insertion] = {}
            order.append(p.insertion)
        grouped[p.insertion].setdefault(p.key(part_key), []).append(p)
    out: list[dict[str, object]] = []
    for ins in order:
        keymap = grouped[ins]
        total = len(keymap)
        first_pass = sum(1 for touches in keymap.values() if touches[0].passed)
        final_pass = sum(1 for touches in keymap.values() if touches[-1].passed)
        retested = sum(1 for touches in keymap.values() if len(touches) > 1)
        out.append({
            "insertion": ins, "total": total, "first_pass": first_pass, "final_pass": final_pass,
            "retested": retested,
            "first_pass_yield": 100.0 * first_pass / total if total else 0.0,
            "final_yield": 100.0 * final_pass / total if total else 0.0,
        })
    return out


def trace_parts(
    parts: Sequence[StdfPart], *, part_key: Sequence[str] = DEFAULT_PART_KEY
) -> tuple[list[str], dict[tuple[str, ...], dict[str, StdfPart]]]:
    """Every part's **final** touchdown at each insertion, for progression tracing initial → final
    (R-STDF-2). Returns ``(insertion_order, {part_key: {insertion: StdfPart}})``."""
    order: list[str] = []
    seen: set[str] = set()
    trace: dict[tuple[str, ...], dict[str, StdfPart]] = {}
    for p in parts:
        if p.insertion not in seen:
            seen.add(p.insertion)
            order.append(p.insertion)
        trace.setdefault(p.key(part_key), {})[p.insertion] = p  # last per insertion = final touchdown
    return order, trace


def insertion_yield_html(
    parts: Sequence[StdfPart], *, part_key: Sequence[str] = DEFAULT_PART_KEY
) -> str:
    """A yield-per-insertion table: first-pass vs final yield (WS1, WS1-RT, …)."""
    rows = insertion_yield(parts, part_key=part_key)
    body = "".join(
        f"<tr><td>{escape(str(r['insertion']))}</td><td>{r['total']}</td>"
        f"<td>{r['first_pass_yield']:.1f}% ({r['first_pass']}/{r['total']})</td>"
        f"<td>{r['final_yield']:.1f}% ({r['final_pass']}/{r['total']})</td>"
        f"<td>{r['retested']}</td></tr>"
        for r in rows
    )
    return (
        '<section class="stdf-yield"><h3>Yield per insertion</h3><div class="scroll">'
        '<table class="grid"><thead><tr><th>insertion</th><th>parts</th><th>first-pass yield</th>'
        f"<th>final yield</th><th>retested</th></tr></thead><tbody>{body}</tbody></table></div></section>"
    )


def part_trace_html(
    parts: Sequence[StdfPart], *, part_key: Sequence[str] = DEFAULT_PART_KEY, limit: int = 100
) -> str:
    """A part-progression table: each part (rows) × insertion (columns) → PASS/FAIL·bin, so you can
    follow a die from initial to final touchdown."""
    order, trace = trace_parts(parts, part_key=part_key)
    head = "".join(f"<th>{escape(i)}</th>" for i in order)
    body = []
    for key in list(trace)[:limit]:
        cells = []
        for ins in order:
            p = trace[key].get(ins)
            cells.append(f"<td>{'PASS' if p.passed else 'FAIL'} · b{p.hard_bin}</td>" if p else "<td>—</td>")
        body.append(f"<tr><td>{escape('/'.join(key))}</td>{''.join(cells)}</tr>")
    return (
        f'<section class="stdf-trace"><h3>Part progression ({escape("/".join(part_key))})</h3>'
        f'<div class="scroll"><table class="grid"><thead><tr><th>part</th>{head}</tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table></div></section>'
    )


# ------------------------------------------------------- themed HTML report builders


def _test_txt(run: StdfRun, test_num: int) -> str:
    return next((t.test_txt for t in run.tests if t.test_num == test_num), f"test {test_num}")


def _stats_line(s: dict[str, float]) -> str:
    if not s.get("n"):
        return "no data"
    return (
        f"n={int(s['n'])} · mean={s['mean']:.4g} · median={s['median']:.4g} · "
        f"std={s['std']:.3g} · min={s['min']:.4g} · max={s['max']:.4g}"
    )


def _page(title: str, body: str, *, subtitle: str = "") -> str:
    from . import report  # lazy: the themed-page wrapper (shared with the cited reports)

    return report.themed_page(title, body, subtitle=subtitle, eyebrow="docusearch · STDF analytics")


def plot_test_html(
    run: StdfRun, test_num: int, *, kind: str = "histogram", backend: str = "matplotlib"
) -> str:
    """A single test's distribution plot + summary stats, in the shared theme."""
    vals = results_for(run, test_num)
    txt = _test_txt(run, test_num)
    plot = analytics.render_plot(
        kind, y=vals, title=f"{txt} (test {test_num})", xlabel=txt, ylabel="count", backend=backend
    )
    card = (
        f'<section class="acard"><h2>{escape(txt)} — test {test_num}</h2>{plot}'
        f"<p class='stats'>{_stats_line(analytics.summary_stats(vals))}</p></section>"
    )
    return _page(f"{txt} — distribution", card, subtitle=f"test {test_num} · {kind}")


def site_compare_html(run: StdfRun, test_num: int, *, backend: str = "matplotlib") -> str:
    """Site-to-site box comparison of one test, themed."""
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
    card = f'<section class="acard"><h2>{escape(txt)} — site-to-site</h2>{plot}<ul>{rows}</ul></section>'
    return _page(f"{txt} — site-to-site", card)


def trend_html(
    runs: list[tuple[str, StdfRun]], test_num: int, *, stat: str = "mean", backend: str = "matplotlib"
) -> str:
    """Long-run trend of a test's ``stat`` across ordered runs, themed."""
    pts = trend_points(runs, test_num, stat)
    txt = runs[0][1] and _test_txt(runs[0][1], test_num)
    plot = analytics.render_plot(
        "linear", x=list(range(len(pts))), y=[p[1] for p in pts],
        title=f"{txt} {stat} trend", xlabel="run", ylabel=f"{txt} {stat}", backend=backend,
    )
    rows = "".join(f"<li>{escape(label)}: {value:.4g}</li>" for label, value in pts)
    card = f'<section class="acard"><h2>{escape(str(txt))} — {stat} trend</h2>{plot}<ul>{rows}</ul></section>'
    return _page(f"{txt} — {stat} trend", card)


# ---- Beyond-Compare-style test diff (revision to revision) ---------------------


@dataclass
class TestDef:
    """A test's definition in one run — the fields the diff compares."""

    test_num: int
    test_txt: str
    lo: float | None
    hi: float | None
    units: str
    conditions: dict[str, str]


@dataclass
class DiffRow:
    name: str
    status: str  # added | removed | changed | identical
    a: TestDef | None
    b: TestDef | None
    changed: set[str] = field(default_factory=set)  # {"tnum","lo","hi","cond:<key>"}


def _defs(run: StdfRun) -> dict[str, TestDef]:
    out: dict[str, TestDef] = {}
    for t in run.tests:
        if t.test_txt not in out:  # a test's definition is constant across its touchdowns
            out[t.test_txt] = TestDef(
                t.test_num, t.test_txt, t.lo_limit, t.hi_limit, t.units, dict(t.conditions)
            )
    return out


def diff_tests(run_a: StdfRun, run_b: StdfRun) -> tuple[list[str], list[DiffRow]]:
    """Align tests **by name + conditions** (the unique id) and flag exactly what changed revision to
    revision — test number, limits (LLM/HLM), or conditions (R-STDF-2). Returns the union of
    condition keys (for columns) and one :class:`DiffRow` per test."""
    da, db = _defs(run_a), _defs(run_b)
    cond_keys = sorted({k for d in (*da.values(), *db.values()) for k in d.conditions})
    rows: list[DiffRow] = []
    for name in sorted(set(da) | set(db)):
        a, b = da.get(name), db.get(name)
        if a and not b:
            rows.append(DiffRow(name, "removed", a, None))
        elif b and not a:
            rows.append(DiffRow(name, "added", None, b))
        else:
            assert a is not None and b is not None
            changed: set[str] = set()
            if a.test_num != b.test_num:
                changed.add("tnum")
            if a.lo != b.lo:
                changed.add("lo")
            if a.hi != b.hi:
                changed.add("hi")
            for k in cond_keys:
                if a.conditions.get(k) != b.conditions.get(k):
                    changed.add(f"cond:{k}")
            rows.append(DiffRow(name, "changed" if changed else "identical", a, b, changed))
    return cond_keys, rows


def _num(v: float | None) -> str:
    return "—" if v is None else f"{v:g}"


def _diff_table_html(run_a: StdfRun, run_b: StdfRun, label_a: str, label_b: str) -> str:
    cond_keys, rows = diff_tests(run_a, run_b)
    heads = ["Status", f"old #<br>{escape(label_a)}", f"new #<br>{escape(label_b)}", "Test",
             "old LLM", "new LLM", "old HLM", "new HLM", "Units"]
    for k in cond_keys:
        heads += [f"old {escape(k)}", f"new {escape(k)}"]
    thead = "".join(f"<th>{h}</th>" for h in heads)

    def cell(val: str, field_name: str, changed: set[str]) -> str:
        cls = ' class="chg"' if field_name in changed else ""
        return f"<td{cls}>{escape(val)}</td>"

    body_rows = []
    for r in rows:
        a, b, ch = r.a, r.b, r.changed
        cells = [f'<td><span class="badge {r.status}">{r.status}</span></td>']
        cells.append(cell(str(a.test_num) if a else "—", "tnum", ch))
        cells.append(cell(str(b.test_num) if b else "—", "tnum", ch))
        cells.append(f"<td>{escape(r.name)}</td>")
        cells.append(cell(_num(a.lo) if a else "—", "lo", ch))
        cells.append(cell(_num(b.lo) if b else "—", "lo", ch))
        cells.append(cell(_num(a.hi) if a else "—", "hi", ch))
        cells.append(cell(_num(b.hi) if b else "—", "hi", ch))
        cells.append(f"<td>{escape((a or b).units)}</td>")  # type: ignore[union-attr]
        for k in cond_keys:
            fk = f"cond:{k}"
            cells.append(cell(a.conditions.get(k, "—") if a else "—", fk, ch))
            cells.append(cell(b.conditions.get(k, "—") if b else "—", fk, ch))
        body_rows.append(f'<tr class="{r.status}">{"".join(cells)}</tr>')
    n = {s: sum(1 for r in rows if r.status == s) for s in ("added", "removed", "changed", "identical")}
    legend = (
        f'<p class="stats">{n["changed"]} changed · {n["added"]} added · {n["removed"]} removed · '
        f'{n["identical"]} identical &nbsp;—&nbsp; changed cells highlighted; every field shown old vs new.</p>'
    )
    return (
        '<section class="acard"><h2>Test diff — revision to revision</h2>'
        f"{legend}<div class='scroll'><table class='grid'><thead><tr>{thead}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody></table></div></section>"
    )


def audit_report_html(
    run_a: StdfRun, run_b: StdfRun, *, backend: str = "matplotlib",
    label_a: str = "A", label_b: str = "B",
) -> str:
    """A themed audit: a summary card (yield + counts), the **Beyond-Compare test-diff table**
    (every field old vs new, changes highlighted — a table view, separate from plots), and a
    yield-per-insertion table (R-STDF-2)."""
    rep = audit_runs(run_a, run_b)
    ya_p, ya_t = rep.yield_a
    yb_p, yb_t = rep.yield_b
    ya = 100 * ya_p / ya_t if ya_t else 0.0
    yb = 100 * yb_p / yb_t if yb_t else 0.0
    summary = (
        '<section class="acard"><h2>Summary</h2>'
        f"<p><strong>Yield</strong>: {escape(label_a)} {ya:.1f}% ({ya_p}/{ya_t}) → "
        f"{escape(label_b)} {yb:.1f}% ({yb_p}/{yb_t}) &nbsp;·&nbsp; <strong>Δ {yb - ya:+.1f}%</strong></p>"
        f"<p><strong>Tests</strong>: {len(rep.matched)} matched · {len(rep.added)} added · "
        f"{len(rep.removed)} removed</p>"
        f"<p><strong>Conditions only in {escape(label_a)}</strong>: {escape(str(rep.conditions_only_a or '—'))}<br>"
        f"<strong>Conditions only in {escape(label_b)}</strong>: {escape(str(rep.conditions_only_b or '—'))}</p>"
        "</section>"
    )
    diff = _diff_table_html(run_a, run_b, label_a, label_b)
    yld = ""
    if run_a.parts or run_b.parts:
        yld = insertion_yield_html([*run_a.parts, *run_b.parts]).replace(
            '<section class="stdf-yield">', '<section class="acard">'
        )
    return _page(
        f"STDF audit — {label_a} vs {label_b}", summary + diff + yld,
        subtitle="test-by-test diff of test number, limits, and conditions",
    )
