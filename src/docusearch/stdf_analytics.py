"""STDF single-log analytics (Phase 6 / GATE 6): the data functions behind the agent's `stdf_*`
tools — audit (two-file compare), site-to-site, trend across runs — built on the general
:mod:`docusearch.analytics` plot/stats engine. Numeric results come from :func:`stdf.parse_stdf_tests`.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from html import escape
from typing import Any

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


def _diff_table_interactive(rows: list[DiffRow], cond_keys: list[str], label_a: str, label_b: str) -> str:
    heads = ["Status", f"old #<br>{escape(label_a)}", f"new #<br>{escape(label_b)}", "Test",
             "old LLM", "new LLM", "old HLM", "new HLM", "Units"]
    for k in cond_keys:
        heads += [f"old {escape(k)}", f"new {escape(k)}"]
    heads.append("Feedback")
    thead = "".join(f'<th class="sortable">{h}</th>' for h in heads)

    def cell(val: str, field_name: str, changed: set[str], *, num: bool = False) -> str:
        cls = " chg" if field_name in changed else ""
        dv = f' data-v="{escape(val)}"' if num and val not in ("—", "") else ""
        return f'<td class="{cls.strip()}"{dv}>{escape(val)}</td>'

    body = []
    for r in rows:
        a, b, ch = r.a, r.b, r.changed
        units = escape((a or b).units) if (a or b) else ""  # type: ignore[union-attr]
        cells = [
            f'<td><span class="badge {r.status}">{r.status}</span></td>',
            cell(str(a.test_num) if a else "—", "tnum", ch, num=True),
            cell(str(b.test_num) if b else "—", "tnum", ch, num=True),
            f"<td>{escape(r.name)}</td>",
            cell(_num(a.lo) if a else "—", "lo", ch, num=True),
            cell(_num(b.lo) if b else "—", "lo", ch, num=True),
            cell(_num(a.hi) if a else "—", "hi", ch, num=True),
            cell(_num(b.hi) if b else "—", "hi", ch, num=True),
            f"<td>{units}</td>",
        ]
        for k in cond_keys:
            fk = f"cond:{k}"
            cells.append(cell(a.conditions.get(k, "—") if a else "—", fk, ch))
            cells.append(cell(b.conditions.get(k, "—") if b else "—", fk, ch))
        cells.append('<td class="fb" contenteditable="true"></td>')  # editable feedback
        body.append(
            f'<tr data-name="{escape(r.name)}" data-status="{r.status}" '
            f'data-flags="{r.status}">{"".join(cells)}</tr>'
        )
    n = {s: sum(1 for r in rows if r.status == s) for s in ("added", "removed", "changed", "identical")}
    toolbar = (
        '<div class="toolbar">'
        '<input type="text" class="rowfilter" placeholder="filter tests…">'
        '<button class="dl-fb">⬇ Download feedback</button>' + _EXPAND_BTN +
        f'<span class="hint">click a header to sort · type to filter · click a Feedback cell to '
        f'edit · {n["changed"]} changed · {n["added"]} added · {n["removed"]} removed · '
        f'{n["identical"]} identical</span></div>'
    )
    present = {r.status for r in rows}
    return (
        '<h2>Test diff — revision to revision</h2><div class="tablepanel">'
        + _chipbar(present, "diff") + toolbar
        + '<div class="tablewrap"><table class="grid"><thead><tr>' + thead
        + f"</tr></thead><tbody>{''.join(body)}</tbody></table></div></div>"
    )


def _cap_fmt(v: float | None) -> str:
    return "—" if v is None else f"{v:.3g}"


def _capability_row(label: str, vals: list[float], lo: float | None, hi: float | None) -> str:
    s = analytics.summary_stats(vals)
    if not s.get("n"):
        return f"<tr><td>{escape(label)}</td><td colspan='9'>no data</td></tr>"
    cap = analytics.capability(vals, lo, hi)
    fmt = _cap_fmt
    return (
        f"<tr><td>{escape(label)}</td><td>{int(s['n'])}</td><td>{s['mean']:.4g}</td>"
        f"<td>{s['median']:.4g}</td><td>{s['std']:.3g}</td><td>{s['min']:.4g}</td>"
        f"<td>{s['max']:.4g}</td><td>{fmt(cap['cpl'])}</td><td>{fmt(cap['cpu'])}</td>"
        f"<td>{fmt(cap['cpk'])}</td></tr>"
    )


_DASHBOARD_JS = """
(function(){
 function tab(name){
  document.querySelectorAll('.tab').forEach(function(x){x.classList.toggle('active',x.dataset.t===name);});
  document.querySelectorAll('.panel').forEach(function(x){x.classList.toggle('hidden',x.dataset.p!==name);});
 }
 document.querySelectorAll('.tab').forEach(function(t){t.addEventListener('click',function(){tab(t.dataset.t);});});
 // sortable tables
 document.querySelectorAll('table.grid th.sortable').forEach(function(th){
  th.addEventListener('click',function(){
   var tb=th.closest('table').querySelector('tbody');var rows=[].slice.call(tb.querySelectorAll('tr'));
   var col=th.cellIndex;var asc=th.dataset.asc!=='1';th.dataset.asc=asc?'1':'0';
   rows.sort(function(a,b){
    var ca=a.cells[col],cb=b.cells[col];
    var x=ca.dataset.v!=null?ca.dataset.v:ca.innerText,y=cb.dataset.v!=null?cb.dataset.v:cb.innerText;
    var nx=parseFloat(x),ny=parseFloat(y);
    if(!isNaN(nx)&&!isNaN(ny))return asc?nx-ny:ny-nx;
    return asc?String(x).localeCompare(y):String(y).localeCompare(x);
   });
   rows.forEach(function(r){tb.appendChild(r);});
  });
 });
 // per-panel filtering: three-state chips (include/exclude) combined under Any/All, + a text box
 document.querySelectorAll('.panel').forEach(function(panel){
  var bar=panel.querySelector('.chipbar');var textInput=panel.querySelector('.rowfilter');
  if(!bar&&!textInput)return;
  function has(f,tok){return f.indexOf(' '+tok+' ')>=0;}
  function apply(){
   var inc=[],exc=[],useAnd=false;
   if(bar){
    bar.querySelectorAll('.chip').forEach(function(c){
     if(c.classList.contains('inc'))inc.push(c.dataset.flag);
     else if(c.classList.contains('exc'))exc.push(c.dataset.flag);});
    var m=bar.querySelector('.chipmode');useAnd=!!m&&m.dataset.mode==='and';
   }
   var q=textInput?textInput.value.toLowerCase():'';
   panel.querySelectorAll('[data-flags]').forEach(function(el){
    var f=' '+el.dataset.flags+' ';
    var passExc=exc.every(function(x){return !has(f,x);});
    var passInc=inc.length===0||(useAnd?inc.every(function(x){return has(f,x);})
                                       :inc.some(function(x){return has(f,x);}));
    var passText=!q||el.innerText.toLowerCase().indexOf(q)>=0;
    el.style.display=(passExc&&passInc&&passText)?'':'none';
   });
  }
  if(bar){
   bar.querySelectorAll('.chip').forEach(function(c){c.addEventListener('click',function(){
    if(c.classList.contains('inc')){c.classList.remove('inc');c.classList.add('exc');}
    else if(c.classList.contains('exc')){c.classList.remove('exc');}
    else{c.classList.add('inc');}apply();});});
   var mode=bar.querySelector('.chipmode');
   if(mode)mode.addEventListener('click',function(){
    var to=mode.dataset.mode==='and'?'or':'and';mode.dataset.mode=to;
    mode.textContent='Match: '+(to==='and'?'All':'Any');apply();});
   var clr=bar.querySelector('.chipclear');
   if(clr)clr.addEventListener('click',function(){
    bar.querySelectorAll('.chip').forEach(function(x){x.classList.remove('inc','exc');});apply();});
  }
  if(textInput)textInput.addEventListener('input',apply);
 });
 // download the editable feedback in whichever table the button lives in
 document.querySelectorAll('.dl-fb').forEach(function(btn){btn.addEventListener('click',function(){
  var panel=btn.closest('.panel');var out=[];
  panel.querySelectorAll('table.grid tbody tr').forEach(function(r){
   var fb=r.querySelector('td.fb');var t=fb?fb.innerText.trim():'';
   if(t)out.push({test:r.dataset.name,status:r.dataset.status,flags:r.dataset.flags,feedback:t});});
  var blob=new Blob([JSON.stringify(out,null,2)],{type:'application/json'});
  var a=document.createElement('a');a.href=URL.createObjectURL(blob);
  a.download='stdf-audit-feedback.json';a.click();});});
 // sort a plot tab's cards by its goodness metric (data-sort); default worst-first, toggles
 document.querySelectorAll('.sortbar').forEach(function(bar){
  var panel=bar.closest('.panel');var grid=panel.querySelector('.plotgrid');
  var btn=bar.querySelector('.sortbtn');if(!grid||!btn)return;
  function run(){
   var worst=btn.dataset.dir!=='best';var wdir=bar.dataset.worst;
   var asc=worst?(wdir==='asc'):(wdir!=='asc');
   var cards=[].slice.call(grid.querySelectorAll('[data-sort]'));
   cards.sort(function(a,b){
    var x=parseFloat(a.dataset.sort),y=parseFloat(b.dataset.sort);
    var xn=isNaN(x),yn=isNaN(y);
    if(xn&&yn)return 0;if(xn)return 1;if(yn)return -1;  // missing metric sinks to the end
    return asc?x-y:y-x;});
   cards.forEach(function(c){grid.appendChild(c);});
  }
  btn.addEventListener('click',function(){
   btn.dataset.dir=btn.dataset.dir==='best'?'worst':'best';
   btn.textContent=btn.dataset.dir==='best'?'best first':'worst first';run();});
  run();
 });
 // expand a table to a full-window overlay (and back); Esc also exits
 function setFull(tp,on){tp.classList.toggle('full',on);var b=tp.querySelector('.expand-btn');
  if(b)b.textContent=on?'\\u2715 Close':'\\u26f6 Full screen';}
 document.querySelectorAll('.expand-btn').forEach(function(btn){btn.addEventListener('click',function(){
  var tp=btn.closest('.tablepanel');if(tp)setFull(tp,!tp.classList.contains('full'));});});
 document.addEventListener('keydown',function(e){if(e.key==='Escape')
  document.querySelectorAll('.tablepanel.full').forEach(function(tp){setFull(tp,false);});});
})();
"""

# a shared "expand this table to full window" button (wired in _DASHBOARD_JS)
_EXPAND_BTN = '<button class="expand-btn">⛶ Full screen</button>'

# distribution shape → the coarse flag/chip it contributes (normal/skewed/sparse contribute none)
_SHAPE_FLAG = {
    "bimodal": "bimodal", "long-tail-right": "long-tail", "long-tail-left": "long-tail",
    "outliers": "outliers", "discrete": "discrete",
}
# every filterable flag, in display order, with its chip label
_FLAG_ORDER = ("changed", "added", "removed", "identical", "shifted", "correlated", "uncorrelated",
               "site-shift", "bimodal", "long-tail", "outliers", "discrete", "functional")
_FLAG_LABELS = {
    "changed": "Changed", "added": "Added", "removed": "Removed", "identical": "Identical",
    "shifted": "Shifted", "correlated": "Correlated", "uncorrelated": "Uncorrelated",
    "site-shift": "Site shift", "bimodal": "Bimodal", "long-tail": "Long-tail",
    "outliers": "Outliers", "discrete": "Discrete", "functional": "Functional",
}
# which flags are meaningful in which tab — a view only shows chips relevant to what it displays
_VIEW_FLAGS = {
    "explore": _FLAG_ORDER,
    "diff": ("changed", "added", "removed", "identical"),
    "qq": ("correlated", "uncorrelated", "shifted"),
    "hist": ("shifted", "bimodal", "long-tail", "outliers", "discrete"),
    "trend": ("shifted", "correlated", "uncorrelated"),
    "site": ("site-shift", "shifted", "bimodal", "long-tail", "outliers", "discrete"),
}
# each plot tab's "goodness" metric to sort its cards by: (label, which end is WORST) — the plot cards
# carry a matching data-sort value, and the tab defaults to worst-first so you review the risks first.
_VIEW_SORT = {
    "qq": ("Q-Q R²", "asc"),      # worst correlation = lowest R²
    "hist": ("Cpk", "asc"),        # worst capability = lowest Cpk
    "trend": ("shift σ", "desc"),  # worst = biggest run-to-run move
    "site": ("site Δσ", "desc"),   # worst site match = biggest site spread
}


def _sort_attr(v: float | None) -> str:
    """A ``data-sort`` attribute for a plot card (empty ⇒ the JS sorts it to the end)."""
    return f' data-sort="{v}"' if v is not None else ' data-sort=""'


def _sortbar(view: str) -> str:
    spec = _VIEW_SORT.get(view)
    if not spec:
        return ""
    label, worst = spec
    return (f'<div class="sortbar" data-worst="{worst}">'
            f'<span class="chiphint">Sort by {escape(label)}:</span>'
            '<button class="sortbtn" data-dir="worst">worst first</button></div>')


@dataclass
class TestAnalysis:
    """Everything the explorer needs about one test, computed on the CPU (no AI): its record type,
    distribution shape, run-to-run comparison, capability, and the coarse flags to sort/filter on."""

    name: str
    status: str
    rec: str
    shape: str
    comp: dict[str, Any]
    lo: float | None
    hi: float | None
    flags: list[str]
    cpk: float | None = None
    site_spread: float | None = None


def _values_by_name(run: StdfRun) -> tuple[dict[str, list[float]], dict[str, str]]:
    """One pass over a run: name → its numeric results, and name → record type (PTR/MPR/FTR)."""
    vals: dict[str, list[float]] = defaultdict(list)
    rec: dict[str, str] = {}
    for t in run.tests:
        rec.setdefault(t.test_txt, t.rec_type)
        if t.result is not None:
            vals[t.test_txt].append(t.result)
    return vals, rec


def _site_values_by_name(run: StdfRun) -> dict[str, dict[int, list[float]]]:
    """One pass: name → {site → its results} — for site-to-site whiskers + site-shift tagging."""
    out: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for t in run.tests:
        if t.result is not None:
            out[t.test_txt][t.site].append(t.result)
    return out


def _analyze(
    rows: list[DiffRow], va_map: dict[str, list[float]], vb_map: dict[str, list[float]],
    rec_map: dict[str, str], da: dict[str, TestDef], db: dict[str, TestDef],
    site_map_b: dict[str, dict[int, list[float]]],
) -> list[TestAnalysis]:
    out: list[TestAnalysis] = []
    for r in rows:
        va, vb = va_map.get(r.name, []), vb_map.get(r.name, [])
        rec = rec_map.get(r.name, "PTR")
        comp = analytics.compare_distributions(va, vb)
        shape = analytics.classify_distribution(vb or va)["shape"]
        d = db.get(r.name) or da.get(r.name)
        lo, hi = (d.lo, d.hi) if d else (None, None)
        site = analytics.site_dispersion(site_map_b.get(r.name, {}))
        flags = [r.status]
        if comp["shifted"]:
            flags.append("shifted")
        if comp["correlated"] is True:
            flags.append("correlated")
        if comp["uncorrelated"]:
            flags.append("uncorrelated")
        if site["site_shift"]:
            flags.append("site-shift")
        if rec == "FTR":
            flags.append("functional")
        if _SHAPE_FLAG.get(shape):
            flags.append(_SHAPE_FLAG[shape])
        cpk = analytics.capability(vb or va, lo, hi)["cpk"]  # Cpk of the current run vs its limits
        out.append(
            TestAnalysis(r.name, r.status, rec, shape, comp, lo, hi, flags, cpk, site["spread_sigma"])
        )
    return out


def _chipbar(present: set[str], view: str) -> str:
    """A context chipbar: only the flags that are (a) meaningful for this tab and (b) actually present
    in the data. Chips are **three-state** (click cycles include → exclude → off) and combine under a
    per-bar **Match: Any/All** toggle, so e.g. Diff "everything but identical" = exclude Identical."""
    chips = "".join(
        f'<button class="chip" data-flag="{f}">{escape(_FLAG_LABELS[f])}</button>'
        for f in _VIEW_FLAGS[view] if f in present
    )
    if not chips:
        return ""
    return (
        '<div class="chipbar"><button class="chipmode" data-mode="or">Match: Any</button>'
        '<button class="chipclear">Clear</button>'
        '<span class="chiphint">click a chip: include → exclude → off</span>' + chips + "</div>"
    )


def _explore_table_html(analyses: list[TestAnalysis], label_a: str, label_b: str) -> str:
    """The **Explore** surface: every test tagged with record type, distribution shape, run-to-run
    shift, and Q-Q correlation — sortable, chip- + text-filterable, with an editable feedback cell.
    This is where you push the load onto scripts and jump straight to the tests that matter."""
    heads = ["Status", "Test", "rec", "shape", f"n {escape(label_a)}", f"n {escape(label_b)}",
             f"mean {escape(label_a)}", f"mean {escape(label_b)}", "shift %", "shift σ", "Q-Q R²",
             "site Δσ", "Cpk", "outlier %", "flags", "Feedback"]
    thead = "".join(f'<th class="sortable">{h}</th>' for h in heads)

    def numcell(v: float | None, fmt: str = "{:.3g}") -> str:
        if v is None:
            return "<td>—</td>"
        return f'<td data-v="{v}">{fmt.format(v)}</td>'

    body = []
    for a in analyses:
        c = a.comp
        qq = c["qq_r2"]
        comparable = bool(c["n_a"] and c["n_b"])
        chips = " ".join(f'<span class="tag">{escape(f)}</span>'
                         for f in a.flags if f not in ("identical",))
        body.append(
            f'<tr data-name="{escape(a.name)}" data-status="{a.status}" '
            f'data-flags="{" ".join(a.flags)}">'
            f'<td><span class="badge {a.status}">{a.status}</span></td>'
            f"<td>{escape(a.name)}</td><td>{a.rec}</td><td>{escape(a.shape)}</td>"
            f'<td data-v="{c["n_a"]}">{c["n_a"]}</td><td data-v="{c["n_b"]}">{c["n_b"]}</td>'
            f"{numcell(c['mean_a'] if c['n_a'] else None, '{:.4g}')}"
            f"{numcell(c['mean_b'] if c['n_b'] else None, '{:.4g}')}"
            f"{numcell(c['pct_shift'] if comparable else None, '{:+.1f}')}"
            f"{numcell(c['z_shift'] if comparable else None, '{:+.2f}')}"
            f"{numcell(qq if qq is not None else None, '{:.3f}')}"
            f"{numcell(a.site_spread)}"
            f"{numcell(a.cpk)}"
            f"{numcell(c.get('outlier_frac_b'))}"
            f"<td>{chips}</td><td class=\"fb\" contenteditable=\"true\"></td></tr>"
        )
    present = {f for a in analyses for f in a.flags}
    toolbar = (
        '<div class="toolbar"><input type="text" class="rowfilter" placeholder="filter tests…">'
        '<button class="dl-fb">⬇ Download feedback</button>' + _EXPAND_BTN +
        '<span class="hint">chips + column headers + the box all filter/sort · '
        f"{len(analyses)} tests · click a Feedback cell to annotate</span></div>"
    )
    return (
        "<h2>Explore — every test classified on the CPU</h2>"
        '<div class="tablepanel">' + _chipbar(present, "explore") + toolbar
        + '<div class="tablewrap"><table class="grid"><thead><tr>' + thead
        + f"</tr></thead><tbody>{''.join(body)}</tbody></table></div></div>"
    )


def _interesting_score(a: TestAnalysis) -> float:
    s = abs(float(a.comp.get("z_shift") or 0.0))
    s += 3.0 if "uncorrelated" in a.flags else 0.0
    s += 1.5 if a.status in ("added", "removed") else 0.0
    s += 1.0 if set(a.flags) & {"bimodal", "long-tail", "outliers"} else 0.0
    return s


def _pick_plotted(analyses: list[TestAnalysis], cap: int) -> list[TestAnalysis]:
    """Choose the tests to plot as a **balanced mix** across the interesting categories (round-robin),
    so the gallery isn't all one flag — the user sees shifts, shape changes, added/removed and shape
    archetypes together, then narrows with the chips."""
    buckets = {
        cat: sorted((a for a in analyses if cat in a.flags), key=_interesting_score, reverse=True)
        for cat in ("uncorrelated", "shifted", "added", "removed", "bimodal", "outliers",
                    "long-tail", "discrete", "correlated")  # correlated = stable examples for contrast
    }
    picked: list[TestAnalysis] = []
    seen: set[str] = set()
    idx = dict.fromkeys(buckets, 0)
    while len(picked) < cap:
        progressed = False
        for cat, bucket in buckets.items():
            i = idx[cat]
            if i < len(bucket):
                idx[cat] += 1
                progressed = True
                a = bucket[i]
                if a.name not in seen:
                    seen.add(a.name)
                    picked.append(a)
                    if len(picked) >= cap:
                        break
        if not progressed:
            break
    if not picked:  # nothing flagged — fall back to the first tests so the tab isn't empty
        picked = analyses[:cap]
    return picked


def audit_report_html(
    run_a: StdfRun, run_b: StdfRun, *, backend: str = "matplotlib",
    label_a: str = "A", label_b: str = "B", max_plots: int = 48,
) -> str:
    """A themed, **tabbed, interactive** STDF audit dashboard (R-STDF-2), built for **thousands** of
    PTR/MPR/FTR tests. Six tabs on one page: **Explore** (every test classified on the CPU —
    distribution shape, run-to-run shift, Q-Q correlation; chip/sort/text filters + editable
    feedback), **Diff** (the field-by-field Beyond-Compare table), **Q-Q**, **Histograms** (red
    LLM/HLM limit lines + Cpl/Cpu/Cpk), **Trend**, **Site**. The plot tabs render the most
    interesting tests (biggest shift / uncorrelated / shape anomalies / added-removed) and carry the
    same chips, so you filter *plots* the way you filter the table."""
    rep = audit_runs(run_a, run_b)
    cond_keys, rows = diff_tests(run_a, run_b)
    da, db = _defs(run_a), _defs(run_b)
    va_map, rec_a = _values_by_name(run_a)
    vb_map, rec_b = _values_by_name(run_b)
    rec_map = {**rec_a, **rec_b}
    site_map = _site_values_by_name(run_b)  # name → {site → results} (whiskers + site-shift tagging)
    analyses = _analyze(rows, va_map, vb_map, rec_map, da, db, site_map)
    # attach run-B outlier fraction for the explore column (cheap, needed only for display)
    for a in analyses:
        a.comp["outlier_frac_b"] = analytics.classify_distribution(
            vb_map.get(a.name) or va_map.get(a.name, [])
        )["outlier_frac"] * 100.0

    ya_p, ya_t = rep.yield_a
    yb_p, yb_t = rep.yield_b
    ya = 100 * ya_p / ya_t if ya_t else 0.0
    yb = 100 * yb_p / yb_t if yb_t else 0.0
    recs = {r: sum(1 for a in analyses if a.rec == r) for r in ("PTR", "MPR", "FTR")}
    n_uncorr = sum(1 for a in analyses if "uncorrelated" in a.flags)
    n_shift = sum(1 for a in analyses if "shifted" in a.flags)
    summary = (
        f'<div class="acard"><strong>Yield</strong>: {escape(label_a)} {ya:.1f}% ({ya_p}/{ya_t}) → '
        f"{escape(label_b)} {yb:.1f}% ({yb_p}/{yb_t}) · <strong>Δ {yb - ya:+.1f}%</strong> "
        f"&nbsp;|&nbsp; {len(analyses)} tests "
        f"(PTR {recs['PTR']} · MPR {recs['MPR']} · FTR {recs['FTR']}) &nbsp;|&nbsp; "
        f"{len(rep.added)} added · {len(rep.removed)} removed · "
        f"<strong>{n_shift} shifted · {n_uncorr} uncorrelated</strong></div>"
    )
    tabbar = (
        '<div class="tabbar">'
        '<button class="tab active" data-t="explore">Explore</button>'
        '<button class="tab" data-t="diff">Diff</button>'
        '<button class="tab" data-t="qq">Q-Q</button>'
        '<button class="tab" data-t="hist">Histograms</button>'
        '<button class="tab" data-t="trend">Trend</button>'
        '<button class="tab" data-t="site">Site</button></div>'
    )
    explore_panel = (
        '<div class="panel" data-p="explore"><section class="acard">'
        + _explore_table_html(analyses, label_a, label_b) + "</section></div>"
    )
    diff_panel = (
        '<div class="panel hidden" data-p="diff"><section class="acard">'
        + _diff_table_interactive(rows, cond_keys, label_a, label_b) + "</section></div>"
    )

    def plot(kind: str, **kw: object) -> str:
        return analytics.render_plot(kind, backend=backend, include_js=False, **kw)  # type: ignore[arg-type]

    # render plots only for the most interesting tests; cards carry data-flags/shape for chip filtering
    plotted = _pick_plotted(analyses, max_plots)

    qq, hist, trend, site = [], [], [], []
    for a in plotted:
        name, va, vb = a.name, va_map.get(a.name, []), vb_map.get(a.name, [])
        attrs = f'data-flags="{" ".join(a.flags)}" data-shape="{a.shape}"'
        sst = "" if a.site_spread is None else f' · site Δ{a.site_spread:.2f}σ'
        head = (f'<h3>{escape(name)}</h3><p class="stats">{a.rec} · {escape(a.shape)}'
                f' · Q-Q R²={_cap_fmt(a.comp["qq_r2"])}'
                f' · shift {a.comp["pct_shift"]:+.1f}%{sst}</p>')
        if len(va) >= 2 and len(vb) >= 2:
            p = plot("qq", series=[(label_a, va), (label_b, vb)],
                     title=f"{name} — Q-Q ({label_a} vs {label_b})", xlabel=label_a, ylabel=label_b)
        else:
            p = "<p class='stats'>needs ≥2 points in each revision for a Q-Q</p>"
        qq.append(f'<section class="acard" {attrs}{_sort_attr(a.comp["qq_r2"])}>{head}{p}</section>')
        # overlaid translucent histogram: old (A) blue, new (B) green — shifts pop out
        hist_series = [(label_a, va), (label_b, vb)] if (va and vb) else None
        vals = vb or va
        vlines = [x for x in (a.lo, a.hi) if x is not None]
        hp = (plot("histogram", series=hist_series, y=None if hist_series else vals,
                   title=f"{name} — {label_a} vs {label_b}", xlabel=f"{name} result",
                   ylabel="count", vlines=vlines) if vals else "<p class='stats'>no data</p>")
        cap_tbl = (
            '<div class="scroll"><table class="grid"><thead><tr><th>run</th><th>n</th><th>mean</th>'
            '<th>median</th><th>std</th><th>min</th><th>max</th><th>Cpl</th><th>Cpu</th><th>Cpk</th>'
            f"</tr></thead><tbody>{_capability_row(label_a, va, a.lo, a.hi)}"
            f"{_capability_row(label_b, vb, a.lo, a.hi)}</tbody></table></div>"
        )
        hist.append(f'<section class="acard" {attrs}{_sort_attr(a.cpk)}>{head}{hp}'
                    f'<p class="stats">spec limits (red): LLM={_num(a.lo)} · HLM={_num(a.hi)}</p>'
                    f"{cap_tbl}</section>")
        tp = [(lbl, analytics.summary_stats(v)["mean"]) for lbl, v in ((label_a, va), (label_b, vb)) if v]
        if tp:
            tpl = plot("linear", x=list(range(len(tp))), y=[p2[1] for p2 in tp],
                       title=f"{name} mean trend", xlabel="revision", ylabel=f"{name} mean")
            zmag = abs(a.comp["z_shift"]) if (a.comp["n_a"] and a.comp["n_b"]) else None
            trend.append(f'<section class="acard" {attrs}{_sort_attr(zmag)}>{head}{tpl}</section>')
        groups = site_map.get(name, {})
        if len(groups) > 1:
            sp = plot("whisker", series=[(f"site {s}", v) for s, v in sorted(groups.items())],
                      title=f"{name} by site", ylabel=name)
            site.append(f'<section class="acard" {attrs}{_sort_attr(a.site_spread)}>{head}{sp}</section>')

    plotted_flags = {f for a in plotted for f in a.flags}

    def panel(pid: str, cards: list[str], empty: str, *, view: str = "") -> str:
        bar = _chipbar(plotted_flags, view) if view else ""
        sortbar = _sortbar(view) if view and cards else ""
        note = (f'<p class="stats">showing the {len(plotted)} most interesting tests '
                "(largest shift · uncorrelated · shape anomalies · added/removed) — chips filter, the "
                "sort button ranks by goodness.</p>") if view and cards else ""
        return (f'<div class="panel hidden" data-p="{pid}">{bar}{sortbar}{note}'
                f'<div class="plotgrid">{"".join(cards) or empty}</div></div>')

    plotly_prefix = (
        analytics.plotly_js_tag() if backend == "plotly" and any((qq, hist, trend, site)) else ""
    )
    body = (
        plotly_prefix + summary + tabbar + explore_panel + diff_panel
        + panel("qq", qq, '<p class="stats">no tests to compare</p>', view="qq")
        + panel("hist", hist, '<p class="stats">no tests</p>', view="hist")
        + panel("trend", trend, '<p class="stats">no trend data</p>', view="trend")
        + panel("site", site, '<p class="stats">single-site data</p>', view="site")
        + f"<script>{_DASHBOARD_JS}</script>"
    )
    return _page(
        f"STDF audit — {label_a} vs {label_b}", body,
        subtitle="explore + diff + capability plots at scale",
    )
