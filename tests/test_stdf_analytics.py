"""STDF single-log analytics (R-STDF-2): audit two runs, site grouping, trend across runs."""

from __future__ import annotations

import pytest
from harness.stdf_synth import StdfBuilder

from docusearch import stdf, stdf_analytics


def _run(vmin_vals: list[float], *, add_extra: bool = False, sites: list[int] | None = None) -> stdf.StdfRun:
    b = StdfBuilder().far().mir(lot_id="L", job_nam="P")
    sites = sites or [1] * len(vmin_vals)
    for i, (v, site) in enumerate(zip(vmin_vals, sites, strict=True)):
        b.pir(site=site)
        b.ptr(1000, "VMIN", v, site=site, fail=(v < 0.70))
        if add_extra:
            b.ptr(2000, "IDDQ", 1e-6, site=site)
        b.prr(part_id=str(i + 1), hard_bin=1 if v >= 0.70 else 5, site=site)
    b.mrr()
    return stdf.parse_stdf_tests(b.to_bytes())


def test_site_groups() -> None:
    run = _run([0.71, 0.72, 0.90, 0.91], sites=[1, 1, 2, 2])
    groups = stdf_analytics.site_groups(run, 1000)
    assert set(groups) == {1, 2}
    assert groups[1] == pytest.approx([0.71, 0.72], abs=1e-5)  # float32 STDF results
    assert len(groups[2]) == 2


def test_trend_points() -> None:
    runs = [("run1", _run([0.71, 0.72])), ("run2", _run([0.75, 0.76])), ("run3", _run([0.80, 0.81]))]
    pts = stdf_analytics.trend_points(runs, 1000, stat="mean")
    labels = [p[0] for p in pts]
    means = [p[1] for p in pts]
    assert labels == ["run1", "run2", "run3"]
    assert means[0] < means[1] < means[2]  # rising VMIN trend


def test_audit_alignment_yield_and_conditions() -> None:
    run_a = _run([0.71, 0.72, 0.73, 0.74])           # all pass
    run_b = _run([0.69, 0.72, 0.73, 0.74], add_extra=True)  # one fail + a new test 2000
    rep = stdf_analytics.audit_runs(run_a, run_b)

    assert 2000 in rep.added and rep.removed == []  # IDDQ only in B
    assert any(d.test_num == 1000 for d in rep.matched)  # VMIN matched
    # yields: A all pass (4/4); B one part fails (3/4)
    assert rep.yield_a == (4, 4) and rep.yield_b == (3, 4)
    vmin = next(d for d in rep.matched if d.test_num == 1000)
    assert vmin.mean_delta is not None and vmin.mean_delta < 0  # B mean lower


def test_diff_tests_detects_limit_tnum_and_membership_changes() -> None:
    def run(tests: list[tuple[int, str, float, float, float]], cod: str) -> stdf.StdfRun:
        b = StdfBuilder().far().mir(lot_id="L", test_cod=cod)
        b.pir()
        for tnum, name, res, lo, hi in tests:
            b.ptr(tnum, name, res, lo=lo, hi=hi, units="V")
        b.prr(part_id="1", hard_bin=1)
        b.mrr()
        return stdf.parse_stdf_tests(b.to_bytes())

    ra = run([(1000, "VMIN", 0.72, 0.70, 0.85), (2000, "IDDQ", 1e-6, 0.0, 2e-6)], "WS1")
    rb = run([(1001, "VMIN", 0.72, 0.70, 0.80), (3000, "FMAX", 1.2, 0.0, 2.0)], "WS2")
    _keys, rows = stdf_analytics.diff_tests(ra, rb)
    by = {r.name: r for r in rows}
    assert by["VMIN"].status == "changed"
    assert by["VMIN"].changed == {"tnum", "hi"}  # test number 1000→1001 AND HLM 0.85→0.80
    assert by["IDDQ"].status == "removed"  # only in A
    assert by["FMAX"].status == "added"  # only in B

    html = stdf_analytics.audit_report_html(ra, rb, label_a="WS1", label_b="WS2")
    assert "0.85" in html and "0.8" in html  # both old and new HLM shown side by side
    assert 'td class="chg"' in html  # the changed cells are highlighted


def test_audit_dashboard_six_tabs_explore_conditions_and_capability() -> None:
    def run(hi: float, cod: str, corner: str) -> stdf.StdfRun:
        b = StdfBuilder().far().mir(lot_id="L", test_cod=cod)
        for i, v in enumerate([0.71, 0.72, 0.90, 0.91]):
            b.pir(site=1 if i < 2 else 2)
            b.dtr(f"COND: corner={corner}, temp=125C")
            b.ptr(1000, "VMIN", v, lo=0.70, hi=hi, units="V", site=1 if i < 2 else 2)
            b.prr(part_id=str(i + 1), hard_bin=1, site=1 if i < 2 else 2)
        b.mrr()
        return stdf.parse_stdf_tests(b.to_bytes())

    ra, rb = run(0.85, "WS1", "slow"), run(0.80, "WS2", "fast")

    # plotly backend so the red limit-line colour is inspectable as text (matplotlib bakes a PNG)
    html = stdf_analytics.audit_report_html(ra, rb, backend="plotly", label_a="WS1", label_b="WS2")
    # six tabs on one page (Explore is new)
    for tab in ("Explore", "Diff", "Q-Q", "Histograms", "Trend", "Site"):
        assert f">{tab}<" in html
    assert html.count('class="panel') == 6
    # Explore surface: algorithmic columns (incl Cpk + correlation/shift metadata to sort by)
    assert "shape" in html and "Q-Q R²" in html and "shift %" in html
    assert "shift σ" in html and "site Δσ" in html and ">Cpk</th>" in html
    # multi-select chips: relevant to the data (this run only has a 'changed' test) + Any/All + Clear
    assert 'class="chipbar"' in html and 'data-flag="changed"' in html
    assert 'class="chipmode"' in html and 'class="chipclear"' in html
    assert 'data-flags="' in html  # rows/cards carry the filter tokens
    # conditions appear in the diff table, old vs new, and the change is flagged
    assert "old corner" in html and "new corner" in html
    assert "slow" in html and "fast" in html and 'class="chg"' in html
    # Excel-like interactivity: sortable headers, class-based filter/export, editable feedback cells
    assert 'class="sortable"' in html and 'class="rowfilter"' in html and 'class="dl-fb"' in html
    assert 'contenteditable="true"' in html
    # fixed-size scrollable table (frozen header) + full-window expand
    assert 'class="tablewrap"' in html and 'class="tablepanel"' in html
    assert 'class="expand-btn"' in html and "position:sticky" in html
    # plot tabs sort by their goodness metric (Q-Q R² / Cpk / shift σ / site Δσ)
    assert 'class="sortbar"' in html and 'class="sortbtn"' in html and "data-sort=" in html
    assert "Sort by Q-Q R²" in html and "Sort by Cpk" in html and "Sort by site Δσ" in html
    # histogram tab: red spec-limit lines + capability stats
    assert "d64545" in html  # red LLM/HLM limit lines
    assert "Cpl" in html and "Cpu" in html and "Cpk" in html
    assert "median" in html and "std" in html


def test_report_builders_produce_self_contained_html() -> None:
    run_a = _run([0.71, 0.72, 0.73, 0.74])
    run_b = _run([0.69, 0.72, 0.73, 0.74], add_extra=True)
    # single-test plot
    plot = stdf_analytics.plot_test_html(run_a, 1000, kind="histogram", backend="matplotlib")
    assert "VMIN" in plot and "data:image/png" in plot and "mean=" in plot
    # site compare
    sc = stdf_analytics.site_compare_html(_run([0.71, 0.9], sites=[1, 2]), 1000)
    assert "site-to-site" in sc and "data:image/png" in sc
    # trend
    tr = stdf_analytics.trend_html([("r1", run_a), ("r2", run_b)], 1000, backend="matplotlib")
    assert "trend" in tr and "data:image/png" in tr
    # audit — themed page + Beyond-Compare test-diff table
    rep = stdf_analytics.audit_report_html(run_a, run_b, label_a="lotA", label_b="lotB")
    assert rep.startswith("<!doctype html>") and "docusearch" in rep  # shared theme
    assert "STDF audit" in rep and "Yield" in rep
    assert "Test diff" in rep and 'class="grid"' in rep  # tabular diff, not per-test plots
    assert "IDDQ" in rep and "2000" in rep  # the added test shown by name + number
    assert "badge added" in rep  # add/remove/changed status badges
