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
