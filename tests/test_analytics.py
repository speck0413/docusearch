"""General analytics/plot engine (GATE 6): stats + pluggable matplotlib/plotly plots, usable by
any engine on any numeric data."""

from __future__ import annotations

import pytest

from docusearch import analytics


def test_summary_stats() -> None:
    s = analytics.summary_stats([1.0, 2.0, 3.0, None, 4.0])
    assert s["n"] == 4
    assert s["mean"] == 2.5 and s["median"] == 2.5
    assert s["min"] == 1.0 and s["max"] == 4.0


def test_summary_stats_empty() -> None:
    assert analytics.summary_stats([None, None]) == {"n": 0}


def test_cpk() -> None:
    vals = [10.0, 10.1, 9.9, 10.05, 9.95]
    c = analytics.cpk(vals, lo=9.0, hi=11.0)
    assert c is not None and c > 0
    assert analytics.cpk([5.0], lo=0, hi=10) is None  # n<2 undefined


@pytest.mark.parametrize("backend", ["matplotlib", "plotly"])
@pytest.mark.parametrize(
    "kind,kwargs",
    [
        ("histogram", {"y": [1, 2, 2, 3, 3, 3, 4]}),
        ("whisker", {"series": [("A", [1, 2, 3]), ("B", [2, 3, 4, 5])]}),
        ("quantile", {"y": [3, 1, 2, 5, 4]}),
        ("qq", {"series": [("A", [1, 2, 3, 4]), ("B", [2, 3, 4, 5])]}),
        ("xy", {"x": [1, 2, 3], "y": [2, 4, 6]}),
        ("linear", {"x": [1, 2, 3, 4], "y": [1, 2, 1, 3]}),
    ],
)
def test_render_plot_kinds_both_backends(backend: str, kind: str, kwargs: dict) -> None:  # type: ignore[type-arg]
    out = analytics.render_plot(kind, title=f"{kind} test", backend=backend, **kwargs)
    assert isinstance(out, str) and len(out) > 100
    if backend == "matplotlib":
        assert out.startswith("<img") and "data:image/png;base64," in out
    else:
        assert "plotly" in out.lower()  # self-contained plotly div


def test_render_plot_rejects_bad_kind_and_backend() -> None:
    with pytest.raises(ValueError, match="plot kind"):
        analytics.render_plot("piechart", y=[1, 2, 3])
    with pytest.raises(ValueError, match="backend"):
        analytics.render_plot("histogram", y=[1, 2, 3], backend="ascii")


# ---------------------------------------------- algorithmic distribution intelligence


def _rng() -> object:
    import numpy as np
    return np.random.default_rng(20260719)  # recorded seed (R-SRCH-5)


def test_classify_distribution_labels_archetypes() -> None:
    import numpy as np
    r = _rng()
    normal = r.normal(0, 1, 600)
    bimodal = np.concatenate([r.normal(-4, 0.4, 300), r.normal(4, 0.4, 300)])
    longtail = r.exponential(1.0, 600)
    discrete = r.integers(0, 3, 600).astype(float)
    assert analytics.classify_distribution(normal)["shape"] == "normal"
    assert analytics.classify_distribution(bimodal)["shape"] == "bimodal"
    assert analytics.classify_distribution(longtail)["shape"] == "long-tail-right"
    assert analytics.classify_distribution(discrete)["shape"] == "discrete"
    assert analytics.classify_distribution([2.5] * 50)["shape"] == "degenerate"
    assert analytics.classify_distribution([1.0, 2.0, 3.0])["shape"] == "sparse"  # n < 8


def test_classify_distribution_is_deterministic() -> None:
    import numpy as np
    xs = _rng().normal(0, 1, 200)
    a = analytics.classify_distribution(xs)
    b = analytics.classify_distribution(np.asarray(xs))
    assert a == b


def test_compare_distributions_correlated_shifted_uncorrelated() -> None:
    import numpy as np
    r = _rng()
    a = r.normal(0, 1, 500)
    # same distribution → runs track each other, nothing flagged
    same = analytics.compare_distributions(a, r.normal(0, 1, 500))
    assert same["correlated"] is True and same["qq_r2"] > 0.98
    assert same["shifted"] is False and same["uncorrelated"] is False
    # same shape, big mean shift → flagged SHIFTED (not a shape change)
    shifted = analytics.compare_distributions(a, a + 3.0)
    assert shifted["shifted"] is True and shifted["uncorrelated"] is False
    # different shape, same mean (normal vs symmetric bimodal) → flagged UNCORRELATED (shape change)
    bim = np.concatenate([r.normal(-4, 0.4, 250), r.normal(4, 0.4, 250)])
    diff = analytics.compare_distributions(a, bim)
    assert diff["uncorrelated"] is True and diff["correlated"] is False and diff["qq_r2"] < 0.95


def test_compare_distributions_too_small() -> None:
    out = analytics.compare_distributions([1.0, 2.0], [3.0, 4.0])
    assert out["qq_r2"] is None and out["correlated"] is None


def test_histogram_overlays_two_series() -> None:
    r = _rng()
    a, b = list(r.normal(0, 1, 200)), list(r.normal(1.5, 1, 200))
    mpl = analytics.render_plot("histogram", series=[("old", a), ("new", b)], backend="matplotlib")
    assert "data:image/png" in mpl  # overlaid translucent bars baked into the PNG
    ply = analytics.render_plot("histogram", series=[("old", a), ("new", b)], backend="plotly",
                                include_js=False)
    assert "overlay" in ply and "old" in ply and "new" in ply  # two named traces, barmode overlay


def test_site_dispersion_flags_site_to_site_shift() -> None:
    r = _rng()
    matched = {s: list(r.normal(0, 1, 40)) for s in (1, 2, 3, 4)}      # sites agree
    assert analytics.site_dispersion(matched)["site_shift"] is False
    offset = {s: list(r.normal((s - 1) * 1.0, 1, 40)) for s in (1, 2, 3, 4)}  # one site walks off
    d = analytics.site_dispersion(offset)
    assert d["site_shift"] is True and d["spread_sigma"] > 0.5 and d["n_sites"] == 4
    assert analytics.site_dispersion({1: [1.0, 2.0, 3.0]})["site_shift"] is None  # single site


def test_whisker_ungrouped_column_does_not_crash() -> None:
    # regression: a whisker of an ungrouped column (series=None, values via y) used to raise in the
    # matplotlib boxplot path ("Dimensions of labels and X must be compatible") — found by the GATE 10
    # data-engine checkout. Both backends must render a single box from y, and stay blank (no crash)
    # when nothing is finite.
    from docusearch import analytics
    for backend in ("matplotlib", "plotly"):
        html = analytics.render_plot("whisker", y=[0.71, 0.72, 0.90, 0.70], series=None,
                                     title="vmin", xlabel="vmin", ylabel="v", backend=backend)
        assert len(html) > 200
        blank = analytics.render_plot("whisker", y=[float("nan"), float("inf")], series=None,
                                      backend=backend)
        assert len(blank) > 0  # degenerate all-non-finite: blank plot, not an exception
