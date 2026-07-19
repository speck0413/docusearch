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
