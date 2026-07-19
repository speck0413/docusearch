"""General analytics + plotting engine (Phase 6 / GATE 6).

A **format-agnostic** capability the agent can invoke on any numeric/tabular data — a column from a
text or Excel file, or STDF test results — to compute summary statistics and render charts for
embedding in a report. The plot backend is **pluggable** (config ``stdf_analytics.plot_backend`` /
the ``backend`` arg): ``matplotlib`` → a deterministic ``data:`` PNG ``<img>`` (Agg, no display),
``plotly`` → a self-contained interactive HTML ``<div>`` (JS inlined). Both are lazy-imported so the
core never pulls a plotting library.

Plot kinds: ``histogram``, ``whisker`` (box), ``quantile``, ``qq`` (quantile-quantile of two
series), ``xy`` (scatter), ``linear`` (line/trend).
"""

from __future__ import annotations

import base64
import io
import statistics
from collections.abc import Sequence
from html import escape

Number = float
Series = Sequence[tuple[str, Sequence[float]]]  # named series for whisker/qq

PLOT_KINDS = ("histogram", "whisker", "quantile", "qq", "xy", "linear")


def summary_stats(values: Sequence[float | None]) -> dict[str, float]:
    """n / mean / median / std (population) / min / max over the non-null values (empty ⇒ {n:0})."""
    xs = [float(v) for v in values if v is not None]
    if not xs:
        return {"n": 0}
    return {
        "n": len(xs),
        "mean": statistics.fmean(xs),
        "median": statistics.median(xs),
        "std": statistics.pstdev(xs) if len(xs) > 1 else 0.0,
        "min": min(xs),
        "max": max(xs),
    }


def cpk(values: Sequence[float | None], lo: float, hi: float) -> float | None:
    """Process capability index vs a lower/upper spec limit, or None if undefined (n<2 or std=0)."""
    s = summary_stats(values)
    if s.get("n", 0) < 2 or s["std"] == 0:
        return None
    return min(hi - s["mean"], s["mean"] - lo) / (3 * s["std"])


def _quantile_points(values: Sequence[float], n: int) -> list[float]:
    """``n`` evenly-spaced quantiles of ``values`` (linear interpolation), for Q-Q / quantile plots."""
    xs = sorted(float(v) for v in values)
    if not xs:
        return []
    if len(xs) == 1 or n <= 1:
        return [xs[len(xs) // 2]] * max(n, 1)
    out: list[float] = []
    for i in range(n):
        pos = i / (n - 1) * (len(xs) - 1)
        lo = int(pos)
        frac = pos - lo
        hi = min(lo + 1, len(xs) - 1)
        out.append(xs[lo] * (1 - frac) + xs[hi] * frac)
    return out


def render_plot(
    kind: str,
    *,
    series: Series | None = None,
    x: Sequence[float] | None = None,
    y: Sequence[float] | None = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    backend: str = "matplotlib",
    bins: int = 20,
) -> str:
    """Render one chart to a **self-contained** HTML fragment. ``histogram``/``quantile`` use ``y``
    (or the first ``series``); ``whisker`` uses all named ``series``; ``qq`` compares the first two
    ``series``; ``xy``/``linear`` use ``x`` + ``y``. Returns a ``data:`` PNG ``<img>`` (matplotlib)
    or a plotly ``<div>``."""
    k = kind.lower()
    if k not in PLOT_KINDS:
        raise ValueError(f"unknown plot kind {kind!r}; expected one of {PLOT_KINDS}")
    if k == "qq" and (not series or len(series) < 2):  # qq compares TWO series (red-team H1)
        raise ValueError("plot kind 'qq' needs two named series to compare")
    if backend == "matplotlib":
        return _render_matplotlib(k, series, x, y, title, xlabel, ylabel, bins)
    if backend == "plotly":
        return _render_plotly(k, series, x, y, title, xlabel, ylabel, bins)
    raise ValueError(f"unknown plot backend {backend!r}; expected 'matplotlib' or 'plotly'")


def _primary(series: Series | None, y: Sequence[float] | None) -> list[float]:
    if y is not None:
        return [float(v) for v in y]
    if series:
        return [float(v) for v in series[0][1]]
    return []


def _render_matplotlib(
    kind: str, series: Series | None, x: Sequence[float] | None, y: Sequence[float] | None,
    title: str, xlabel: str, ylabel: str, bins: int,
) -> str:
    import matplotlib

    matplotlib.use("Agg")  # headless, deterministic — no display, no GUI backend
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    if kind == "histogram":
        ax.hist(_primary(series, y), bins=bins)
    elif kind == "whisker":
        data = [list(map(float, s[1])) for s in (series or [])]
        ax.boxplot(data, tick_labels=[s[0] for s in (series or [])])
    elif kind == "quantile":
        vals = sorted(_primary(series, y))
        qs = [i / (len(vals) - 1) for i in range(len(vals))] if len(vals) > 1 else [0.0]
        ax.plot(qs, vals, marker="o")
    elif kind == "qq":
        s = series or []
        n = min(len(s[0][1]), len(s[1][1])) if len(s) >= 2 else 0
        qa, qb = _quantile_points(s[0][1], n), _quantile_points(s[1][1], n)
        ax.scatter(qa, qb)
        if qa and qb:
            lo, hi = min(qa + qb), max(qa + qb)
            ax.plot([lo, hi], [lo, hi], color="gray", linestyle="--")  # y=x reference
    elif kind == "xy":
        ax.scatter(list(x or []), list(y or []))
    elif kind == "linear":
        ax.plot(list(x or []), list(y or []), marker="o")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80)
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f'<img class="plot" alt="{escape(title)}" src="data:image/png;base64,{b64}">'


def _render_plotly(
    kind: str, series: Series | None, x: Sequence[float] | None, y: Sequence[float] | None,
    title: str, xlabel: str, ylabel: str, bins: int,
) -> str:
    import plotly.graph_objects as go

    fig = go.Figure()
    if kind == "histogram":
        fig.add_histogram(x=_primary(series, y), nbinsx=bins)
    elif kind == "whisker":
        for label, data in series or []:
            fig.add_box(y=list(map(float, data)), name=label)
    elif kind == "quantile":
        vals = sorted(_primary(series, y))
        qs = [i / (len(vals) - 1) for i in range(len(vals))] if len(vals) > 1 else [0.0]
        fig.add_scatter(x=qs, y=vals, mode="lines+markers")
    elif kind == "qq":
        s = series or []
        n = min(len(s[0][1]), len(s[1][1])) if len(s) >= 2 else 0
        qa, qb = _quantile_points(s[0][1], n), _quantile_points(s[1][1], n)
        fig.add_scatter(x=qa, y=qb, mode="markers")
        if qa and qb:
            lo, hi = min(qa + qb), max(qa + qb)
            fig.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines", line={"dash": "dash"})
    elif kind == "xy":
        fig.add_scatter(x=list(x or []), y=list(y or []), mode="markers")
    elif kind == "linear":
        fig.add_scatter(x=list(x or []), y=list(y or []), mode="lines+markers")
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel, showlegend=bool(series))
    # Deterministic div id (default is a random UUID) so the same inputs render byte-identical HTML.
    import hashlib

    div_id = "plot-" + hashlib.sha256(
        f"{kind}|{title}|{xlabel}|{ylabel}|{series}|{x}|{y}".encode()
    ).hexdigest()[:16]
    html: str = fig.to_html(full_html=False, include_plotlyjs="inline", div_id=div_id)
    return html
