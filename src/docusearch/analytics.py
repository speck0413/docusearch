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
from collections.abc import Mapping, Sequence
from html import escape
from typing import Any

import numpy as np

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


def capability(
    values: Sequence[float | None], lo: float | None, hi: float | None
) -> dict[str, float | None]:
    """Process-capability indices vs the spec limits: **Cpl** = (μ−LSL)/3σ, **Cpu** = (USL−μ)/3σ,
    **Cpk** = min(Cpl, Cpu). Each is None when undefined (n<2, σ=0, or the limit is missing)."""
    s = summary_stats(values)
    if s.get("n", 0) < 2 or s["std"] == 0:
        return {"cpl": None, "cpu": None, "cpk": None}
    cpl = (s["mean"] - lo) / (3 * s["std"]) if lo is not None else None
    cpu = (hi - s["mean"]) / (3 * s["std"]) if hi is not None else None
    both = [c for c in (cpl, cpu) if c is not None]
    return {"cpl": cpl, "cpu": cpu, "cpk": min(both) if both else None}


def cpk(values: Sequence[float | None], lo: float, hi: float) -> float | None:
    """Cpk = min(Cpl, Cpu) vs a lower/upper spec limit, or None if undefined (n<2 or std=0)."""
    return capability(values, lo, hi)["cpk"]


# --------------------------------------------------------- algorithmic distribution intelligence
# The point of these is to push the "which tests are interesting" judgement onto the CPU: every test
# is tagged with a distribution SHAPE and a run-to-run VERDICT so a human (or an agent) filters and
# sorts to the handful that matter, instead of eyeballing thousands of plots. No AI in the loop.

DISTRIBUTION_SHAPES = (
    "sparse", "degenerate", "discrete", "bimodal", "long-tail-right", "long-tail-left",
    "outliers", "normal", "skewed",
)


def _clean(values: Sequence[float | None]) -> list[float]:
    return [float(v) for v in values if v is not None]


def classify_distribution(values: Sequence[float | None]) -> dict[str, Any]:
    """Label a numeric test's distribution **shape** from its own data (no limits needed), with the
    moments/fractions behind the call. Shapes: ``sparse`` (n<8), ``degenerate`` (σ≈0), ``discrete``
    (few distinct values), ``bimodal`` (Sarle's bimodality coefficient high), ``long-tail-right/left``
    (heavy skew), ``outliers`` (fat IQR tails), ``normal`` (near-symmetric, mesokurtic), else
    ``skewed``. Deterministic — same input, same label."""
    xs = np.asarray(_clean(values), dtype=float)
    n = int(xs.size)
    base: dict[str, Any] = {
        "n": n, "mean": 0.0, "std": 0.0, "skew": 0.0, "kurtosis": 0.0,
        "outlier_frac": 0.0, "unique_frac": 0.0, "bimodality": 0.0,
    }
    if n < 8:
        return {**base, "shape": "sparse", "mean": float(xs.mean()) if n else 0.0}
    mean, std = float(xs.mean()), float(xs.std())
    base.update(mean=mean, std=std, unique_frac=float(np.unique(xs).size) / n)
    if std == 0:
        return {**base, "shape": "degenerate"}
    z = (xs - mean) / std
    skew = float((z ** 3).mean())
    kurt_excess = float((z ** 4).mean() - 3.0)
    q1, q3 = (float(v) for v in np.percentile(xs, [25, 75]))
    iqr = q3 - q1
    outlier_frac = (
        float(np.mean((xs < q1 - 1.5 * iqr) | (xs > q3 + 1.5 * iqr))) if iqr > 0 else 0.0
    )
    # Sarle's bimodality coefficient (excess kurtosis + small-sample correction); > ~0.55 → two modes.
    correction = 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3)) if n > 3 else 0.0
    denom = kurt_excess + correction
    bc = (skew ** 2 + 1.0) / denom if denom > 0 else 0.0
    base.update(skew=skew, kurtosis=kurt_excess, outlier_frac=outlier_frac, bimodality=bc)

    if base["unique_frac"] <= 0.05 or np.unique(xs).size <= 5:
        shape = "discrete"
    elif skew > 1.0:  # heavy skew is a tail, even if BC is also high — check it before bimodal
        shape = "long-tail-right"
    elif skew < -1.0:
        shape = "long-tail-left"
    elif bc > 0.60:  # near-symmetric but two-humped
        shape = "bimodal"
    elif outlier_frac > 0.03:
        shape = "outliers"
    elif abs(skew) < 0.5 and abs(kurt_excess) < 1.0:
        shape = "normal"
    else:
        shape = "skewed"
    return {**base, "shape": shape}


def site_dispersion(groups: Mapping[int, Sequence[float]]) -> dict[str, Any]:
    """Site-to-site agreement for one test: ``spread_sigma`` is the gap between the highest and
    lowest site mean in pooled-σ units. ``site_shift`` True means the sites don't track each other
    (a site-to-site mismatch) — the site analog of an uncorrelated run-to-run pair. The threshold is
    **sample-size aware** (a real ≥0.5σ effect plus a per-site-mean noise allowance), so 4 sites of
    small samples don't false-flag just from sampling scatter. None when there aren't ≥2 sites."""
    valid = {s: [float(x) for x in v] for s, v in groups.items() if len(v) >= 2}
    means = [statistics.fmean(v) for v in valid.values()]
    allv = [x for v in valid.values() for x in v]
    if len(means) < 2 or len(allv) < 8:
        return {"n_sites": len(valid), "spread_sigma": None, "site_shift": None}
    pooled = statistics.pstdev(allv) or 1.0
    spread = (max(means) - min(means)) / pooled
    n_min = min(len(v) for v in valid.values())
    crit = 0.5 + 2.5 / (n_min ** 0.5)  # 0.5σ real effect + a few site-mean standard errors
    return {"n_sites": len(means), "spread_sigma": spread, "site_shift": bool(spread > crit)}


def compare_distributions(
    a: Sequence[float | None], b: Sequence[float | None]
) -> dict[str, Any]:
    """Compare a test across two runs. ``qq_r2`` is the R² of the two runs' matched quantiles — the
    Q-Q linearity: ~1 means the runs track the same shape (**correlated**), low means the shape
    changed (**uncorrelated** — a shift or an unstable/new distribution). Also returns the mean shift
    (raw, % of |mean A|, and in pooled-σ), and the KS distance. ``None`` fields when a run is too
    small (<4 points)."""
    xa, xb = _clean(a), _clean(b)
    out: dict[str, Any] = {
        "n_a": len(xa), "n_b": len(xb), "mean_a": float(np.mean(xa)) if xa else 0.0,
        "mean_b": float(np.mean(xb)) if xb else 0.0, "mean_shift": 0.0, "pct_shift": 0.0,
        "z_shift": 0.0, "ks": 0.0, "ks_crit": 0.0, "qq_r2": None, "differs": None,
        "correlated": None, "shifted": None, "uncorrelated": None,
    }
    if len(xa) < 4 or len(xb) < 4:
        return out
    va, vb = np.asarray(xa), np.asarray(xb)
    ma, mb = float(va.mean()), float(vb.mean())
    shift = mb - ma
    pooled = float(np.sqrt((va.var() + vb.var()) / 2.0)) or 1.0
    n = min(len(va), len(vb), 256)
    # Q-Q on the trimmed body [2%, 98%]: the extreme tails of a heavy-tailed/bimodal test are noisy
    # sample-to-sample even when the shape is unchanged, and would otherwise read as "uncorrelated".
    probs = np.linspace(0.02, 0.98, n)
    qa, qb = np.quantile(va, probs), np.quantile(vb, probs)
    if qa.std() == 0 or qb.std() == 0:
        qq_r2 = 1.0 if qa.std() == qb.std() else 0.0
    else:
        qq_r2 = float(np.corrcoef(qa, qb)[0, 1] ** 2)
    grid = np.linspace(min(va.min(), vb.min()), max(va.max(), vb.max()), 128)
    cdf_a = np.searchsorted(np.sort(va), grid, side="right") / len(va)
    cdf_b = np.searchsorted(np.sort(vb), grid, side="right") / len(vb)
    ks = float(np.max(np.abs(cdf_a - cdf_b)))
    z_shift = shift / pooled
    # The robust run-to-run change detector is the two-sample KS gap vs its critical value — unlike
    # Q-Q R² it's archetype-independent (a same-distribution bimodal/discrete/outlier pair doesn't
    # false-flag). Require 1.3× the α=0.01 critical value for a confident "differs".
    ks_crit = 1.63 * float(np.sqrt(1.0 / len(va) + 1.0 / len(vb)))
    differs = ks > 1.3 * ks_crit
    mean_moved = abs(z_shift) > 0.5
    out.update(
        mean_a=ma, mean_b=mb, mean_shift=shift, pct_shift=100.0 * shift / (abs(ma) or 1.0),
        z_shift=z_shift, ks=ks, ks_crit=ks_crit, qq_r2=qq_r2, differs=bool(differs),
        correlated=bool(not differs),               # runs track each other (Q-Q on a line)
        shifted=bool(differs and mean_moved),        # differs, driven by a mean shift
        uncorrelated=bool(differs and not mean_moved),  # differs by SHAPE — a Q-Q non-linearity
    )
    return out


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
    vlines: Sequence[float] = (),
    include_js: bool = True,
) -> str:
    """Render one chart to a **self-contained** HTML fragment. ``histogram``/``quantile`` use ``y``
    (or the first ``series``); ``whisker`` uses all named ``series``; ``qq`` compares the first two
    ``series``; ``xy``/``linear`` use ``x`` + ``y``. ``vlines`` draws **red vertical limit lines``
    (e.g. LLM/HLM on a histogram). ``include_js`` only matters for the plotly backend: it inlines the
    ~3.5 MB plotly library — set it False on all-but-the-first plot of a multi-plot page (they share
    the one runtime) so the page doesn't embed the library N times. Returns a ``data:`` PNG ``<img>``
    (matplotlib, ``include_js`` ignored) or a plotly ``<div>``."""
    k = kind.lower()
    if k not in PLOT_KINDS:
        raise ValueError(f"unknown plot kind {kind!r}; expected one of {PLOT_KINDS}")
    if k == "qq" and (not series or len(series) < 2):  # qq compares TWO series (red-team H1)
        raise ValueError("plot kind 'qq' needs two named series to compare")
    lines = [float(v) for v in vlines if v is not None]
    if backend == "matplotlib":
        return _render_matplotlib(k, series, x, y, title, xlabel, ylabel, bins, lines)
    if backend == "plotly":
        return _render_plotly(k, series, x, y, title, xlabel, ylabel, bins, lines, include_js)
    raise ValueError(f"unknown plot backend {backend!r}; expected 'matplotlib' or 'plotly'")


# overlaid-histogram colours: first series (old/A) blue, second (new/B) green, then amber/violet
_HIST_COLORS = ("#4c8dff", "#3cb371", "#e6a020", "#c060d0")


def _primary(series: Series | None, y: Sequence[float] | None) -> list[float]:
    if y is not None:
        return [float(v) for v in y]
    if series:
        return [float(v) for v in series[0][1]]
    return []


def _hist_series(series: Series | None, y: Sequence[float] | None) -> list[tuple[str, list[float]]]:
    """The datasets to draw on a histogram: every named ``series`` (overlaid, translucent) or, for a
    single distribution, ``y`` alone."""
    if series:
        return [(str(lbl), [float(v) for v in data]) for lbl, data in series]
    if y is not None:
        return [("", [float(v) for v in y])]
    return []


def _render_matplotlib(
    kind: str, series: Series | None, x: Sequence[float] | None, y: Sequence[float] | None,
    title: str, xlabel: str, ylabel: str, bins: int, vlines: Sequence[float],
) -> str:
    import matplotlib

    matplotlib.use("Agg")  # headless, deterministic — no display, no GUI backend
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    if kind == "histogram":
        hs = _hist_series(series, y)
        allv = [v for _, vals in hs for v in vals]
        edges: list[float] | int = (
            [float(e) for e in np.histogram_bin_edges(allv, bins=bins)] if allv else bins
        )
        for i, (label, vals) in enumerate(hs):
            ax.hist(vals, bins=edges, alpha=0.55 if len(hs) > 1 else 1.0, label=label or None,
                    color=_HIST_COLORS[i % len(_HIST_COLORS)])
        if len(hs) > 1:
            ax.legend()
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
    for xv in vlines:  # red spec-limit lines (LLM/HLM on a histogram)
        ax.axvline(xv, color="#d64545", linestyle="--", linewidth=1.4)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80)
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f'<img class="plot" alt="{escape(title)}" src="data:image/png;base64,{b64}">'


def plotly_js_tag() -> str:
    """A single ``<script>`` carrying the whole plotly runtime — put it once at the top of a page,
    then render every plot with ``include_js=False`` so the ~3.5 MB library is embedded exactly once
    (and loads before any plot div, whatever panel order the plots sit in)."""
    from plotly.offline import get_plotlyjs

    return f"<script>{get_plotlyjs()}</script>"


def _render_plotly(
    kind: str, series: Series | None, x: Sequence[float] | None, y: Sequence[float] | None,
    title: str, xlabel: str, ylabel: str, bins: int, vlines: Sequence[float], include_js: bool = True,
) -> str:
    import plotly.graph_objects as go

    fig = go.Figure()
    if kind == "histogram":
        hs = _hist_series(series, y)
        for i, (label, vals) in enumerate(hs):
            fig.add_histogram(x=vals, name=label or None, nbinsx=bins,
                              opacity=0.55 if len(hs) > 1 else 1.0,
                              marker_color=_HIST_COLORS[i % len(_HIST_COLORS)])
        if len(hs) > 1:
            fig.update_layout(barmode="overlay")
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
    for xv in vlines:  # red spec-limit lines
        fig.add_vline(x=xv, line={"color": "#d64545", "dash": "dash", "width": 1.4})
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel, showlegend=bool(series))
    # Deterministic div id (default is a random UUID) so the same inputs render byte-identical HTML.
    import hashlib

    div_id = "plot-" + hashlib.sha256(
        f"{kind}|{title}|{xlabel}|{ylabel}|{series}|{x}|{y}".encode()
    ).hexdigest()[:16]
    # inline the library only when asked (once per page); other plots on the page reuse that runtime.
    html: str = fig.to_html(
        full_html=False, include_plotlyjs=("inline" if include_js else False), div_id=div_id
    )
    return html
