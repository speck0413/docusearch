"""Wafer & production analytics (Phase 7 / GATE 7) — a specialized engine to the side of the STDF
per-test tools, built on the same parsed parts. STDF already records each part's wafer / x / y / bin
(:class:`docusearch.stdf.StdfPart`), so a **wafer map** (die grid coloured by pass-fail or bin), a
**mother-lot** view (every wafer's yield in a lot), and a **long-term production trend** (yield across
lots over time) fall out of that. Rendering is a self-contained themed HTML fragment (a CSS die grid
+ the shared plot engine), so it embeds in a report exactly like the STDF dashboards.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from html import escape

from . import analytics, report
from .stdf import StdfPart

# a categorical palette for soft-bin colouring (bin 1 = pass green; others cycle)
_BIN_COLORS = ("#3cb371", "#d64545", "#4c8dff", "#e6a020", "#c060d0", "#48cae4", "#9fb6d6", "#e07b39")


@dataclass
class WaferStat:
    """One wafer's roll-up: die count, pass count, yield %, and its coordinate extent."""

    wafer: str
    total: int
    passed: int
    x_min: int
    x_max: int
    y_min: int
    y_max: int

    @property
    def yield_pct(self) -> float:
        return 100.0 * self.passed / self.total if self.total else 0.0


def _mapped(parts: Sequence[StdfPart]) -> list[StdfPart]:
    """Parts that carry die coordinates (x and y set) — the ones that place on a wafer map."""
    return [p for p in parts if p.x is not None and p.y is not None]


def wafer_stats(parts: Sequence[StdfPart]) -> list[WaferStat]:
    """Per-wafer roll-up over the coordinate-bearing parts, in first-seen wafer order."""
    order: list[str] = []
    by: dict[str, list[StdfPart]] = {}
    for p in _mapped(parts):
        if p.wafer_id not in by:
            by[p.wafer_id] = []
            order.append(p.wafer_id)
        by[p.wafer_id].append(p)
    out: list[WaferStat] = []
    for w in order:
        dies = by[w]
        xs = [int(p.x) for p in dies]  # type: ignore[arg-type]
        ys = [int(p.y) for p in dies]  # type: ignore[arg-type]
        out.append(WaferStat(
            wafer=w, total=len(dies), passed=sum(1 for p in dies if p.passed),
            x_min=min(xs), x_max=max(xs), y_min=min(ys), y_max=max(ys),
        ))
    return out


def _die_grid(dies: Sequence[StdfPart], color_by: str) -> tuple[str, dict[int, str]]:
    """The CSS-grid of die cells + the {soft_bin: colour} legend used (empty when colouring by pass)."""
    xs = [int(p.x) for p in dies]  # type: ignore[arg-type]
    ys = [int(p.y) for p in dies]  # type: ignore[arg-type]
    x_min, y_min = min(xs), min(ys)
    ncols = max(xs) - x_min + 1
    bins = sorted({p.soft_bin for p in dies})
    bin_color = {b: _BIN_COLORS[i % len(_BIN_COLORS)] for i, b in enumerate(bins)}
    cells = []
    for p in dies:
        col = int(p.x) - x_min + 1  # type: ignore[arg-type]
        row = int(p.y) - y_min + 1  # type: ignore[arg-type]
        if color_by == "softbin":
            style = f"background:{bin_color[p.soft_bin]};"
            cls = "die"
        else:
            style = ""
            cls = "die pass" if p.passed else "die fail"
        title = f"({p.x},{p.y}) HB{p.hard_bin} SB{p.soft_bin} {'PASS' if p.passed else 'FAIL'}"
        cells.append(
            f'<div class="{cls}" style="grid-column:{col};grid-row:{row};{style}" '
            f'title="{escape(title)}"></div>'
        )
    grid = (f'<div class="wafermap" style="grid-template-columns:repeat({ncols},13px)">'
            + "".join(cells) + "</div>")
    return grid, (bin_color if color_by == "softbin" else {})


def wafer_map_html(
    parts: Sequence[StdfPart], *, wafer_id: str = "", color_by: str = "pass",
) -> str:
    """A **wafer map**: a die grid at each part's (x, y), coloured by **pass/fail** (default) or by
    **soft bin** (``color_by="softbin"``), with the wafer's yield + a bin legend. ``wafer_id`` picks
    the wafer (else the first one present). Deterministic; a self-contained themed page."""
    dies_all = _mapped(parts)
    if not dies_all:
        return report.themed_page("Wafer map", '<p class="stats">no die-coordinate data</p>',
                                  subtitle="no wafer map available")
    stats = {s.wafer: s for s in wafer_stats(parts)}
    wafer = wafer_id or next(iter(stats))
    dies = [p for p in dies_all if p.wafer_id == wafer]
    if not dies:
        return report.themed_page(
            f"Wafer {escape(wafer)}",
            f'<p class="stats">no dies for wafer {escape(wafer)} (have: '
            f'{escape(", ".join(stats))})</p>', subtitle="wafer not found")
    st = stats[wafer]
    grid, legend = _die_grid(dies, color_by)
    legend_html = ""
    if legend:
        legend_html = '<p class="stats">soft bins: ' + " · ".join(
            f'<span class="tag" style="border-color:{c}">{b}</span>' for b, c in legend.items()
        ) + "</p>"
    else:
        legend_html = ('<p class="stats">'
                       '<span class="tag" style="border-color:#3cb371">pass</span> '
                       '<span class="tag" style="border-color:#d64545">fail</span></p>')
    body = (
        f'<section class="acard"><h2>Wafer {escape(wafer)} — {st.yield_pct:.1f}% yield '
        f'({st.passed}/{st.total})</h2>'
        f'<p class="stats">{st.x_max - st.x_min + 1} × {st.y_max - st.y_min + 1} die grid · '
        f'coloured by {escape(color_by)}</p>{legend_html}{grid}</section>'
    )
    return report.themed_page(f"Wafer map — {wafer}", body,
                              subtitle=f"{st.total} dies · {st.yield_pct:.1f}% yield")


def _yield_table(rows: Sequence[tuple[str, int, int, float]]) -> str:
    body = "".join(
        f"<tr><td>{escape(label)}</td><td>{n}</td><td>{p}</td><td>{y:.1f}%</td></tr>"
        for label, n, p, y in rows
    )
    return ('<div class="scroll"><table class="grid"><thead><tr><th>unit</th><th>dies</th>'
            f"<th>pass</th><th>yield</th></tr></thead><tbody>{body}</tbody></table></div>")


def mother_lot_html(parts: Sequence[StdfPart], *, backend: str = "matplotlib") -> str:
    """A **mother-lot** view: every wafer's yield across the lot — a wafer-by-wafer yield trend plot
    + a per-wafer table + the pooled lot yield. Highlights a low wafer against the lot."""
    stats = wafer_stats(parts)
    if not stats:
        return report.themed_page("Mother lot", '<p class="stats">no wafer data</p>')
    lot = next((p.lot_id for p in parts if p.lot_id), "")
    ys = [s.yield_pct for s in stats]
    total, passed = sum(s.total for s in stats), sum(s.passed for s in stats)
    lot_y = 100.0 * passed / total if total else 0.0
    plot = analytics.render_plot(
        "linear", x=list(range(len(stats))), y=ys, title=f"Yield per wafer — lot {lot}",
        xlabel="wafer #", ylabel="yield %", backend=backend, vlines=(),
    )
    table = _yield_table([(s.wafer, s.total, s.passed, s.yield_pct) for s in stats])
    worst = min(stats, key=lambda s: s.yield_pct)
    body = (
        f'<section class="acard"><h2>Lot {escape(lot)} — {lot_y:.1f}% yield '
        f'({passed}/{total} across {len(stats)} wafers)</h2>'
        f'<p class="stats">lowest wafer: {escape(worst.wafer)} at {worst.yield_pct:.1f}%</p>'
        f"{plot}{table}</section>"
    )
    return report.themed_page(f"Mother lot — {lot}", body,
                              subtitle=f"{len(stats)} wafers · {lot_y:.1f}% lot yield")


def production_trend_html(
    lots: Sequence[tuple[str, Sequence[StdfPart]]], *, backend: str = "matplotlib"
) -> str:
    """A **long-term production trend**: yield across an ordered list of ``(label, parts)`` — one
    point per lot/date — to spot drift over time. ``label`` is the lot id or a date."""
    points = []
    for label, parts in lots:
        mapped = _mapped(parts)
        n = len(mapped)
        if n:
            points.append((label, 100.0 * sum(1 for p in mapped if p.passed) / n, n))
    if not points:
        return report.themed_page("Production trend", '<p class="stats">no lots with wafer data</p>')
    plot = analytics.render_plot(
        "linear", x=list(range(len(points))), y=[y for _, y, _ in points],
        title="Yield trend across lots", xlabel="lot (time →)", ylabel="yield %", backend=backend,
    )
    table = _yield_table([(lbl, n, round(n * y / 100), y) for lbl, y, n in points])
    body = (f'<section class="acard"><h2>Production yield trend — {len(points)} lots</h2>'
            f"{plot}{table}</section>")
    return report.themed_page("Production trend", body, subtitle=f"{len(points)} lots over time")
