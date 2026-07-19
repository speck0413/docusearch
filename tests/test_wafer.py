"""Wafer & production analytics (Phase 7 / GATE 7): wafer map + per-wafer roll-up from parsed parts."""

from __future__ import annotations

from docusearch import wafer
from docusearch.stdf import StdfPart


def _part(w: str, x: int, y: int, hb: int, sb: int) -> StdfPart:
    return StdfPart(lot_id="LOT", sublot_id="", wafer_id=w, x=x, y=y, part_id=f"{x},{y}",
                    head=1, site=1, hard_bin=hb, soft_bin=sb, passed=(hb == 1), insertion="WS1")


def _wafer(w: str) -> list[StdfPart]:
    # a 3×3 wafer: the (2,2) centre die fails (hard bin 5), the rest pass
    return [_part(w, x, y, 5 if (x, y) == (2, 2) else 1, 5 if (x, y) == (2, 2) else 1)
            for x in range(1, 4) for y in range(1, 4)]


def test_wafer_stats_rollup() -> None:
    parts = _wafer("W01") + _wafer("W02")
    stats = {s.wafer: s for s in wafer.wafer_stats(parts)}
    assert set(stats) == {"W01", "W02"}
    w1 = stats["W01"]
    assert w1.total == 9 and w1.passed == 8
    assert round(w1.yield_pct, 1) == 88.9
    assert (w1.x_min, w1.x_max, w1.y_min, w1.y_max) == (1, 3, 1, 3)


def test_wafer_map_renders_die_grid_and_yield() -> None:
    html = wafer.wafer_map_html(_wafer("W01"), wafer_id="W01")
    assert html.startswith("<!doctype html>")
    assert "Wafer W01" in html and "88.9% yield" in html
    assert html.count('class="die pass"') == 8 and html.count('class="die fail"') == 1
    assert 'class="wafermap"' in html and "grid-column:2;grid-row:2" in html  # the centre die placed


def test_wafer_map_softbin_colouring_and_missing() -> None:
    html = wafer.wafer_map_html(_wafer("W01"), wafer_id="W01", color_by="softbin")
    assert "coloured by softbin" in html and "soft bins:" in html
    # a wafer that isn't present is reported cleanly, not crashed
    absent = wafer.wafer_map_html(_wafer("W01"), wafer_id="NOPE")
    assert "wafer not found" in absent
    # parts with no coordinates → no map
    flat = [StdfPart("L", "", "", None, None, "1", 1, 1, 1, 1, True, "FT")]
    assert "no die-coordinate data" in wafer.wafer_map_html(flat)
