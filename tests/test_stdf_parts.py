"""STDF part traceability + per-insertion yield (R-STDF-2): unique part identity, first-pass vs
final yield, and part progression across insertions (WS1, WS1-RT)."""

from __future__ import annotations

from pathlib import Path

from harness.stdf_synth import sample_wafer_flow

from docusearch import stdf, stdf_analytics


def _parts(tmp_path: Path) -> list[stdf.StdfPart]:
    ws1, ws1rt = sample_wafer_flow(tmp_path / "ws1.stdf", tmp_path / "ws1rt.stdf")
    run1 = stdf.parse_stdf_tests(ws1.read_bytes())
    run2 = stdf.parse_stdf_tests(ws1rt.read_bytes())
    assert run1.insertion == "WS1" and run2.insertion == "WS1-RT"  # from MIR TEST_COD
    return run1.parts + run2.parts


def test_part_key_is_unique_wafer_xy(tmp_path: Path) -> None:
    parts = _parts(tmp_path)
    p = next(p for p in parts if p.x == 1 and p.y == 1)
    assert p.key(("lot", "sublot", "wafer", "x", "y")) == ("LOTW", "01", "W01", "1", "1")
    # a packaged-part key falls back / uses part_id
    assert p.key(("part_id",)) == ("1",)


def test_first_pass_vs_final_yield_per_insertion(tmp_path: Path) -> None:
    rows = stdf_analytics.insertion_yield(_parts(tmp_path))
    ws1 = next(r for r in rows if r["insertion"] == "WS1")
    # WS1: 4 die, (2,2) fails first touchdown then passes on the intra-WS1 retest
    assert ws1["total"] == 4
    assert ws1["first_pass"] == 3 and ws1["first_pass_yield"] == 75.0
    assert ws1["final_pass"] == 4 and ws1["final_yield"] == 100.0
    assert ws1["retested"] == 1
    ws1rt = next(r for r in rows if r["insertion"] == "WS1-RT")
    assert ws1rt["total"] == 1 and ws1rt["final_pass"] == 1


def test_trace_part_across_insertions(tmp_path: Path) -> None:
    order, trace = stdf_analytics.trace_parts(_parts(tmp_path))
    assert order == ["WS1", "WS1-RT"]  # first-seen insertion order
    die_21 = trace[("LOTW", "01", "W01", "2", "1")]  # die (2,1): WS1 pass, retouched in WS1-RT
    assert die_21["WS1"].passed and die_21["WS1-RT"].passed
    die_22 = trace[("LOTW", "01", "W01", "2", "2")]  # die (2,2): only WS1 (final touchdown = pass)
    assert die_22["WS1"].passed and "WS1-RT" not in die_22


def test_yield_and_trace_html(tmp_path: Path) -> None:
    parts = _parts(tmp_path)
    y = stdf_analytics.insertion_yield_html(parts)
    assert "Yield per insertion" in y and "WS1" in y and "75.0%" in y and "100.0%" in y
    t = stdf_analytics.part_trace_html(parts)
    assert "Part progression" in t and "PASS" in t
