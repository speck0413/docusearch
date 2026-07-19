"""Wafer & production analytics (Phase 7 / GATE 7): wafer map + per-wafer roll-up from parsed parts."""

from __future__ import annotations

from docusearch import wafer
from docusearch.stdf import StdfPart, StdfTest


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


def _lot(lot: str, wafers: int, fail_die: tuple[int, int]) -> list[StdfPart]:
    parts = []
    for w in range(1, wafers + 1):
        for x in range(1, 4):
            for y in range(1, 4):
                hb = 5 if (x, y) == fail_die else 1
                parts.append(StdfPart(lot_id=lot, sublot_id="", wafer_id=f"{lot}-W{w:02d}",
                                      x=x, y=y, part_id=f"{x},{y}", head=1, site=1,
                                      hard_bin=hb, soft_bin=hb, passed=(hb == 1), insertion="WS1"))
    return parts


def test_mother_lot_view() -> None:
    html = wafer.mother_lot_html(_lot("LOTA", 4, (2, 2)), backend="matplotlib")
    assert html.startswith("<!doctype html>")
    assert "Lot LOTA" in html and "4 wafers" in html
    assert "lowest wafer" in html and "data:image/png" in html  # yield-per-wafer plot
    assert html.count("LOTA-W0") >= 4  # a table row per wafer


def test_production_trend_view() -> None:
    lots = [("LOTA", _lot("LOTA", 2, (2, 2))),   # 8/9 per wafer
            ("LOTB", _lot("LOTB", 2, (1, 1)))]   # 8/9 per wafer, different failing die
    html = wafer.production_trend_html(lots, backend="matplotlib")
    assert "Production yield trend" in html and "2 lots" in html
    assert "data:image/png" in html and "LOTA" in html and "LOTB" in html


def test_wafer_views_are_deterministic() -> None:
    parts = _lot("LOTA", 3, (2, 2))
    assert wafer.mother_lot_html(parts) == wafer.mother_lot_html(parts)
    assert wafer.wafer_map_html(parts) == wafer.wafer_map_html(parts)


def _wafer_stdf(path: object, wafer: str, fail_die: tuple[int, int]) -> None:
    from harness.stdf_synth import StdfBuilder
    b = StdfBuilder().far().mir(lot_id="LOTW", job_nam="WSORT", test_cod="WS1")
    b.wir(wafer)
    i = 0
    for x in range(1, 4):
        for y in range(1, 4):
            hb = 5 if (x, y) == fail_die else 1
            b.pir()
            b.ptr(1000, "VMIN", 0.72 if hb == 1 else 0.60, fail=(hb != 1))
            b.prr(part_id=str(i + 1), hard_bin=hb, x=x, y=y)
            i += 1
    b.mrr().write(path)  # type: ignore[arg-type]


def test_wafer_service_over_ingested_stdf(tmp_path) -> None:  # type: ignore[no-untyped-def]

    from docusearch import config as cfg
    from docusearch import ingest
    from docusearch.server import Service
    from docusearch.store import Store

    d = tmp_path / "ate"
    d.mkdir()
    _wafer_stdf(d / "w1.stdf", "W01", (2, 2))
    _wafer_stdf(d / "w2.stdf", "W02", (1, 1))
    path = tmp_path / "c.yaml"
    path.write_text(
        f'store_type: "data"\npaths:\n  staging_dir: "{(tmp_path / "s").as_posix()}"\n'
        f'  db_path: "{(tmp_path / "c.db").as_posix()}"\n  tmp_dir: "{(tmp_path / "t").as_posix()}"\n'
        f'sources:\n  - name: ate\n    location: "{d.as_posix()}"\n'
        '    include: ["*.stdf"]\n    min_content_chars: 1\n    insertion: "WS1"\n'
        'embed:\n  model: "none"\n', encoding="utf-8")
    config = cfg.load(path)
    with Store.open(config.paths.db_path) as store:
        ingest.run_ingest(config, store)
        ids = sorted(int(r["id"]) for r in
                     store._conn.execute("SELECT id FROM documents WHERE fmt='stdf'").fetchall())  # noqa: SLF001
    svc = Service(config)
    wm = svc.wafer_map(ids[0], wafer_id="W01")["html"]
    assert "Wafer W01" in wm and 'class="die' in wm
    # a parametric map for a test number with no data → service flags empty + a one-line note
    absent = svc.wafer_map(ids[0], wafer_id="W01", test_num=999999)
    assert absent.get("empty") is True and "finite" in absent["note"]
    assert "empty" not in svc.wafer_map(ids[0], wafer_id="W01")  # a real map has no flag
    ml = svc.mother_lot(ids[0])["html"]
    assert "Lot LOTW" in ml
    tr = svc.production_trend(ids)["html"]
    assert "Production yield trend" in tr and "2 lots" in tr


def test_parametric_wafer_map() -> None:
    # a gradient of VMIN across the wafer: value rises with x
    tests, parts = [], []
    for i, (x, y) in enumerate((x, y) for x in range(1, 4) for y in range(1, 4)):
        pid = str(i + 1)
        tests.append(StdfTest(test_num=1000, test_txt="VMIN", result=0.70 + 0.01 * x, head=1,
                              site=1, passed=True, part_id=pid, conditions={}, units="V"))
        parts.append(StdfPart(lot_id="L", sublot_id="", wafer_id="W01", x=x, y=y, part_id=pid,
                              head=1, site=1, hard_bin=1, soft_bin=1, passed=True, insertion="WS1"))
    html = wafer.param_wafer_map_html(tests, parts, 1000, wafer_id="W01")
    assert "VMIN" in html and "parametric map" in html and 'class="die"' in html
    assert "background:#" in html  # heat-coloured dies
    missing = wafer.param_wafer_map_html(tests, parts, 9999)  # a test number with no data
    assert "no parametric map" in missing


def test_phase7_redteam_regressions() -> None:
    import re
    p3 = [StdfPart("L", "", "W1", x, 1, f"p{x}", 1, 1, 1, 1, True, "FT") for x in (1, 2, 3)]
    # H1: every result non-finite → clean message, no KeyError
    allnan = [StdfTest(1000, "V", float("nan"), 1, 1, True, f"p{x}", {}, units="V") for x in (1, 2, 3)]
    assert "no finite" in wafer.param_wafer_map_html(allnan, p3, 1000, wafer_id="W1")
    # M1: a NaN FIRST must not corrupt the rest of the heatmap gradient
    mixed = [StdfTest(1000, "V", float("nan"), 1, 1, True, "p1", {}),
             StdfTest(1000, "V", 1.0, 1, 1, True, "p2", {}),
             StdfTest(1000, "V", 3.0, 1, 1, True, "p3", {})]
    m1 = wafer.param_wafer_map_html(mixed, p3, 1000, wafer_id="W1")
    bgs = re.findall(r"background:(#[0-9a-f]{6})", m1)
    assert len(set(bgs)) > 1  # two finite values → two different colours
    # M3: a spec-legal but absurd coordinate span is refused, not turned into a giant CSS grid
    big = [StdfPart("L", "", "W1", 0, 0, "a", 1, 1, 1, 1, True, "FT"),
           StdfPart("L", "", "W1", 40000, 0, "b", 1, 1, 1, 1, True, "FT")]
    assert "span exceeds" in wafer.wafer_map_html(big, wafer_id="W1")
    assert "repeat(40001" not in wafer.wafer_map_html(big, wafer_id="W1")
    # M4: a `$` in lot_id must not crash the matplotlib title
    assert "data:image/png" in wafer.mother_lot_html(
        [StdfPart("LOT$$$$", "", "W1", 1, 1, "p", 1, 1, 1, 1, True, "FT")], backend="matplotlib")
    # L1: subtitle is single-escaped (no &amp;amp;)
    named = [StdfTest(1000, 'V&"x"', 1.0, 1, 1, True, "p1", {})]
    assert "&amp;amp;" not in wafer.param_wafer_map_html(named, p3[:1], 1000, wafer_id="W1")
    # attack9: an empty-state page carries a machine-readable note; a real report does not
    allnan_html = wafer.param_wafer_map_html(allnan, p3, 1000, wafer_id="W1")
    note = wafer.empty_note(allnan_html)
    assert note is not None and "finite" in note and "-->" not in note  # comment-safe
    good = [StdfTest(1000, "V", 1.0, 1, 1, True, "p1", {}),
            StdfTest(1000, "V", 2.0, 1, 1, True, "p2", {}),
            StdfTest(1000, "V", 3.0, 1, 1, True, "p3", {})]
    assert wafer.empty_note(wafer.param_wafer_map_html(good, p3, 1000, wafer_id="W1")) is None
