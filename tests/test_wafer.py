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
    ml = svc.mother_lot(ids[0])["html"]
    assert "Lot LOTW" in ml
    tr = svc.production_trend(ids)["html"]
    assert "Production yield trend" in tr and "2 lots" in tr
