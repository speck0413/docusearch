"""Regression tests for the GATE 6 red-team findings (3 HIGH, 4 MEDIUM)."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
from harness.stdf_synth import sample_conditioned_run

from docusearch import analytics, ingest, report_export
from docusearch import config as cfg
from docusearch.server import Service
from docusearch.store import Store


def _spec(body: str) -> dict:  # type: ignore[type-arg]
    return {"title": "R", "sections": [{"heading": "s", "body": body}], "evidence": set()}


# H1 — qq with <2 series raises a clean ValueError, never an IndexError/500.
def test_h1_qq_needs_two_series() -> None:
    with pytest.raises(ValueError, match="two"):
        analytics.render_plot("qq", y=[1, 2, 3])
    with pytest.raises(ValueError, match="two"):
        analytics.render_plot("qq", series=[("only", [1, 2])])


# H2 — an analytics tool on a NON-STDF doc in a mixed store fails clean (ValueError), no pystdf crash.
def test_h2_non_stdf_doc_fails_clean(tmp_path: Path) -> None:
    root = tmp_path / "d"
    root.mkdir()
    (root / "a.html").write_text("<body><h1>Hi</h1><p>plain prose here</p></body>", encoding="utf-8")
    p = tmp_path / "c.yaml"
    p.write_text(
        f'paths:\n  staging_dir: "{(tmp_path / "s").as_posix()}"\n'
        f'  db_path: "{(tmp_path / "c.db").as_posix()}"\n  tmp_dir: "{(tmp_path / "t").as_posix()}"\n'
        f'sources:\n  - name: d\n    location: "{root.as_posix()}"\n    min_content_chars: 1\n'
        'embed:\n  model: "none"\n',
        encoding="utf-8",
    )
    config = cfg.load(p)
    with Store.open(config.paths.db_path) as store:
        ingest.run_ingest(config, store)
        doc_id = next(iter(store.document_path_to_id().values()))
    with pytest.raises(ValueError, match="not a readable STDF"):
        Service(config).stdf_plot(doc_id, 1000)


# H3 — a spreadsheet cell starting with '=' is stored as TEXT (formula injection neutralized).
def test_h3_xlsx_formula_injection_neutralized(tmp_path: Path) -> None:
    from openpyxl import load_workbook

    data = report_export.export_report(fmt="xlsx", **_spec("=1+2 and =cmd|'/c calc'!A1"))  # type: ignore[arg-type]
    p = tmp_path / "r.xlsx"
    p.write_bytes(data)
    wb = load_workbook(p)
    ws = wb.active
    for row in ws.iter_rows():
        for cell in row:
            assert cell.data_type != "f"  # never a live formula
    assert report_export.xlsx_cell("=danger()") == "'=danger()"
    assert report_export.xlsx_cell("safe") == "safe"


# M1 — a NUL / control char in the body doesn't crash docx/xlsx export.
@pytest.mark.parametrize("fmt", ["docx", "xlsx", "pdf", "pptx"])
def test_m1_control_chars_do_not_crash_export(tmp_path: Path, fmt: str) -> None:
    data = report_export.export_report(fmt=fmt, **_spec("bad \x00 \x07 char here"))  # type: ignore[arg-type]
    assert isinstance(data, bytes) and len(data) > 300
    assert report_export.xml_safe("a\x00b\x07c") == "abc"


# M2 — stdf.granularity=part emits per-part rollup chunks.
def test_m2_granularity_part_emits_part_chunks(tmp_path: Path) -> None:
    root = tmp_path / "d"
    root.mkdir()
    sample_conditioned_run(root / "r.stdf")
    p = tmp_path / "c.yaml"
    p.write_text(
        f'paths:\n  staging_dir: "{(tmp_path / "s").as_posix()}"\n'
        f'  db_path: "{(tmp_path / "c.db").as_posix()}"\n  tmp_dir: "{(tmp_path / "t").as_posix()}"\n'
        f'sources:\n  - name: d\n    location: "{root.as_posix()}"\n    include: ["*.stdf"]\n'
        '    min_content_chars: 1\n    insertion: "WS1"\n'
        'embed:\n  model: "none"\nstdf:\n  granularity: "part"\n',
        encoding="utf-8",
    )
    config = cfg.load(p)
    with Store.open(config.paths.db_path) as store, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ingest.run_ingest(config, store)
        parts = store._conn.execute("SELECT COUNT(*) FROM chunks WHERE kind='part'").fetchone()[0]  # noqa: SLF001
    assert parts == 2  # two parts in the sample -> two rollup chunks


# M4 — the plotly backend renders byte-identical for identical inputs (deterministic div id).
def test_m4_plotly_is_deterministic() -> None:
    a = analytics.render_plot("linear", x=[1, 2, 3], y=[1, 4, 9], title="t", backend="plotly")
    b = analytics.render_plot("linear", x=[1, 2, 3], y=[1, 4, 9], title="t", backend="plotly")
    assert a == b
