"""Query + plot the generic data store via the Service (Phase 10) — the same access an MCP agent,
a REST/web UI, or a script uses. Works on any ingested CSV/table, not just STDF."""

from __future__ import annotations

from pathlib import Path

from docusearch import config as cfg
from docusearch import ingest
from docusearch.server import Service
from docusearch.store import Store


def _service(tmp_path: Path) -> Service:
    d = tmp_path / "data"
    d.mkdir()
    # a long/tidy CSV so we get spec limits + a group column
    (d / "results.csv").write_text(
        "test,value,lo,hi,site\n"
        + "".join(f"VMIN,{0.70 + 0.001 * i},0.68,0.86,{1 + i % 2}\n" for i in range(40)),
        encoding="utf-8")
    path = tmp_path / "d.yaml"
    path.write_text(
        f'store_type: "data"\npaths:\n  staging_dir: "{(tmp_path / "s").as_posix()}"\n'
        f'  db_path: "{(tmp_path / "c.db").as_posix()}"\n  tmp_dir: "{(tmp_path / "t").as_posix()}"\n'
        f'sources:\n  - name: data\n    location: "{d.as_posix()}"\n'
        '    include: ["*.csv"]\n    min_content_chars: 1\n'
        '    csv:\n      label: "test"\n      value: "value"\n      lo: "lo"\n      hi: "hi"\n'
        '      group: "site"\n' + 'embed:\n  model: "none"\n', encoding="utf-8")
    config = cfg.load(path)
    with Store.open(config.paths.db_path) as store:
        ingest.run_ingest(config, store)
    return Service(config)


def test_data_columns_and_values(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    cols = svc.data_columns()["columns"]
    vmin = next(c for c in cols if c["name"] == "VMIN")
    assert vmin["n"] == 40 and vmin["lo"] == 0.68 and vmin["hi"] == 0.86
    vals = svc.data_values(vmin["id"])["values"]
    assert len(vals) == 40 and vals[0]["group"] in ("1", "2")


def test_data_plot_and_by_group(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    cid = next(c for c in svc.data_columns()["columns"] if c["name"] == "VMIN")["id"]
    plot = svc.data_plot(cid, kind="histogram", backend="plotly")
    assert "column" in plot and plot["column"] == "VMIN" and plot["n"] == 40
    assert plot["stats"]["n"] == 40 and "cpk" in plot["capability"]
    assert "d64545" in plot["html"]  # red lo/hi limit lines drawn
    grouped = svc.data_plot(cid, kind="whisker", backend="plotly", by_group=True)
    assert "site" not in grouped["column"] and "plotly" in grouped["html"].lower()


def test_data_plot_bad_column(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    import pytest
    with pytest.raises(ValueError, match="no data column"):
        svc.data_plot(999999)


def test_phase10_redteam_service_and_report(tmp_path: Path) -> None:
    import pytest

    from docusearch.cli import _save_report
    svc = _service(tmp_path)
    # H2: an out-of-int64 column_id is a clean ValueError (→ MCP DATA error / REST 404), not an
    # uncaught OverflowError; consistent with data_plot's "no data column" error
    with pytest.raises(ValueError, match="no data column"):
        svc.data_values(2**64)
    with pytest.raises(ValueError, match="no data column"):
        svc.data_values(-(2**70))

    # M1: a NULL value (a NaN written via the public Store API) must not crash data_plot
    cols = svc.data_columns()["columns"]
    vmin_id = next(c["id"] for c in cols if c["name"] == "VMIN")
    with Store.open(svc.config.paths.db_path) as store:
        cid = store.add_data_column(doc_id=1, dataset="x", name="withnan", kind="numeric",
                                    units="", lo=None, hi=None, values=[1.0, float("nan"), 3.0])
    assert svc.data_plot(cid, kind="histogram")["n"] == 2          # the NULL value skipped, no crash
    assert svc.data_plot(vmin_id, kind="whisker")["html"]          # ungrouped whisker still fine

    # H1: an auto-derived report filename from a hostile column name stays under tmp_dir/reports
    from types import SimpleNamespace
    fake = SimpleNamespace(paths=SimpleNamespace(tmp_dir=str(tmp_path / "rep")))
    p = _save_report(fake, None, "data_ds_../../../../etc/PWNED_histogram", "<html>x</html>")
    assert p.parent == (tmp_path / "rep" / "reports") and "/" not in p.name and ".." not in p.name
