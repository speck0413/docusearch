"""STDF analytics tools via the Service (R-STDF-2): plot/audit/site/trend + the general plot_data,
over ingested STDF documents."""

from __future__ import annotations

from pathlib import Path

from harness.stdf_synth import StdfBuilder, sample_conditioned_run

from docusearch import config as cfg
from docusearch import ingest
from docusearch.server import Service
from docusearch.store import Store


def _run_file(path: Path, vmins: list[float], sites: list[int] | None = None) -> None:
    b = StdfBuilder().far().mir(lot_id="L", job_nam="P")
    sites = sites or [1] * len(vmins)
    for i, (v, site) in enumerate(zip(vmins, sites, strict=True)):
        b.pir(site=site)
        b.ptr(1000, "VMIN", v, site=site, fail=(v < 0.70))
        b.prr(part_id=str(i + 1), hard_bin=1 if v >= 0.70 else 5, site=site)
    b.mrr()
    b.write(path)


def _service(tmp_path: Path) -> tuple[Service, dict[str, int]]:
    root = tmp_path / "data"
    root.mkdir()
    sample_conditioned_run(root / "a.stdf")  # doc with VMIN/VMAX/IDDQ + conditions
    _run_file(root / "b.stdf", [0.71, 0.72, 0.90, 0.91], sites=[1, 1, 2, 2])
    _run_file(root / "c.stdf", [0.75, 0.76, 0.92, 0.93], sites=[1, 1, 2, 2])
    path = tmp_path / "d.yaml"
    path.write_text(
        f'store_type: "data"\npaths:\n  staging_dir: "{(tmp_path / "s").as_posix()}"\n'
        f'  db_path: "{(tmp_path / "c.db").as_posix()}"\n  tmp_dir: "{(tmp_path / "t").as_posix()}"\n'
        f'sources:\n  - name: ate\n    location: "{root.as_posix()}"\n'
        '    include: ["*.stdf"]\n    min_content_chars: 1\n'
        'embed:\n  model: "none"\n',
        encoding="utf-8",
    )
    config = cfg.load(path)
    with Store.open(config.paths.db_path) as store:
        ingest.run_ingest(config, store)
        ids = {
            Path(str(r["path"])).stem: int(r["id"])
            for r in store._conn.execute("SELECT id, path FROM documents").fetchall()  # noqa: SLF001
        }
    return Service(config), ids


def test_stdf_plot_and_site_and_trend(tmp_path: Path) -> None:
    svc, ids = _service(tmp_path)
    plot = svc.stdf_plot(ids["b"], 1000, kind="histogram")
    assert "data:image/png" in plot["html"] and "VMIN" in plot["html"]

    site = svc.stdf_site_compare(ids["b"], 1000)
    assert "site-to-site" in site["html"]

    trend = svc.stdf_trend([ids["b"], ids["c"]], 1000, stat="mean")
    assert "trend" in trend["html"] and "data:image/png" in trend["html"]


def test_stdf_audit_drilldown(tmp_path: Path) -> None:
    svc, ids = _service(tmp_path)
    audit = svc.stdf_audit(ids["b"], ids["c"])
    assert "STDF audit" in audit["html"] and "Yield" in audit["html"]
    assert "<details>" in audit["html"]  # drill-down


def test_plot_data_general(tmp_path: Path) -> None:
    svc, _ = _service(tmp_path)
    out = svc.plot_data(kind="histogram", y=[1, 2, 2, 3, 3, 3], title="from a column")
    assert "data:image/png" in out["html"]
    # plotly backend on demand
    out2 = svc.plot_data(kind="linear", x=[1, 2, 3], y=[1, 4, 9], backend="plotly")
    assert "plotly" in out2["html"].lower()


def test_stdf_tools_bad_doc_id(tmp_path: Path) -> None:
    import pytest

    svc, _ = _service(tmp_path)
    with pytest.raises(ValueError, match="STDF document"):
        svc.stdf_plot(99999, 1000)
