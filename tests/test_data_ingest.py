"""Ingest delimited/fixed-width data tables into the generic data store (Phase 10) — CSV, TSV, and a
long/tidy table via the source's csv role-map. STDF is not special; this is the general data path."""

from __future__ import annotations

from pathlib import Path

from docusearch import config as cfg
from docusearch import ingest
from docusearch.store import Store


def _cfg(tmp_path: Path, data_dir: Path, extra_source: str = "") -> cfg.Config:
    path = tmp_path / "d.yaml"
    path.write_text(
        f'store_type: "data"\npaths:\n  staging_dir: "{(tmp_path / "s").as_posix()}"\n'
        f'  db_path: "{(tmp_path / "c.db").as_posix()}"\n  tmp_dir: "{(tmp_path / "t").as_posix()}"\n'
        f'sources:\n  - name: data\n    location: "{data_dir.as_posix()}"\n'
        '    include: ["*.csv", "*.tsv"]\n    min_content_chars: 1\n'
        + extra_source
        + 'embed:\n  model: "none"\n',
        encoding="utf-8")
    return cfg.load(path)


def test_ingest_wide_csv_and_tsv_into_data_store(tmp_path: Path) -> None:
    d = tmp_path / "data"
    d.mkdir()
    (d / "meas.csv").write_text("vmin,iddq,site\n0.71,1e-6,1\n0.72,2e-6,1\n0.90,1.5e-6,2\n",
                                encoding="utf-8")
    (d / "log.tsv").write_text("temp\tvolt\n25\t1.1\n30\t1.2\n", encoding="utf-8")
    config = _cfg(tmp_path, d)
    with Store.open(config.paths.db_path) as store:
        result = ingest.run_ingest(config, store)
        cols = {(r["dataset"], r["name"]): r for r in store.data_columns()}
        n_docs = result.documents
        n_data_cols = result.data_columns
    assert n_docs == 2 and n_data_cols == 5  # vmin, iddq, site, temp, volt
    assert cols[("meas", "vmin")]["n"] == 3 and cols[("log", "volt")]["n"] == 2


def test_ingest_long_table_with_role_map(tmp_path: Path) -> None:
    d = tmp_path / "data"
    d.mkdir()
    (d / "results.csv").write_text(
        "test,value,lo,hi,site\nVMIN,0.71,0.70,0.85,1\nVMIN,0.72,0.70,0.85,2\n"
        "IDDQ,1e-6,0,2e-6,1\nIDDQ,2e-6,0,2e-6,2\n", encoding="utf-8")
    extra = ('    csv:\n      label: "test"\n      value: "value"\n'
             '      lo: "lo"\n      hi: "hi"\n      group: "site"\n')
    # rebuild the source with the csv block (only one source, so replace include too)
    path = tmp_path / "d.yaml"
    path.write_text(
        f'store_type: "data"\npaths:\n  staging_dir: "{(tmp_path / "s").as_posix()}"\n'
        f'  db_path: "{(tmp_path / "c.db").as_posix()}"\n  tmp_dir: "{(tmp_path / "t").as_posix()}"\n'
        f'sources:\n  - name: data\n    location: "{d.as_posix()}"\n'
        '    include: ["*.csv"]\n    min_content_chars: 1\n' + extra
        + 'embed:\n  model: "none"\n', encoding="utf-8")
    config = cfg.load(path)
    with Store.open(config.paths.db_path) as store:
        ingest.run_ingest(config, store)
        cols = {r["name"]: r for r in store.data_columns()}
        vmin_id = cols["VMIN"]["id"]
        vals = store.data_values(column_id=vmin_id)
    assert set(cols) == {"VMIN", "IDDQ"}  # pivoted long → per-metric columns
    assert cols["VMIN"]["lo"] == 0.70 and cols["VMIN"]["hi"] == 0.85
    assert [v["value"] for v in vals] == [0.71, 0.72]
    assert [v["grp"] for v in vals] == ["1", "2"]  # group carried per observation


def test_ingest_fixed_width_via_delimiter_config(tmp_path: Path) -> None:
    d = tmp_path / "data"
    d.mkdir()
    (d / "fw.dat").write_text("vmin  iddq  \n0.71  1e-6  \n0.72  2e-6  \n", encoding="utf-8")
    path = tmp_path / "d.yaml"
    path.write_text(
        f'store_type: "data"\npaths:\n  staging_dir: "{(tmp_path / "s").as_posix()}"\n'
        f'  db_path: "{(tmp_path / "c.db").as_posix()}"\n  tmp_dir: "{(tmp_path / "t").as_posix()}"\n'
        f'sources:\n  - name: data\n    location: "{d.as_posix()}"\n'
        '    include: ["*.dat"]\n    min_content_chars: 1\n'
        '    csv:\n      widths: [6, 6]\n' + 'embed:\n  model: "none"\n', encoding="utf-8")
    config = cfg.load(path)
    with Store.open(config.paths.db_path) as store:
        result = ingest.run_ingest(config, store)  # .dat routed to the table writer via widths config
        names = {r["name"] for r in store.data_columns()}
    assert result.data_columns == 2 and names == {"vmin", "iddq"}
