"""Structured STDF data store + non-AI REST endpoints (R-STDF-2): numeric results + parts land in
queryable tables a thin web UI can hit directly, no AI."""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from pathlib import Path

import pytest
from harness.stdf_synth import sample_wafer_flow

from docusearch import config as cfg
from docusearch import ingest
from docusearch.server import create_app
from docusearch.store import Store

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from fastapi.testclient import TestClient


@pytest.fixture
def data_client(tmp_path: Path) -> Iterator[tuple[TestClient, cfg.Config]]:
    root = tmp_path / "data"
    root.mkdir()
    sample_wafer_flow(root / "ws1.stdf", root / "ws1rt.stdf")
    path = tmp_path / "d.yaml"
    path.write_text(
        f'store_type: "data"\npaths:\n  staging_dir: "{(tmp_path / "s").as_posix()}"\n'
        f'  db_path: "{(tmp_path / "c.db").as_posix()}"\n  tmp_dir: "{(tmp_path / "t").as_posix()}"\n'
        f'sources:\n  - name: ate\n    location: "{root.as_posix()}"\n'
        '    include: ["*.stdf"]\n    min_content_chars: 1\n    insertion: "WS1"\n'
        'embed:\n  model: "none"\n',
        encoding="utf-8",
    )
    config = cfg.load(path)
    with Store.open(config.paths.db_path) as store:
        ingest.run_ingest(config, store)
    yield TestClient(create_app(config)), config


def test_structured_tables_populated(data_client: tuple[TestClient, cfg.Config]) -> None:
    _, config = data_client
    with Store.open(config.paths.db_path) as store:
        assert store.count_stdf_results() > 0
        parts = store.stdf_parts_all()
        assert len(parts) >= 5  # 4 die + 1 intra-retest + 1 in WS1-RT
        assert any(p["wafer"] == "W01" and p["x"] == 2 and p["y"] == 2 for p in parts)


def test_data_endpoints_no_ai(data_client: tuple[TestClient, cfg.Config]) -> None:
    client, _ = data_client
    tests = client.get("/v1/data/stdf/tests").json()["tests"]
    assert any(t["test_txt"] == "VMIN_core" for t in tests)

    tn = next(t["test_num"] for t in tests if t["test_txt"] == "VMIN_core")
    results = client.get("/v1/data/stdf/results", params={"test_num": tn}).json()["results"]
    assert len(results) >= 5 and all("result" in r for r in results)

    ins = client.get("/v1/data/stdf/yield").json()["insertions"]
    # source insertion "WS1" applies to both files, so all parts fall under WS1 here
    assert any(row["first_pass_yield"] >= 0 for row in ins)


def test_plot_from_data_endpoint(data_client: tuple[TestClient, cfg.Config]) -> None:
    client, _ = data_client
    resp = client.post("/v1/data/plot", json={"kind": "histogram", "y": [1, 2, 2, 3, 3, 3], "title": "t"})
    assert resp.status_code == 200 and "data:image/png" in resp.json()["html"]
    bad = client.post("/v1/data/plot", json={"kind": "nope", "y": [1, 2]})
    assert bad.status_code == 400
