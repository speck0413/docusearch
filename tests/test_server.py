"""FastAPI REST surface over the service layer (§10, R-API-1/2)."""

from __future__ import annotations

import hashlib
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from docusearch import config as cfg
from docusearch import ingest
from docusearch.server import create_app
from docusearch.store import Store

with warnings.catch_warnings():  # starlette TestClient warns about httpx2 at import
    warnings.simplefilter("ignore")
    from fastapi.testclient import TestClient

_PNG = b"\x89PNG\r\n\x1a\n" + b"fake-image-bytes-for-serving"
_PNG_SHA = hashlib.sha256(_PNG).hexdigest()


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    root = tmp_path / "docs"
    root.mkdir()
    (root / "spi.html").write_text(
        "<html><head><title>SPI</title></head><body><h1>SPI</h1>"
        "<p>The SPI timing nonce ZZQ42 configures the peripheral bus.</p>"
        '<img src="pic.png" alt="the wiring diagram">'
        "</body></html>",
        encoding="utf-8",
    )
    (root / "pic.png").write_bytes(_PNG)
    config_path = tmp_path / "docusearch.yaml"
    config_path.write_text(
        f'paths:\n  staging_dir: "{(tmp_path / "staging").as_posix()}"\n'
        f'  db_path: "{(tmp_path / "catalog.db").as_posix()}"\n'
        f'  tmp_dir: "{(tmp_path / "tmp").as_posix()}"\n'
        f'sources:\n  - name: d\n    location: "{root.as_posix()}"\n    min_content_chars: 5\n'
        'embed:\n  model: "none"\n',
        encoding="utf-8",
    )
    config = cfg.load(config_path)
    with Store.open(config.paths.db_path) as store:
        ingest.run_ingest(config, store)
    yield TestClient(create_app(config))


@pytest.fixture
def vector_client(tmp_path: Path) -> Iterator[Any]:
    """A server over an index embedded with a fake provider (model 'fake-v1', dim 8)."""
    from ._fakes import FakeProvider

    root = tmp_path / "docs"
    root.mkdir()
    for i in range(4):
        (root / f"d{i}.html").write_text(
            f"<body><h1>Doc {i}</h1><p>content about topic {i} timing bus</p></body>", "utf-8"
        )
    config_path = tmp_path / "docusearch.yaml"
    config_path.write_text(
        f'paths:\n  staging_dir: "{(tmp_path / "s").as_posix()}"\n'
        f'  db_path: "{(tmp_path / "catalog.db").as_posix()}"\n  tmp_dir: "{(tmp_path / "t").as_posix()}"\n'
        f'sources:\n  - name: d\n    location: "{root.as_posix()}"\n    min_content_chars: 5\n'
        'embed:\n  model: "none"\n',
        encoding="utf-8",
    )
    config = cfg.load(config_path)
    with Store.open(config.paths.db_path) as store:
        ingest.run_ingest(config, store, provider=FakeProvider("fake-v1", dim=8))
    yield TestClient(create_app(config))


def test_cli_and_mcp_reports_are_identical_except_ref_scheme(tmp_path: Path) -> None:
    # The MCP build_report and the CLI render the SAME spec through the SAME deterministic renderer,
    # so the reports must match byte-for-byte except the reference LINK (served /v1/documents vs
    # local file://) — the labels and every other line are identical.
    import re

    from docusearch import report, runlog
    from docusearch.server import Service

    root = tmp_path / "docs"
    root.mkdir()
    (root / "spi.html").write_text("<body><h1>SPI</h1><p>PA drives the SPI bus.</p></body>", "utf-8")
    config_path = tmp_path / "docusearch.yaml"
    config_path.write_text(
        f'paths:\n  staging_dir: "{(tmp_path / "s").as_posix()}"\n'
        f'  db_path: "{(tmp_path / "c.db").as_posix()}"\n  tmp_dir: "{(tmp_path / "t").as_posix()}"\n'
        f'sources:\n  - name: docs\n    location: "{root.as_posix()}"\n    min_content_chars: 5\n'
        'embed:\n  model: "none"\n',
        encoding="utf-8",
    )
    config = cfg.load(config_path)
    with Store.open(config.paths.db_path) as store:
        ingest.run_ingest(config, store)
        d, c = store._conn.execute(
            "SELECT d.id, ch.id FROM chunks ch JOIN documents d ON ch.document_id=d.id LIMIT 1"
        ).fetchone()
    d, c = int(d), int(c)
    spec = {
        "title": "PA Overview", "subtitle": "sub", "request": "overview", "requested_by": "S",
        "model": "m", "audience": ["engineering"], "evidence": [[d, c]],
        "sections": [{"heading": "Overview", "kind": "overview", "body": f"PA drives SPI [D:{d}#{c}]."}],
    }
    base = "http://localhost:8321"
    evidence = {(d, c)}
    mcp = Service(config).build_report(spec, base_url=base, fmt="md")
    cli = report.render_report(
        title=spec["title"], subtitle=spec["subtitle"], sections=spec["sections"], evidence=evidence,
        base_url=base, fmt="md", run_id=runlog.RUN_ID, audience=spec["audience"], embed_model="none",
        sources=["docs"], request=spec["request"], requested_by=spec["requested_by"],
        model=spec["model"],
        ref_targets=report.reference_targets(config.paths.db_path, evidence),  # file:// (local CLI)
    )

    def norm(t: str) -> str:
        t = re.sub(r"\d{4}-\d\d-\d\dT[\d:+.-]+", "TS", t)  # timestamp
        t = re.sub(r"\d{8}T\d{6}-[0-9a-f]+", "RUN", t)  # run id
        t = re.sub(r"(file://\S+|http://localhost:8321/v1/documents/\S+)", "REF", t)  # ref link
        return t

    assert norm(cli) == norm(mcp)  # identical content; only the ref link host differs
    assert "file://" in cli and "/v1/documents/" in mcp  # each uses its context-correct scheme


def _fake_vec() -> list[float]:
    from ._fakes import FakeProvider

    return FakeProvider("fake-v1", dim=8).embed(["topic 1 timing"])[0].tolist()


def test_search_vectors_matching_model(vector_client: TestClient) -> None:
    resp = vector_client.post(
        "/v1/search", json={"query_vectors": [_fake_vec()], "embed_model": "fake-v1"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["search_mode"] == "vector"
    assert body["results"][0]  # vector search returned hits


def test_search_vectors_model_mismatch_409(vector_client: TestClient) -> None:
    resp = vector_client.post(
        "/v1/search", json={"query_vectors": [_fake_vec()], "embed_model": "some-other-model"}
    )
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["error"] == "EMBED_MODEL_MISMATCH"
    assert detail["server_model"] == "fake-v1" and detail["server_dim"] == 8
    assert "text" in detail["hint"]  # recoverable: re-send as text


def test_search_vectors_without_model_is_400(vector_client: TestClient) -> None:
    resp = vector_client.post("/v1/search", json={"query_vectors": [_fake_vec()]})
    assert resp.status_code == 400


def test_embed_endpoint_without_model_is_409(client: TestClient) -> None:
    # the plain `client` fixture is a BM25-only (model: none) server
    resp = client.post("/v1/embed", json={"texts": ["anything"]})
    assert resp.status_code == 409
    assert resp.json()["detail"]["error"] == "NO_EMBED_MODEL"


def test_health(client: TestClient) -> None:
    body = client.get("/v1/health").json()
    assert body["documents"] == 1
    assert body["chunks"] >= 1
    assert body["embed_model"] == "none"
    assert body["images"] == 1


def test_embed_info(client: TestClient) -> None:
    body = client.get("/v1/embed-info").json()
    assert body["model"] == "none" and body["dim"] == 0


def test_search_returns_hits_with_citation_url(client: TestClient) -> None:
    resp = client.post("/v1/search", json={"query_texts": ["ZZQ42"], "top_k": 5}).json()
    assert resp["search_mode"] == "bm25"
    assert resp["embed_model_used"] == "none"
    hits = resp["results"][0]
    assert hits
    hit = hits[0]
    assert hit["citation"] == f"D:{hit['doc_id']}#{hit['chunk_id']}"
    assert hit["url"].endswith(f"/v1/documents/{hit['doc_id']}?chunk={hit['chunk_id']}")


def test_batch_search(client: TestClient) -> None:
    resp = client.post("/v1/search", json={"query_texts": ["ZZQ42", "diagram"]}).json()
    assert len(resp["results"]) == 2


def test_get_document_and_highlight(client: TestClient) -> None:
    doc = client.get("/v1/documents/1").json()
    assert doc["id"] == 1 and doc["title"] == "SPI"
    assert any("ZZQ42" in c["text"] for c in doc["chunks"])
    highlighted = client.get("/v1/documents/1", params={"chunk": doc["chunks"][0]["id"]}).json()
    assert highlighted["chunks"][0]["highlight"] is True


def test_document_download_streams_original(client: TestClient) -> None:
    resp = client.get("/v1/documents/1", params={"download": 1})
    assert resp.status_code == 200
    assert b"<h1>SPI</h1>" in resp.content


def test_image_serving(client: TestClient) -> None:
    resp = client.get(f"/v1/images/{_PNG_SHA}")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert resp.content == _PNG


def test_report_endpoint_renders_with_references(client: TestClient) -> None:
    doc = client.get("/v1/documents/1").json()
    cid = doc["chunks"][0]["id"]
    resp = client.post(
        "/v1/reports",
        json={
            "title": "SPI note",
            "body": f"SPI is configured here [D:1#{cid}].",
            "evidence": [[1, cid]],
        },
    )
    assert resp.status_code == 200
    rendered = resp.json()["report"]
    assert "# SPI note" in rendered and "## References" in rendered
    assert f"/v1/documents/1?chunk={cid}" in rendered


def test_report_rejects_hallucinated_citation(client: TestClient) -> None:
    resp = client.post(
        "/v1/reports",
        json={"title": "T", "body": "Fabricated [D:9#99999].", "evidence": [[1, 1]]},
    )
    assert resp.status_code == 400
    assert resp.json()["detail"]["error"] == "HALLUCINATED_CITATION"


def test_report_rejects_citation_in_title(client: TestClient) -> None:
    # red-team H1: a fabricated citation in the title must be refused too
    resp = client.post(
        "/v1/reports",
        json={"title": "Sneaky [D:9#99999]", "body": "ok [GK].", "evidence": [[1, 1]]},
    )
    assert resp.status_code == 400
    assert resp.json()["detail"]["error"] == "HALLUCINATED_CITATION"


def test_absurd_document_id_is_404_not_500(client: TestClient) -> None:
    # red-team L1: a huge id must 404, not raise a sqlite OverflowError (500)
    assert client.get("/v1/documents/999999999999999999999999999").status_code == 404


def test_wrong_dimension_vector_is_400(vector_client: TestClient) -> None:
    # red-team L1: right model tag but wrong vector dim -> clean 400, not a crash
    resp = vector_client.post(
        "/v1/search", json={"query_vectors": [[0.1, 0.2, 0.3]], "embed_model": "fake-v1"}
    )
    assert resp.status_code == 400


def test_relations_endpoint(client: TestClient) -> None:
    resp = client.get("/v1/relations/1")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_404s(client: TestClient) -> None:
    assert client.get("/v1/documents/9999").status_code == 404
    assert client.get("/v1/images/deadbeef").status_code == 404


def test_path_traversal_image_id_is_safe(client: TestClient) -> None:
    # an absurd/hostile sha must 404, never escape the images dir
    assert client.get("/v1/images/..%2f..%2f..%2fetc%2fpasswd").status_code in (404, 400)


def test_mcp_is_mounted_at_configured_path(tmp_path: Path) -> None:
    from docusearch.server import build_mcp, create_app

    config_path = tmp_path / "docusearch.yaml"
    config_path.write_text(
        f'paths:\n  db_path: "{(tmp_path / "c.db").as_posix()}"\n'
        f'  staging_dir: "{(tmp_path / "s").as_posix()}"\n  tmp_dir: "{(tmp_path / "t").as_posix()}"\n'
        'embed:\n  model: "none"\n',
        encoding="utf-8",
    )
    config = cfg.load(config_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        app = create_app(config)
        # the stable tool names are registered on the MCP server (agents depend on them)
        from docusearch.server import Service

        tool_names = {t.name for t in build_mcp(Service(config), config)._tool_manager.list_tools()}
    assert config.serve.mcp_path in {getattr(r, "path", None) for r in app.routes}
    assert {"search_docs", "get_document", "related_documents", "catalog_stats"} <= tool_names


@pytest.mark.model
def test_hybrid_search_over_rest_real_model(tmp_path: Path) -> None:
    model = "sentence-transformers/all-MiniLM-L6-v2"
    root = tmp_path / "docs"
    root.mkdir()
    for name, text in {
        "spi.html": "serial peripheral interface bus transfers data with a shared clock",
        "uart.html": "universal asynchronous receiver transmitter over a serial line",
    }.items():
        (root / name).write_text(f"<body><h1>{name}</h1><p>{text}</p></body>", "utf-8")
    config_path = tmp_path / "docusearch.yaml"
    config_path.write_text(
        f'paths:\n  db_path: "{(tmp_path / "c.db").as_posix()}"\n'
        f'  staging_dir: "{(tmp_path / "s").as_posix()}"\n  tmp_dir: "{(tmp_path / "t").as_posix()}"\n'
        f'sources:\n  - name: d\n    location: "{root.as_posix()}"\n    min_content_chars: 5\n'
        f'embed:\n  model: "{model}"\n  device: cpu\n',
        encoding="utf-8",
    )
    config = cfg.load(config_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with Store.open(config.paths.db_path) as store:
            ingest.run_ingest(config, store)
        app_client = TestClient(create_app(config))
        resp = app_client.post("/v1/search", json={"query_texts": ["how does the SPI clock work"]})
        info = app_client.get("/v1/embed-info").json()
    body = resp.json()
    assert body["search_mode"] == "hybrid"
    assert body["embed_model_used"] == model
    assert body["results"][0][0]["path"].endswith("spi.html")
    assert info["model"] == model and info["dim"] == 384 and info["approx_mb"] > 0
