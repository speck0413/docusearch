"""FastAPI REST surface over the service layer (§10, R-API-1/2)."""

from __future__ import annotations

import hashlib
import warnings
from collections.abc import Iterator
from pathlib import Path

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


def test_404s(client: TestClient) -> None:
    assert client.get("/v1/documents/9999").status_code == 404
    assert client.get("/v1/images/deadbeef").status_code == 404


def test_path_traversal_image_id_is_safe(client: TestClient) -> None:
    # an absurd/hostile sha must 404, never escape the images dir
    assert client.get("/v1/images/..%2f..%2f..%2fetc%2fpasswd").status_code in (404, 400)


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
