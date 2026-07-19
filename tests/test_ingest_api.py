"""Write-path API (Phase 4e): POST new documents into a store — a server-side folder or an
uploaded .zip/.tar.gz — labelled and attributed to the submitting username."""

from __future__ import annotations

import io
import warnings
import zipfile
from pathlib import Path

from docusearch import config as cfg

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from fastapi.testclient import TestClient

    from docusearch.server import create_app


def _client(tmp_path: Path) -> TestClient:
    (tmp_path / "seed").mkdir()
    (tmp_path / "seed" / "seed.html").write_text("<body><h1>Seed</h1><p>seed doc.</p></body>", "utf-8")
    path = tmp_path / "d.yaml"
    path.write_text(
        f'paths:\n  staging_dir: "{(tmp_path / "s").as_posix()}"\n'
        f'  db_path: "{(tmp_path / "c.db").as_posix()}"\n  tmp_dir: "{(tmp_path / "t").as_posix()}"\n'
        f'sources:\n  - name: seed\n    location: "{(tmp_path / "seed").as_posix()}"\n    min_content_chars: 3\n'
        'embed:\n  model: "none"\n',
        encoding="utf-8",
    )
    from docusearch import ingest
    from docusearch.store import Store

    conf = cfg.load(path)
    with Store.open(conf.paths.db_path) as store:
        ingest.run_ingest(conf, store)
    return TestClient(create_app(conf))


def test_ingest_folder_then_search_finds_it(tmp_path: Path) -> None:
    client = _client(tmp_path)
    # a server-side folder ingest is confined to the store's inbound dir (red-team H4)
    add = tmp_path / "s" / "inbound" / "new"
    add.mkdir(parents=True)
    (add / "note.html").write_text("<body><h1>Note</h1><p>UPLOADNONCE1 fresh content.</p></body>", "utf-8")
    r = client.post(
        "/v1/ingest",
        json={"path": str(add), "label": "vendor-drop", "min_content_chars": 3},
        headers={"X-Docusearch-User": "alice"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["documents"] == 1 and r.json()["uploaded_by"] == "alice"
    hits = client.post("/v1/search", json={"query_texts": ["UPLOADNONCE1"]}).json()["results"][0]
    assert hits and "UPLOADNONCE1" in hits[0]["snippet"]


def test_ingest_folder_outside_inbound_is_rejected(tmp_path: Path) -> None:
    # arbitrary server-filesystem paths must NOT be ingestable via the API (red-team H4)
    client = _client(tmp_path)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "secret.html").write_text("<body><p>SECRETFS content</p></body>", "utf-8")
    r = client.post("/v1/ingest", json={"path": str(outside)}, headers={"X-Docusearch-User": "alice"})
    assert r.status_code == 400 and "inbound" in r.json()["detail"].lower()


def test_ingest_zip_upload(tmp_path: Path) -> None:
    client = _client(tmp_path)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("a/inside.html", "<body><h1>Zipped</h1><p>ZIPNONCE2 from an archive.</p></body>")
    buf.seek(0)
    r = client.post(
        "/v1/ingest/upload",
        files={"file": ("docs.zip", buf, "application/zip")},
        data={"label": "archived"},
        headers={"X-Docusearch-User": "bob"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["documents"] == 1
    hits = client.post("/v1/search", json={"query_texts": ["ZIPNONCE2"]}).json()["results"][0]
    assert hits and "ZIPNONCE2" in hits[0]["snippet"]


def test_ingest_requires_username(tmp_path: Path) -> None:
    client = _client(tmp_path)
    add = tmp_path / "new2"
    add.mkdir()
    (add / "x.html").write_text("<body><p>content here</p></body>", "utf-8")
    r = client.post("/v1/ingest", json={"path": str(add)})  # no X-Docusearch-User
    assert r.status_code == 401


def test_zip_slip_is_rejected(tmp_path: Path) -> None:
    client = _client(tmp_path)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("../../escape.html", "<body><p>evil</p></body>")  # path traversal
    buf.seek(0)
    r = client.post(
        "/v1/ingest/upload",
        files={"file": ("evil.zip", buf, "application/zip")},
        data={"label": "evil"},
        headers={"X-Docusearch-User": "mallory"},
    )
    assert r.status_code == 400 and "unsafe" in r.json()["detail"].lower()


def test_feedback_is_recorded(tmp_path: Path) -> None:
    client = _client(tmp_path)
    r = client.post(
        "/v1/feedback",
        json={"text": "the SPI result was spot on", "doc_id": 1, "rating": 5},
        headers={"X-Docusearch-User": "alice"},
    )
    assert r.status_code == 200 and r.json()["recorded"] is True
    log = tmp_path / "t" / "feedback" / "feedback.jsonl"
    assert log.is_file() and "spot on" in log.read_text() and "alice" in log.read_text()
    # feedback also requires a username
    assert client.post("/v1/feedback", json={"text": "anon"}).status_code == 401


def test_ingest_targz_upload(tmp_path: Path) -> None:
    # Red-team M3: a .tar.gz upload must ingest (the temp-file suffix must keep both extensions).
    import io
    import tarfile

    client = _client(tmp_path)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        data = b"<body><h1>Tar</h1><p>TARNONCE3 from a gzip tarball.</p></body>"
        info = tarfile.TarInfo("t/inside.html")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    buf.seek(0)
    r = client.post(
        "/v1/ingest/upload",
        files={"file": ("docs.tar.gz", buf, "application/gzip")},
        data={"label": "tarred"},
        headers={"X-Docusearch-User": "bob"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["documents"] == 1
    hits = client.post("/v1/search", json={"query_texts": ["TARNONCE3"]}).json()["results"][0]
    assert hits and "TARNONCE3" in hits[0]["snippet"]


def test_malformed_archive_is_clean_400(tmp_path: Path) -> None:
    # Red-team M2: a corrupt/non-archive named .zip must 400, not crash with 500.
    import io

    client = _client(tmp_path)
    r = client.post(
        "/v1/ingest/upload",
        files={"file": ("bad.zip", io.BytesIO(b"not a real zip"), "application/zip")},
        data={"label": "bad"},
        headers={"X-Docusearch-User": "bob"},
    )
    assert r.status_code == 400
