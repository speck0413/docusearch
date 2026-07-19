"""Store access control (Phase 4e): a doc store is public (anyone on the server) or private (a
whitelist of usernames / groups, verified from the request's username). Default: public."""

from __future__ import annotations

from pathlib import Path

from docusearch import config


def _write(tmp_path: Path, access_block: str) -> config.Config:
    path = tmp_path / "d.yaml"
    path.write_text(
        f'paths:\n  staging_dir: "{(tmp_path / "s").as_posix()}"\n'
        f'  db_path: "{(tmp_path / "c.db").as_posix()}"\n  tmp_dir: "{(tmp_path / "t").as_posix()}"\n'
        'sources: []\nembed:\n  model: "none"\n' + access_block,
        encoding="utf-8",
    )
    return config.load(path)


def test_access_defaults_to_public(tmp_path: Path) -> None:
    # No access: section -> public (anyone can search).
    cfg = _write(tmp_path, "")
    assert cfg.access.visibility == "public"
    assert cfg.access.permits(user=None, groups=set())  # even an anonymous request


def test_private_store_requires_whitelisted_user_or_group(tmp_path: Path) -> None:
    cfg = _write(
        tmp_path,
        'access:\n  visibility: private\n  allowed_users: ["alice", "bob"]\n'
        '  allowed_groups: ["engineering"]\n',
    )
    acc = cfg.access
    assert acc.visibility == "private"
    assert not acc.permits(user=None, groups=set())  # anonymous -> denied
    assert acc.permits(user="alice", groups=set())  # whitelisted user
    assert not acc.permits(user="carol", groups=set())  # not whitelisted
    assert acc.permits(user="carol", groups={"engineering"})  # whitelisted group
    assert not acc.permits(user="carol", groups={"sales"})  # wrong group


def _serve(tmp_path: Path, access_block: str):  # type: ignore[no-untyped-def]
    """A served app over a one-doc store with the given access: block."""
    import warnings

    from docusearch import ingest
    from docusearch.store import Store

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from fastapi.testclient import TestClient

        from docusearch.server import create_app

    root = tmp_path / "docs"
    root.mkdir(parents=True)
    (root / "a.html").write_text("<body><h1>Secret</h1><p>WIDGET55 confidential note.</p></body>", "utf-8")
    path = tmp_path / "d.yaml"
    path.write_text(
        f'paths:\n  staging_dir: "{(tmp_path / "s").as_posix()}"\n'
        f'  db_path: "{(tmp_path / "c.db").as_posix()}"\n  tmp_dir: "{(tmp_path / "t").as_posix()}"\n'
        f'sources:\n  - name: d\n    location: "{root.as_posix()}"\n    min_content_chars: 5\n'
        'embed:\n  model: "none"\n' + access_block,
        encoding="utf-8",
    )
    conf = config.load(path)
    with Store.open(conf.paths.db_path) as store:
        ingest.run_ingest(conf, store)
    return TestClient(create_app(conf))


def test_served_public_store_is_open(tmp_path: Path) -> None:
    client = _serve(tmp_path, "")  # public
    r = client.post("/v1/search", json={"query_texts": ["WIDGET55"]})  # no user header
    assert r.status_code == 200 and r.json()["results"][0]  # anyone sees it


def test_served_private_store_gates_by_header(tmp_path: Path) -> None:
    client = _serve(tmp_path, 'access:\n  visibility: private\n  allowed_users: ["alice"]\n')
    # anonymous -> 403
    assert client.post("/v1/search", json={"query_texts": ["WIDGET55"]}).status_code == 403
    # wrong user -> 403
    r = client.post("/v1/search", json={"query_texts": ["WIDGET55"]},
                    headers={"X-Docusearch-User": "carol"})
    assert r.status_code == 403
    # whitelisted user via header -> 200 with results
    r = client.post("/v1/search", json={"query_texts": ["WIDGET55"]},
                    headers={"X-Docusearch-User": "alice"})
    assert r.status_code == 200 and r.json()["results"][0]
    # group whitelist via header
    client2 = _serve(tmp_path / "g", 'access:\n  visibility: private\n  allowed_groups: ["eng"]\n')
    r = client2.post("/v1/search", json={"query_texts": ["WIDGET55"]},
                     headers={"X-Docusearch-User": "dave", "X-Docusearch-Groups": "eng,sales"})
    assert r.status_code == 200 and r.json()["results"][0]


def test_federation_hides_private_member_from_non_whitelisted_user(tmp_path: Path) -> None:
    # A company federation with a PUBLIC 'python' store and a PRIVATE 'acme' store (whitelisted to
    # alice). A non-whitelisted user sees only python; alice sees both. A private store the user
    # can't access is invisible (looks 'unknown' if explicitly requested) — its existence isn't leaked.
    import warnings

    from docusearch import ingest
    from docusearch.store import Store

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from fastapi.testclient import TestClient

        from docusearch.server import create_app

    def member(name: str, token: str, access_block: str) -> None:
        root = tmp_path / f"{name}-docs"
        root.mkdir(parents=True)
        (root / "d.html").write_text(f"<body><h1>{name}</h1><p>{token} shared term.</p></body>", "utf-8")
        (tmp_path / f"{name}.yaml").write_text(
            f'paths:\n  staging_dir: "{(tmp_path / name / "s").as_posix()}"\n'
            f'  db_path: "{(tmp_path / name / "c.db").as_posix()}"\n  tmp_dir: "{(tmp_path / name / "t").as_posix()}"\n'
            f'sources:\n  - name: {name}\n    location: "{root.as_posix()}"\n    min_content_chars: 5\n'
            'embed:\n  model: "none"\n' + access_block,
            encoding="utf-8",
        )
        conf = config.load(tmp_path / f"{name}.yaml")
        with Store.open(conf.paths.db_path) as store:
            ingest.run_ingest(conf, store)

    member("python", "SHARED88", "")  # public
    member("acme", "SHARED88", 'access:\n  visibility: private\n  allowed_users: ["alice"]\n')
    (tmp_path / "federation.yaml").write_text(
        f'paths:\n  staging_dir: "{(tmp_path / "f/s").as_posix()}"\n'
        f'  db_path: "{(tmp_path / "f/c.db").as_posix()}"\n  tmp_dir: "{(tmp_path / "f/t").as_posix()}"\n'
        'sources: []\nembed:\n  model: "none"\n'
        f'federation:\n  - name: python\n    config: "{(tmp_path / "python.yaml").as_posix()}"\n'
        f'  - name: acme\n    config: "{(tmp_path / "acme.yaml").as_posix()}"\n',
        encoding="utf-8",
    )
    client = TestClient(create_app(config.load(tmp_path / "federation.yaml")))

    def paths(resp) -> list[str]:  # type: ignore[no-untyped-def]
        return [h["path"] for h in resp.json()["results"][0]]

    # anonymous: only the public python store
    anon = client.post("/v1/search", json={"query_texts": ["SHARED88"]})
    assert anon.status_code == 200
    assert any("python-docs" in p for p in paths(anon))
    assert not any("acme-docs" in p for p in paths(anon))
    # alice: both stores
    alice = client.post("/v1/search", json={"query_texts": ["SHARED88"]},
                        headers={"X-Docusearch-User": "alice"})
    assert any("acme-docs" in p for p in paths(alice))
    # anonymous explicitly scoping to the private store -> 'unknown' (existence not leaked)
    scoped = client.post("/v1/search", json={"query_texts": ["SHARED88"], "stores": ["acme"]})
    assert scoped.status_code == 400 and "unknown" in scoped.json()["detail"].lower()


def test_private_store_gates_all_read_paths(tmp_path: Path) -> None:
    # Red-team H1/M1: EVERY read path (not just search) must 403 for a non-whitelisted caller on a
    # private store — get_document/images/relations/health/embed-info, and ?download.
    client = _serve(tmp_path, 'access:\n  visibility: private\n  allowed_users: ["alice"]\n')
    for path in ("/v1/health", "/v1/embed-info", "/v1/documents/1", "/v1/documents/1?download=1",
                 "/v1/relations/1", "/v1/images/deadbeef"):
        assert client.get(path).status_code == 403, f"{path} leaked to anonymous"
    # a whitelisted user gets through (200/404, never 403)
    assert client.get("/v1/health", headers={"X-Docusearch-User": "alice"}).status_code == 200


def test_list_stores_omits_private_members_for_outsider(tmp_path: Path) -> None:
    # Red-team H2: list_stores must not leak a private member's name to a non-whitelisted caller.
    from docusearch.server import Service

    for name, priv in (("pub", ""), ("sec", 'access:\n  visibility: private\n  allowed_users: ["alice"]\n')):
        root = tmp_path / f"{name}-docs"
        root.mkdir(parents=True)
        (root / "d.html").write_text("<body><p>x here</p></body>", "utf-8")
        (tmp_path / f"{name}.yaml").write_text(
            f'paths:\n  staging_dir: "{(tmp_path/name/"s").as_posix()}"\n  db_path: "{(tmp_path/name/"c.db").as_posix()}"\n'
            f'  tmp_dir: "{(tmp_path/name/"t").as_posix()}"\nsources:\n  - name: {name}\n    location: "{root.as_posix()}"\n'
            '    min_content_chars: 3\nembed:\n  model: "none"\n' + priv, encoding="utf-8")
    fed = tmp_path / "fed.yaml"
    fed.write_text(
        f'paths:\n  staging_dir: "{(tmp_path/"f/s").as_posix()}"\n  db_path: "{(tmp_path/"f/c.db").as_posix()}"\n'
        f'  tmp_dir: "{(tmp_path/"f/t").as_posix()}"\nsources: []\nembed:\n  model: "none"\n'
        f'federation:\n  - name: pub\n    config: "{(tmp_path/"pub.yaml").as_posix()}"\n'
        f'  - name: sec\n    config: "{(tmp_path/"sec.yaml").as_posix()}"\n', encoding="utf-8")
    svc = Service(config.load(fed))
    assert svc.list_stores(user=None, groups=set())["stores"] == ["pub"]  # outsider: no 'sec'
    assert set(svc.list_stores(user="alice", groups=set())["stores"]) == {"pub", "sec"}  # insider


def test_write_to_private_store_requires_whitelist(tmp_path: Path) -> None:
    # Red-team H3: writing a private store requires the uploader be whitelisted.
    from docusearch.server import Service

    root = tmp_path / "sec-docs"
    root.mkdir()
    (root / "d.html").write_text("<body><p>x</p></body>", "utf-8")
    (tmp_path / "sec.yaml").write_text(
        f'paths:\n  staging_dir: "{(tmp_path/"sec/s").as_posix()}"\n  db_path: "{(tmp_path/"sec/c.db").as_posix()}"\n'
        f'  tmp_dir: "{(tmp_path/"sec/t").as_posix()}"\nsources: []\nembed:\n  model: "none"\n'
        'access:\n  visibility: private\n  allowed_users: ["alice"]\n', encoding="utf-8")
    fed = tmp_path / "fed.yaml"
    fed.write_text(
        f'paths:\n  staging_dir: "{(tmp_path/"f/s").as_posix()}"\n  db_path: "{(tmp_path/"f/c.db").as_posix()}"\n'
        f'  tmp_dir: "{(tmp_path/"f/t").as_posix()}"\nsources: []\nembed:\n  model: "none"\n'
        f'federation:\n  - name: sec\n    config: "{(tmp_path/"sec.yaml").as_posix()}"\n', encoding="utf-8")
    svc = Service(config.load(fed))
    inbound = tmp_path / "sec/s/inbound/drop"
    inbound.mkdir(parents=True)
    (inbound / "n.html").write_text("<body><p>NEWSEC content here</p></body>", "utf-8")
    import pytest
    with pytest.raises(PermissionError):
        svc.ingest_from_path(inbound, store="sec", uploaded_by="mallory", groups=set(), min_content_chars=3)
    # whitelisted user can write
    res = svc.ingest_from_path(inbound, store="sec", uploaded_by="alice", groups=set(), min_content_chars=3)
    assert res["documents"] == 1
