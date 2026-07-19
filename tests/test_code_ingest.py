"""Ingest a source repo into a `store_type: code` store (Phase 9): files parse into per-symbol chunks
+ a structured code_symbols table, and the symbols are searchable. Multi-language in one store."""

from __future__ import annotations

from pathlib import Path

from docusearch import config as cfg
from docusearch import ingest
from docusearch import search as search_mod
from docusearch.store import Store

PY = '''\
"""A tiny client module."""


def connect(url: str) -> bool:
    """Open a connection to the given URL."""
    return bool(url)


class Client:
    """A reusable client."""

    def send(self, payload: bytes) -> int:
        """Send a payload and return the byte count."""
        return len(payload)
'''

JS = "function helper(x) { return x + 1 }\n"


def _cfg(tmp_path: Path, code_dir: Path) -> cfg.Config:
    path = tmp_path / "d.yaml"
    path.write_text(
        f'store_type: "code"\npaths:\n  staging_dir: "{(tmp_path / "s").as_posix()}"\n'
        f'  db_path: "{(tmp_path / "c.db").as_posix()}"\n  tmp_dir: "{(tmp_path / "t").as_posix()}"\n'
        f'sources:\n  - name: repo\n    location: "{code_dir.as_posix()}"\n'
        '    include: ["*.py", "*.js"]\n    min_content_chars: 1\n'
        'embed:\n  model: "none"\n', encoding="utf-8")
    return cfg.load(path)


def test_ingest_code_repo_symbols_and_search(tmp_path: Path) -> None:
    d = tmp_path / "repo"
    d.mkdir()
    (d / "client.py").write_text(PY, encoding="utf-8")
    (d / "util.js").write_text(JS, encoding="utf-8")
    config = _cfg(tmp_path, d)
    with Store.open(config.paths.db_path) as store:
        result = ingest.run_ingest(config, store)
        assert result.documents == 2
        assert result.code_symbols == 4  # connect, Client, Client.send, helper

        syms = {r["qualname"]: r for r in store.code_symbols_query()}
        assert set(syms) == {"connect", "Client", "Client.send", "helper"}
        assert syms["connect"]["kind"] == "function" and syms["connect"]["language"] == "python"
        assert syms["Client.send"]["kind"] == "method" and syms["Client.send"]["parent"] == "Client"
        assert syms["Client.send"]["docstring"] == "Send a payload and return the byte count."
        assert syms["helper"]["language"] == "javascript"
        assert store.code_symbols_query(kind="method")[0]["qualname"] == "Client.send"

        # the symbols are BM25-searchable by docstring text and by name
        hits = search_mod.search(store, "open a connection URL", top_k=5, bm25_only=True)
        assert isinstance(hits, list) and hits and "connect" in hits[0].locator

        js_hits = search_mod.search(store, "helper", top_k=5, bm25_only=True)
        assert any("helper" in h.locator for h in js_hits)


def test_document_store_does_not_treat_code_as_symbols(tmp_path: Path) -> None:
    # the SAME .py file in a `document` store chunks as prose (no code_symbols) — routing is opt-in
    d = tmp_path / "docs"
    d.mkdir()
    (d / "client.py").write_text(PY, encoding="utf-8")
    path = tmp_path / "d.yaml"
    path.write_text(
        f'store_type: "document"\npaths:\n  staging_dir: "{(tmp_path / "s").as_posix()}"\n'
        f'  db_path: "{(tmp_path / "c.db").as_posix()}"\n  tmp_dir: "{(tmp_path / "t").as_posix()}"\n'
        f'sources:\n  - name: docs\n    location: "{d.as_posix()}"\n'
        '    include: ["*.py"]\n    min_content_chars: 1\n'
        'embed:\n  model: "none"\n', encoding="utf-8")
    config = cfg.load(path)
    with Store.open(config.paths.db_path) as store:
        result = ingest.run_ingest(config, store)
        assert result.code_symbols == 0 and store.count_code_symbols() == 0
        assert result.documents == 1  # still ingested, just as a plain-text document
