"""Optional AI summaries (§17 Phase 5): config-gated per-document summaries persisted as searchable
enrichment chunks. Determinism by persistence; built against an injected fake runner (no network)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from docusearch import config as cfg
from docusearch import ingest
from docusearch.catalog import Catalog
from docusearch.config import ConfigError
from docusearch.store import Store

from ._fakes import FakeProvider


def _config(tmp_path: Path, *, ai_summaries: bool) -> cfg.Config:
    root = tmp_path / "docs"
    root.mkdir(exist_ok=True)
    (root / "reset.html").write_text(
        "<body><h1>Reset</h1><p>The nonce ZQX9 documents the reset procedure for the tester.</p></body>",
        encoding="utf-8",
    )
    path = tmp_path / "d.yaml"
    path.write_text(
        f'paths:\n  staging_dir: "{(tmp_path / "s").as_posix()}"\n'
        f'  db_path: "{(tmp_path / "c.db").as_posix()}"\n  tmp_dir: "{(tmp_path / "t").as_posix()}"\n'
        f'sources:\n  - name: d\n    location: "{root.as_posix()}"\n    min_content_chars: 5\n'
        f'embed:\n  model: "none"\nenrich:\n  ai_summaries: {"true" if ai_summaries else "false"}\n',
        encoding="utf-8",
    )
    return cfg.load(path)


def _fake_runner(argv: list[str]) -> tuple[int, str, str]:
    return 0, json.dumps({"result": "Summary: covers the tester reset procedure.", "is_error": False}), ""


def test_enrich_summaries_makes_searchable_chunk_and_is_idempotent(tmp_path: Path) -> None:
    config = _config(tmp_path, ai_summaries=True)
    with Store.open(config.paths.db_path) as store:
        ingest.run_ingest(config, store)
    cat = Catalog(config)
    result = cat.enrich_summaries(model="m", runner=_fake_runner)
    assert result.summarized == 1 and result.failed == 0
    with Store.open(config.paths.db_path) as store:
        assert store.chunk_ids_matching("procedure")  # summary is BM25-searchable
        rows = store._conn.execute(  # noqa: SLF001
            "SELECT text FROM chunks WHERE kind='enrichment' AND locator='summary'"
        ).fetchall()
        assert len(rows) == 1 and "reset procedure" in rows[0][0]
    # a re-run summarizes nothing new (idempotent, no double spend)
    again = cat.enrich_summaries(model="m", runner=_fake_runner)
    assert again.pending == 0 and again.summarized == 0


def test_enrich_summaries_embeds_new_chunks_on_hybrid_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # On a hybrid index the summary chunk must be embedded (so hybrid search finds it) and the ANN
    # refreshed — the embed-after-enrich path.
    config = _config(tmp_path, ai_summaries=True)
    with Store.open(config.paths.db_path) as store:
        ingest.run_ingest(config, store, provider=FakeProvider())
    monkeypatch.setattr(Catalog, "_provider", lambda self: FakeProvider())
    result = Catalog(config).enrich_summaries(model="m", runner=_fake_runner)
    assert result.summarized == 1
    with Store.open(config.paths.db_path) as store:
        assert store.chunks_without_embeddings() == []  # summary chunk got embedded too


def test_enrich_summaries_refuses_when_disabled(tmp_path: Path) -> None:
    config = _config(tmp_path, ai_summaries=False)
    with Store.open(config.paths.db_path) as store:
        ingest.run_ingest(config, store)
    with pytest.raises(ConfigError):
        Catalog(config).enrich_summaries(model="m", runner=_fake_runner)
