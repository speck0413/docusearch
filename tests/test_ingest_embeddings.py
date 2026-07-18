"""Embeddings at index time + provenance (R-EMB-2/3/6, R-CFG-4).

Uses a deterministic fake provider (no torch) so these stay fast and offline; the real
LocalProvider is covered in test_embed.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from docusearch import config as cfg
from docusearch import embed, ingest
from docusearch.store import Store

from ._fakes import FakeProvider


def _config(tmp: Path, root: Path, *, model: str = "none") -> cfg.Config:
    path = tmp / "docusearch.yaml"
    path.write_text(
        f'paths:\n  staging_dir: "{(tmp / "s").as_posix()}"\n'
        f'  db_path: "{(tmp / "c.db").as_posix()}"\n'
        f'  tmp_dir: "{(tmp / "t").as_posix()}"\n'
        f'sources:\n  - name: d\n    location: "{root.as_posix()}"\n    min_content_chars: 5\n'
        f'embed:\n  model: "{model}"\n  batch_size: 4\n',
        encoding="utf-8",
    )
    return cfg.load(path)


def _corpus(root: Path, n: int = 6) -> None:
    root.mkdir()
    for i in range(n):
        (root / f"doc{i}.html").write_text(
            f"<body><h1>Doc {i}</h1><p>content about topic {i} and timing and interfaces</p></body>",
            encoding="utf-8",
        )


def test_embeddings_stored_with_provenance(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    _corpus(root)
    config = _config(tmp_path, root)
    provider = FakeProvider()
    with Store.open(config.paths.db_path) as store:
        result = ingest.run_ingest(config, store, provider=provider)
        assert result.embedded == store.count_chunks()
        assert store.count_embeddings() == store.count_chunks()
        assert store.get_meta("embed_model") == "fake-v1"
        assert store.get_meta("embed_dim") == "8"


def test_none_skips_embeddings(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    _corpus(root)
    config = _config(tmp_path, root, model="none")
    with Store.open(config.paths.db_path) as store:
        result = ingest.run_ingest(config, store)  # provider from config -> None
        assert result.embedded == 0
        assert store.count_embeddings() == 0
        assert store.get_meta("embed_model") is None


def test_incremental_embeds_only_new_chunks(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    _corpus(root)
    config = _config(tmp_path, root)
    provider = FakeProvider()
    with Store.open(config.paths.db_path) as store:
        first = ingest.run_ingest(config, store, provider=provider)
        assert first.embedded > 0
        second = ingest.run_ingest(config, store, provider=provider)  # nothing changed
        assert second.embedded == 0
        assert store.count_embeddings() == store.count_chunks()
        # change one file -> only its new chunks embedded
        (root / "doc0.html").write_text(
            "<body><h1>Doc 0</h1><p>rewritten content with different words entirely</p></body>",
            encoding="utf-8",
        )
        third = ingest.run_ingest(config, store, provider=provider)
        assert third.embedded > 0
        assert store.count_embeddings() == store.count_chunks()  # no orphans/gaps


def test_switching_models_is_refused(tmp_path: Path) -> None:
    # Switching embed.model on an existing index (without --reembed) is refused with an
    # actionable message, rather than mixing embedding spaces (R-EMB-3).
    root = tmp_path / "docs"
    _corpus(root)
    config = _config(tmp_path, root)
    with Store.open(config.paths.db_path) as store:
        ingest.run_ingest(config, store, provider=FakeProvider("model-a"))
        with pytest.raises(embed.EmbedError, match="different models|reembed"):
            ingest.run_ingest(config, store, provider=FakeProvider("model-b"))


def test_force_reingest_cleanly_switches_model(tmp_path: Path) -> None:
    # --force is a full rebuild: it drops all vectors and re-embeds with the current model,
    # so switching models under --force yields a uniform index rather than mixing dims.
    root = tmp_path / "docs"
    _corpus(root)
    config = _config(tmp_path, root)
    with Store.open(config.paths.db_path) as store:
        ingest.run_ingest(config, store, provider=FakeProvider("model-a", dim=8))
        ingest.run_ingest(config, store, provider=FakeProvider("model-b", dim=16), force=True)
        assert store.existing_embedding_model() == ("model-b", 16)
        assert store.count_embeddings() == store.count_chunks()


def test_force_reembeds_even_stale_vectors_from_untouched_docs(tmp_path: Path) -> None:
    # Regression for Stephen's report: --force must forcefully re-embed even when old
    # vectors belong to documents that are NOT re-ingested this run (e.g. after a folder
    # rename left orphaned bge-large vectors). --force clears them up front instead of
    # tripping the model-mismatch guard on the stragglers.
    root = tmp_path / "docs"
    _corpus(root)
    config = _config(tmp_path, root)
    with Store.open(config.paths.db_path) as store:
        # a stale doc under a path NOT in the current source, with a big-model vector
        ghost = store.add_document(path="/old/renamed/ghost.html", source="old")
        gchunk = store.add_chunk(document_id=ghost, ord=0, text="ghost content to embed")
        store.add_embeddings([(gchunk, "BAAI/bge-large-en-v1.5", 1024, b"\x00" * 4 * 1024)])
        # a plain re-ingest would be refused; --force rebuilds everything cleanly
        ingest.run_ingest(config, store, provider=FakeProvider("small", dim=8), force=True)
        assert store.existing_embedding_model() == ("small", 8)
        assert store.count_embeddings() == store.count_chunks()  # uniform, no stale 1024-dim


def test_switching_models_refused_even_with_no_pending_chunks(tmp_path: Path) -> None:
    # red-team H1: a same-dim model swap on unchanged content (no new chunks to embed)
    # must still be refused — the guard runs before the no-pending early-out (R-EMB-3).
    root = tmp_path / "docs"
    _corpus(root)
    config = _config(tmp_path, root)
    with Store.open(config.paths.db_path) as store:
        ingest.run_ingest(config, store, provider=FakeProvider("model-a", dim=8))
        with pytest.raises(embed.EmbedError, match="different models|reembed"):
            ingest.run_ingest(config, store, provider=FakeProvider("model-b", dim=8))


def test_interrupted_partial_embed_then_switch_is_refused(tmp_path: Path) -> None:
    # Regression for the mixed-dimension crash: a killed embed run leaves vectors
    # committed but no embed_model meta; a later, different-dim model must be REFUSED
    # (clear error) instead of mixing dims and blowing up VectorIndex.build.
    root = tmp_path / "docs"
    _corpus(root)
    config = _config(tmp_path, root)
    with Store.open(config.paths.db_path) as store:
        ingest.run_ingest(config, store, provider=None)  # BM25 ingest: chunks, no vectors
        pending = store.chunks_without_embeddings()
        half = pending[: max(1, len(pending) // 2)]
        # simulate a big model that was interrupted: vectors committed, meta never written
        store.add_embeddings([(cid, "big-model", 1024, b"\x00" * 4 * 1024) for cid, _ in half])
        assert store.get_meta("embed_model") is None  # the exact gap that caused the bug
        with pytest.raises(embed.EmbedError, match="reembed|different models"):
            ingest.run_ingest(config, store, provider=FakeProvider("small", dim=8))


def test_reembed_heals_and_switches_model(tmp_path: Path) -> None:
    # `docusearch ingest --reembed`: drop existing vectors, re-embed all with the new
    # model, no crash — the supported way to switch models on an existing index.
    root = tmp_path / "docs"
    _corpus(root)
    config = _config(tmp_path, root)
    with Store.open(config.paths.db_path) as store:
        ingest.run_ingest(config, store, provider=FakeProvider("model-a", dim=8))
        assert store.existing_embedding_model() == ("model-a", 8)
        result = ingest.run_ingest(
            config, store, provider=FakeProvider("model-b", dim=16), reembed=True
        )
        assert result.embedded == store.count_chunks()
        assert store.count_embeddings() == store.count_chunks()  # no leftover model-a rows
        assert store.existing_embedding_model() == ("model-b", 16)
        assert store.get_meta("embed_model") == "model-b"
        assert store.get_meta("embed_dim") == "16"


def test_progress_callback_reports_ingest_and_embed(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    _corpus(root, n=6)
    config = _config(tmp_path, root)
    events: list[tuple[str, int, int]] = []
    with Store.open(config.paths.db_path) as store:
        ingest.run_ingest(
            config,
            store,
            provider=FakeProvider(dim=8),
            progress=lambda phase, done, total: events.append((phase, done, total)),
        )
    phases = {p for p, _, _ in events}
    assert {"ingest", "embed"} <= phases
    ingest_events = [(d, t) for p, d, t in events if p == "ingest"]
    assert ingest_events[-1][0] == ingest_events[-1][1] == 6  # finished all 6 files
    embed_events = [(d, t) for p, d, t in events if p == "embed"]
    assert embed_events[-1][0] == embed_events[-1][1]  # embed reached 100%


def test_embedding_vectors_roundtrip(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    _corpus(root, n=1)
    config = _config(tmp_path, root)
    provider = FakeProvider()
    with Store.open(config.paths.db_path) as store:
        ingest.run_ingest(config, store, provider=provider)
        stored = store.all_embeddings()
        assert stored
        _, blob = stored[0]
        vec = embed.from_blob(blob)
        assert vec.shape == (8,)
        assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-4)
