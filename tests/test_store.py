"""Schema + migration tests for store.py (§6, R-LOG-3).

store.py owns every DB read/write. These red tests pin the §6 schema, the
versioned/idempotent migration mechanism, the WAL + foreign-key pragmas, the meta
key/value store, and that the FTS5 external-content index actually stays in sync
with the chunks table via triggers.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from docusearch import store as st

EXPECTED_TABLES = {
    "meta",
    "documents",
    "chunks",
    "chunks_fts",
    "embeddings",
    "relations",
    "images",
    "flags",
    "annotations",
}


def test_open_creates_every_table(tmp_path: Path) -> None:
    with st.Store.open(tmp_path / "catalog.db") as store:
        assert store.table_names() >= EXPECTED_TABLES


def test_schema_version_is_current(tmp_path: Path) -> None:
    with st.Store.open(tmp_path / "catalog.db") as store:
        assert store.schema_version == st.SCHEMA_VERSION


def test_wal_mode_enabled(tmp_path: Path) -> None:
    with st.Store.open(tmp_path / "catalog.db") as store:
        mode = store._conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal"


def test_foreign_keys_enforced(tmp_path: Path) -> None:
    with st.Store.open(tmp_path / "catalog.db") as store:
        on = store._conn.execute("PRAGMA foreign_keys").fetchone()[0]
        assert on == 1
        # A chunk pointing at a nonexistent document must be rejected.
        with pytest.raises(sqlite3.IntegrityError):
            store.add_chunk(document_id=999, ord=0, text="orphan")


def test_migrations_are_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "catalog.db"
    with st.Store.open(db) as store:
        store.add_document(path="/a.html")
        v1 = store.schema_version
    # Re-opening an existing DB must not error or change the version.
    with st.Store.open(db) as store:
        assert store.schema_version == v1
        assert store.table_names() >= EXPECTED_TABLES
        assert store.count_documents() == 1  # data survived


def test_future_schema_version_is_rejected(tmp_path: Path) -> None:
    db = tmp_path / "catalog.db"
    with st.Store.open(db) as store:
        store.set_meta("schema_version", "999")
    with pytest.raises(st.StoreError, match="newer"):
        st.Store.open(db)


def test_meta_roundtrip_and_missing(tmp_path: Path) -> None:
    with st.Store.open(tmp_path / "catalog.db") as store:
        assert store.get_meta("does-not-exist") is None
        store.set_meta("embed_model", "sentence-transformers/all-MiniLM-L6-v2")
        assert store.get_meta("embed_model") == "sentence-transformers/all-MiniLM-L6-v2"
        # set overwrites
        store.set_meta("embed_model", "none")
        assert store.get_meta("embed_model") == "none"
        # 'created' is stamped on a fresh DB (R-LOG-3 provenance)
        assert store.get_meta("created") is not None


def test_add_document_returns_id_and_enforces_unique_path(tmp_path: Path) -> None:
    with st.Store.open(tmp_path / "catalog.db") as store:
        doc_id = store.add_document(path="/docs/spi.html", title="SPI")
        assert doc_id == 1
        assert store.count_documents() == 1
        with pytest.raises(sqlite3.IntegrityError):
            store.add_document(path="/docs/spi.html")  # duplicate path


def test_add_document_records_source_version(tmp_path: Path) -> None:
    with st.Store.open(tmp_path / "catalog.db") as store:
        doc_id = store.add_document(path="/a.html", source="vendor", source_version="2024.3")
        row = store.get_document(doc_id)
        assert row is not None
        assert row["source"] == "vendor"
        assert row["source_version"] == "2024.3"


def test_migrations_upgrade_a_v1_database(tmp_path: Path) -> None:
    # a database created at schema v1 (no source_version, no vision columns) must upgrade
    # cleanly to the current version and accept the new columns — additive migrations,
    # existing rows preserved.
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.executescript(st._SCHEMA_V1)
    conn.execute("INSERT INTO documents(path) VALUES('/legacy.html')")
    conn.execute("INSERT INTO meta(key, value) VALUES('schema_version', '1')")
    conn.commit()
    conn.close()
    with st.Store.open(db) as store:
        assert store.schema_version == st.SCHEMA_VERSION >= 3
        assert store.count_documents() == 1  # legacy row survived
        doc_id = store.add_document(path="/new.html", source_version="v9")
        row = store.get_document(doc_id)
        assert row is not None and row["source_version"] == "v9"
        cols = {r[1] for r in store._conn.execute("PRAGMA table_info(images)")}
        assert {"vision_text", "vision_model"} <= cols  # v3 columns present


def test_embedding_provenance_read_and_clear(tmp_path: Path) -> None:
    with st.Store.open(tmp_path / "catalog.db") as store:
        doc = store.add_document(path="/a.html")
        chunk = store.add_chunk(document_id=doc, ord=0, text="content")
        assert store.existing_embedding_model() is None
        store.add_embeddings([(chunk, "model-x", 8, b"\x00" * 32)])
        assert store.existing_embedding_model() == ("model-x", 8)
        assert store.clear_embeddings() == 1
        assert store.existing_embedding_model() is None
        assert store.count_embeddings() == 0
        assert store.get_meta("embed_model") is None  # provenance meta cleared too


def test_deferred_commits_batches_and_is_durable_on_commit(tmp_path: Path) -> None:
    db = tmp_path / "catalog.db"
    with st.Store.open(db) as store:
        with store.deferred_commits():
            doc = store.add_document(path="/a.html")
            store.add_chunk(document_id=doc, ord=0, text="x")
            # uncommitted: another connection still sees the last committed state
            with st.Store.open(db) as other:
                assert other.count_documents() == 0
            store.commit()  # explicit batch boundary
            with st.Store.open(db) as other:
                assert other.count_documents() == 1
        # autocommit restored after the block: this write is durable immediately
        store.add_document(path="/b.html")
        with st.Store.open(db) as other:
            assert other.count_documents() == 2


def test_deferred_commits_rolls_back_on_error(tmp_path: Path) -> None:
    db = tmp_path / "catalog.db"
    with st.Store.open(db) as store:
        with pytest.raises(RuntimeError), store.deferred_commits():
            store.add_document(path="/a.html")
            raise RuntimeError("boom mid-batch")
        assert store.count_documents() == 0  # the whole uncommitted batch rolled back
        store.add_document(path="/c.html")  # autocommit restored -> commits normally
        with st.Store.open(db) as other:
            assert other.count_documents() == 1


def test_source_names_and_ids(tmp_path: Path) -> None:
    with st.Store.open(tmp_path / "catalog.db") as store:
        store.add_document(path="/a.html", source="x")
        store.add_document(path="/b.html", source="x")
        store.add_document(path="/c.html", source="y")
        assert dict(store.source_names()) == {"x": 2, "y": 1}
        assert len(store.document_ids_for_source("x")) == 2
        assert store.document_ids_for_source("z") == []


def test_fts_insert_trigger_keeps_index_in_sync(tmp_path: Path) -> None:
    with st.Store.open(tmp_path / "catalog.db") as store:
        doc = store.add_document(path="/docs/spi.html")
        chunk = store.add_chunk(
            document_id=doc,
            ord=0,
            kind="body",
            locator="Interfaces > SPI > Timing",
            text="SPI timing configuration for the peripheral bus",
        )
        assert store.count_chunks() == 1
        assert store.chunk_ids_matching("timing") == [chunk]
        assert store.chunk_ids_matching("nonexistentterm") == []


def test_open_creates_parent_directory(tmp_path: Path) -> None:
    # Windows-first: pathlib mkdir, no assumption the parent exists (R-ARCH-5).
    db = tmp_path / "deep" / "nested" / "catalog.db"
    with st.Store.open(db):
        pass
    assert db.exists()


def test_in_memory_store(tmp_path: Path) -> None:
    with st.Store.open(":memory:") as store:
        assert store.table_names() >= EXPECTED_TABLES
        assert store.schema_version == st.SCHEMA_VERSION
