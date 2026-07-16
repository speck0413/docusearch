"""SQLite persistence: the one and only place that touches the database (§6).

Every schema definition, migration, read, and write lives here (R-ARCH-3, R-REUSE-2).
The database is a single file per index (§6). We use SQLite FTS5 for BM25 (R-SRCH-1)
with an *external-content* index over ``chunks.text`` kept in sync by triggers, WAL
mode for concurrent readers during ingest, and enforced foreign keys.

Public surface:
    Store               -- context-managed connection wrapper
    Store.open(path)    -- connect + migrate; returns a ready Store
    StoreError          -- raised on schema/version problems
    SCHEMA_VERSION      -- integer the code understands

Phase 0 provides schema + migrations + the minimal document/chunk write path needed
to prove the FTS wiring; ranking (bm25()) and the richer read paths arrive with
search.py in later phases.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType

SCHEMA_VERSION = 1

_MEMORY = ":memory:"


class StoreError(Exception):
    """A database is unusable by this build (e.g. its schema is newer than the code)."""


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat()


# --------------------------------------------------------------------------- schema
# Migration 1 = the full §6 schema. Later phases append migrations; they never edit
# an already-shipped one.

_SCHEMA_V1 = """
CREATE TABLE meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE documents (
    id           INTEGER PRIMARY KEY,
    path         TEXT UNIQUE NOT NULL,
    source       TEXT,
    title        TEXT,
    doc_id       TEXT,
    content_hash TEXT,
    content_type TEXT,              -- documentation | code
    fmt          TEXT,              -- html | pdf | docx | md | ...
    audience     TEXT,              -- JSON list
    mtime        REAL,
    ingested_at  TEXT,
    status       TEXT               -- active | superseded
);
CREATE INDEX idx_documents_content_hash ON documents(content_hash);
CREATE INDEX idx_documents_status ON documents(status);

CREATE TABLE chunks (
    id          INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id),
    ord         INTEGER NOT NULL,
    kind        TEXT NOT NULL,      -- body | code | gotcha | enrichment | image_ref
    locator     TEXT,               -- "H1>H2>H3" heading path or "page 7"
    text        TEXT NOT NULL
);
CREATE INDEX idx_chunks_document_id ON chunks(document_id);

-- BM25 index over chunk text (external content -> no duplicated text storage).
CREATE VIRTUAL TABLE chunks_fts USING fts5(text, content='chunks', content_rowid='id');

-- Keep the FTS index in lock-step with the chunks table (R-SRCH-1).
CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;
CREATE TRIGGER chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TABLE embeddings (
    chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id),
    model    TEXT NOT NULL,         -- provenance: which model produced the vector (R-EMB-2)
    dim      INTEGER NOT NULL,
    vector   BLOB NOT NULL
);

CREATE TABLE relations (
    id         INTEGER PRIMARY KEY,
    src_doc    INTEGER NOT NULL REFERENCES documents(id),
    dst_doc    INTEGER REFERENCES documents(id),   -- NULL until/if resolved
    dst_raw    TEXT,                                -- original href/target as written
    link_type  TEXT,               -- html_href | md_link | docx_hyperlink | code_ref | ai_inferred
    confidence REAL,
    created_at TEXT
);
CREATE INDEX idx_relations_src ON relations(src_doc);
CREATE INDEX idx_relations_dst ON relations(dst_doc);

CREATE TABLE images (
    sha256  TEXT PRIMARY KEY,
    ext     TEXT,
    doc_id  INTEGER REFERENCES documents(id),
    locator TEXT,
    alt     TEXT,
    caption TEXT,
    bytes   INTEGER
);
CREATE INDEX idx_images_doc ON images(doc_id);

CREATE TABLE flags (
    id         INTEGER PRIMARY KEY,
    doc_id     INTEGER REFERENCES documents(id),
    chunk_id   INTEGER REFERENCES chunks(id),
    kind       TEXT,                -- gotcha | discrepancy | ...
    source     TEXT,
    rule_id    TEXT,
    note       TEXT,
    created_at TEXT
);
CREATE INDEX idx_flags_doc ON flags(doc_id);
CREATE INDEX idx_flags_chunk ON flags(chunk_id);

CREATE TABLE annotations (
    id            INTEGER PRIMARY KEY,
    document_path TEXT,
    origin        TEXT,             -- scripted | ai | user | code
    author        TEXT,
    text          TEXT,
    audience      TEXT,
    created_at    TEXT
);
"""

_MIGRATIONS: tuple[tuple[int, str], ...] = ((1, _SCHEMA_V1),)


class Store:
    """A live connection to one catalog database. Use as a context manager."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    # -- lifecycle ---------------------------------------------------------------

    @classmethod
    def open(cls, db_path: Path | str) -> Store:
        """Connect to (creating if needed) the database and bring it to SCHEMA_VERSION."""
        if isinstance(db_path, str) and db_path == _MEMORY:
            target = _MEMORY
        else:
            path = Path(db_path)
            if path.parent != Path():
                path.parent.mkdir(parents=True, exist_ok=True)
            target = str(path)
        conn = sqlite3.connect(target)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA synchronous=NORMAL")
            store = cls(conn)
            store._migrate()
        except Exception:
            conn.close()  # never leak a connection when open() fails
            raise
        return store

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # -- migrations --------------------------------------------------------------

    def _table_exists(self, name: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        return row is not None

    def _read_version(self) -> int:
        if not self._table_exists("meta"):
            return 0
        row = self._conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        return int(row[0]) if row is not None else 0

    def _migrate(self) -> None:
        current = self._read_version()
        if current > SCHEMA_VERSION:
            raise StoreError(
                f"Database schema version {current} is newer than this build "
                f"(supports {SCHEMA_VERSION}). Upgrade docusearch or use a matching DB."
            )
        for version, sql in _MIGRATIONS:
            if version > current:
                self._conn.executescript(sql)
                self.set_meta("schema_version", str(version))
        if self.get_meta("created") is None:
            self.set_meta("created", _utcnow_iso())

    @property
    def schema_version(self) -> int:
        return self._read_version()

    # -- meta key/value (schema_version, embed_model, embed_dim, config_hash) -----

    def get_meta(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return None if row is None else str(row[0])

    def set_meta(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO meta(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        self._conn.commit()

    # -- introspection (used by the audit + the red team's recount) ---------------

    def table_names(self) -> set[str]:
        rows = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
        ).fetchall()
        return {str(r[0]) for r in rows}

    def count_documents(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0])

    def count_chunks(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    # -- minimal write path (Phase 1 builds ingestion on top of these) ------------

    def add_document(
        self,
        *,
        path: str,
        source: str = "",
        title: str = "",
        doc_id: str = "",
        content_hash: str = "",
        content_type: str = "documentation",
        fmt: str = "html",
        audience: list[str] | None = None,
        mtime: float = 0.0,
        status: str = "active",
    ) -> int:
        """Insert a document row; returns its new integer id. ``path`` must be unique."""
        cur = self._conn.execute(
            "INSERT INTO documents"
            "(path, source, title, doc_id, content_hash, content_type, fmt, "
            " audience, mtime, ingested_at, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                path,
                source,
                title,
                doc_id,
                content_hash,
                content_type,
                fmt,
                json.dumps(audience or []),
                mtime,
                _utcnow_iso(),
                status,
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid or 0)

    def add_chunk(
        self,
        *,
        document_id: int,
        ord: int,
        text: str,
        kind: str = "body",
        locator: str = "",
    ) -> int:
        """Insert a chunk; the FTS index is updated by trigger. Returns the new id."""
        cur = self._conn.execute(
            "INSERT INTO chunks(document_id, ord, kind, locator, text) VALUES (?, ?, ?, ?, ?)",
            (document_id, ord, kind, locator, text),
        )
        self._conn.commit()
        return int(cur.lastrowid or 0)

    def chunk_ids_matching(self, query: str) -> list[int]:
        """Raw FTS5 rowid match for ``query`` (unranked). Ranking lands in search.py.

        ``query`` is passed straight to FTS5 MATCH here; search.py owns the sanitizer
        that quotes arbitrary user text before it reaches this method (R-SRCH: §9).
        """
        rows = self._conn.execute(
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY rowid", (query,)
        ).fetchall()
        return [int(r[0]) for r in rows]
