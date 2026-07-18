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
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType

SCHEMA_VERSION = 3

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

# Migration 2 = per-document source version label (provenance for "which release of the
# docs did this come from"). Additive column; NULL for rows ingested before v2.
_SCHEMA_V2 = "ALTER TABLE documents ADD COLUMN source_version TEXT;"

# Migration 3 = vision enrichment provenance on images (enrich.vision_images). The stored
# OCR text + producing model are what keep search deterministic (R-SRCH-5): the vision API
# is called once here, never at query time. NULL vision_model = "not yet enriched".
_SCHEMA_V3 = (
    "ALTER TABLE images ADD COLUMN vision_text TEXT;\n"
    "ALTER TABLE images ADD COLUMN vision_model TEXT;"
)

_MIGRATIONS: tuple[tuple[int, str], ...] = (
    (1, _SCHEMA_V1),
    (2, _SCHEMA_V2),
    (3, _SCHEMA_V3),
)


class Store:
    """A live connection to one catalog database. Use as a context manager."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._autocommit = True  # per-write commits; ingest defers these to batch fsyncs

    # -- lifecycle ---------------------------------------------------------------

    def _maybe_commit(self) -> None:
        """Commit now unless inside a ``deferred_commits`` block (bulk ingest batches)."""
        if self._autocommit:
            self._conn.commit()

    def commit(self) -> None:
        """Flush pending writes to disk (one fsync). Used to batch a deferred block."""
        self._conn.commit()

    @contextmanager
    def deferred_commits(self) -> Iterator[None]:
        """Suppress per-write commits inside the block so the caller can batch them.

        Each ``add_*`` normally fsyncs on commit; on a bulk ingest that is hundreds of
        thousands of fsyncs (the real "13% CPU, slow" cause). Inside this block the writes
        accumulate in one transaction until the caller calls :meth:`commit`; on exit any
        remainder is committed (or rolled back if the block raised)."""
        previous = self._autocommit
        self._autocommit = False
        try:
            yield
            self._conn.commit()
        except BaseException:
            self._conn.rollback()
            raise
        finally:
            self._autocommit = previous

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
        source_version: str = "",
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
            "(path, source, source_version, title, doc_id, content_hash, content_type, fmt, "
            " audience, mtime, ingested_at, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                path,
                source,
                source_version,
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
        self._maybe_commit()
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
        self._maybe_commit()
        return int(cur.lastrowid or 0)

    def add_chunks(self, document_id: int, chunks: Sequence[tuple[int, str, str, str]]) -> None:
        """Bulk-insert a document's chunks in one ``executemany`` (ord, text, kind, locator).

        The FTS index stays in sync via trigger. This is the ingest hot path — one call per
        document instead of one per chunk cuts the Python↔SQLite round-trips substantially."""
        if not chunks:
            return
        self._conn.executemany(
            "INSERT INTO chunks(document_id, ord, kind, locator, text) VALUES (?, ?, ?, ?, ?)",
            [(document_id, ordv, kind, locator, text) for ordv, text, kind, locator in chunks],
        )
        self._maybe_commit()

    def chunk_ids_matching(self, query: str) -> list[int]:
        """Raw FTS5 rowid match for ``query`` (unranked). Ranking lands in search.py.

        ``query`` is passed straight to FTS5 MATCH here; search.py owns the sanitizer
        that quotes arbitrary user text before it reaches this method (R-SRCH: §9).
        """
        rows = self._conn.execute(
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY rowid", (query,)
        ).fetchall()
        return [int(r[0]) for r in rows]

    # -- ingest read/write paths (Phase 1) ---------------------------------------

    def document_content_hash(self, path: str) -> str | None:
        """The stored content hash for a path, or None if unseen (incremental skip)."""
        row = self._conn.execute(
            "SELECT content_hash FROM documents WHERE path=?", (path,)
        ).fetchone()
        return None if row is None else str(row[0])

    def document_id_for_path(self, path: str) -> int | None:
        row = self._conn.execute("SELECT id FROM documents WHERE path=?", (path,)).fetchone()
        return None if row is None else int(row[0])

    def document_hashes(self) -> dict[str, str]:
        """``{path: content_hash}`` for every document — the incremental-skip lookup, read
        once so parallel parse workers can decide skip vs. re-parse without the DB."""
        rows = self._conn.execute(
            "SELECT path, content_hash FROM documents WHERE content_hash IS NOT NULL"
        ).fetchall()
        return {str(r[0]): str(r[1]) for r in rows}

    def delete_document(self, doc_id: int) -> None:
        """Remove a document and everything it owns (for re-ingest of a changed file).

        Incoming relations (other docs -> this one) are kept but reset to unresolved so
        the link post-pass re-resolves them to the replacement document.
        """
        c = self._conn
        with c:
            c.execute(
                "DELETE FROM embeddings WHERE chunk_id IN "
                "(SELECT id FROM chunks WHERE document_id=?)",
                (doc_id,),
            )
            c.execute(
                "DELETE FROM flags WHERE doc_id=? OR chunk_id IN "
                "(SELECT id FROM chunks WHERE document_id=?)",
                (doc_id, doc_id),
            )
            c.execute("DELETE FROM chunks WHERE document_id=?", (doc_id,))
            c.execute("DELETE FROM relations WHERE src_doc=?", (doc_id,))
            c.execute("UPDATE relations SET dst_doc=NULL WHERE dst_doc=?", (doc_id,))
            c.execute("DELETE FROM images WHERE doc_id=?", (doc_id,))
            c.execute("DELETE FROM documents WHERE id=?", (doc_id,))

    def document_ids_for_source(self, source: str) -> list[int]:
        """Every document id ingested under the given source label (for purge)."""
        rows = self._conn.execute(
            "SELECT id FROM documents WHERE source=? ORDER BY id", (source,)
        ).fetchall()
        return [int(r[0]) for r in rows]

    def all_document_paths(self) -> list[tuple[int, str]]:
        """``(id, path)`` for every document — used to prune ones whose source file is gone."""
        rows = self._conn.execute("SELECT id, path FROM documents ORDER BY id").fetchall()
        return [(int(r[0]), str(r[1])) for r in rows]

    def source_names(self) -> list[tuple[str, int]]:
        """``(source, document_count)`` for every distinct source label, most docs first."""
        rows = self._conn.execute(
            "SELECT COALESCE(source, ''), COUNT(*) FROM documents "
            "GROUP BY source ORDER BY COUNT(*) DESC, source"
        ).fetchall()
        return [(str(r[0]), int(r[1])) for r in rows]

    def add_relation(
        self, *, src_doc: int, dst_raw: str, link_type: str, confidence: float | None = None
    ) -> int:
        cur = self._conn.execute(
            "INSERT INTO relations(src_doc, dst_doc, dst_raw, link_type, confidence, created_at) "
            "VALUES (?, NULL, ?, ?, ?, ?)",
            (src_doc, dst_raw, link_type, confidence, _utcnow_iso()),
        )
        self._maybe_commit()
        return int(cur.lastrowid or 0)

    def set_relation_dst(self, relation_id: int, dst_doc: int) -> None:
        self._conn.execute("UPDATE relations SET dst_doc=? WHERE id=?", (dst_doc, relation_id))
        self._maybe_commit()

    def unresolved_relations(self) -> list[tuple[int, str, str]]:
        """(relation_id, source document path, raw target) for every unresolved link."""
        rows = self._conn.execute(
            "SELECT r.id, d.path, r.dst_raw FROM relations r "
            "JOIN documents d ON r.src_doc = d.id WHERE r.dst_doc IS NULL"
        ).fetchall()
        return [(int(r[0]), str(r[1]), str(r[2])) for r in rows]

    def document_path_to_id(self) -> dict[str, int]:
        rows = self._conn.execute("SELECT path, id FROM documents").fetchall()
        return {str(r[0]): int(r[1]) for r in rows}

    def add_image(
        self,
        *,
        sha256: str,
        ext: str,
        doc_id: int,
        locator: str,
        alt: str,
        caption: str,
        num_bytes: int,
    ) -> None:
        """Record an image (dedup by content sha; originals live under staging)."""
        self._conn.execute(
            "INSERT OR IGNORE INTO images(sha256, ext, doc_id, locator, alt, caption, bytes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (sha256, ext, doc_id, locator, alt, caption, num_bytes),
        )
        self._maybe_commit()

    def count_relations(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0])

    def count_resolved_relations(self) -> int:
        return int(
            self._conn.execute(
                "SELECT COUNT(*) FROM relations WHERE dst_doc IS NOT NULL"
            ).fetchone()[0]
        )

    def count_images(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM images").fetchone()[0])

    # -- vision enrichment (enrich.vision_images) --------------------------------

    def images_needing_vision(self, *, limit: int | None = None) -> list[sqlite3.Row]:
        """Retained images with no vision result yet, ordered by sha for determinism.

        ``vision_model IS NULL`` is the worklist marker, so a re-run only processes images
        that were never enriched (idempotent, and no double API spend)."""
        sql = (
            "SELECT sha256, ext, doc_id, locator, alt, caption FROM images "
            "WHERE vision_model IS NULL ORDER BY sha256"
        )
        if limit is not None:
            return self._conn.execute(sql + " LIMIT ?", (limit,)).fetchall()
        return self._conn.execute(sql).fetchall()

    def set_image_vision(self, sha256: str, text: str, model: str) -> None:
        """Persist an image's vision OCR text + producing model (provenance, R-SRCH-5)."""
        self._conn.execute(
            "UPDATE images SET vision_text=?, vision_model=? WHERE sha256=?",
            (text, model, sha256),
        )
        self._maybe_commit()

    def add_enrichment_chunk(
        self, document_id: int, text: str, locator: str, *, kind: str = "enrichment"
    ) -> int:
        """Append an AI-generated searchable chunk to a document (kind='enrichment', §6).

        Assigned the next ``ord`` after the document's existing chunks, so it never collides
        with the ingest-time chunks. Firing the FTS trigger makes it BM25-searchable at once;
        the caller embeds it if the index is hybrid."""
        row = self._conn.execute(
            "SELECT COALESCE(MAX(ord), -1) FROM chunks WHERE document_id=?", (document_id,)
        ).fetchone()
        ordv = int(row[0]) + 1
        cur = self._conn.execute(
            "INSERT INTO chunks(document_id, ord, kind, locator, text) VALUES (?, ?, ?, ?, ?)",
            (document_id, ordv, kind, locator, text),
        )
        self._maybe_commit()
        return int(cur.lastrowid or 0)

    # -- embeddings (Phase 2) ----------------------------------------------------

    def add_embeddings(self, rows: Sequence[tuple[int, str, int, bytes]]) -> None:
        """Insert (chunk_id, model, dim, vector-blob) rows in one batch (R-EMB-2)."""
        self._conn.executemany(
            "INSERT OR REPLACE INTO embeddings(chunk_id, model, dim, vector) VALUES (?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()

    def chunks_without_embeddings(self) -> list[tuple[int, str]]:
        """(id, text) for chunks with no vector yet — the incremental embed worklist."""
        rows = self._conn.execute(
            "SELECT c.id, c.text FROM chunks c "
            "LEFT JOIN embeddings e ON e.chunk_id = c.id "
            "WHERE e.chunk_id IS NULL ORDER BY c.id"
        ).fetchall()
        return [(int(r[0]), str(r[1])) for r in rows]

    def all_embeddings(self) -> list[tuple[int, bytes]]:
        """(chunk_id, vector-blob) for every embedding, ordered by chunk id (ANN build)."""
        rows = self._conn.execute(
            "SELECT chunk_id, vector FROM embeddings ORDER BY chunk_id"
        ).fetchall()
        return [(int(r[0]), bytes(r[1])) for r in rows]

    def count_embeddings(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0])

    def existing_embedding_model(self) -> tuple[str, int] | None:
        """The ``(model, dim)`` of the vectors already in the index, or ``None`` if empty.

        Read authoritatively from the embeddings rows themselves — not the ``embed_model``
        meta flag, which is only written after a full pass and so is absent if a run was
        interrupted (that gap is what let mixed-dimension vectors accumulate)."""
        row = self._conn.execute("SELECT model, dim FROM embeddings LIMIT 1").fetchone()
        return (str(row[0]), int(row[1])) if row is not None else None

    def clear_embeddings(self) -> int:
        """Drop every vector and its provenance meta; returns how many were removed.

        The recovery path for switching embedding models (or healing a mixed index):
        the chunks/documents stay, only the vectors are rebuilt on the next embed pass."""
        n = self.count_embeddings()
        with self._conn:
            self._conn.execute("DELETE FROM embeddings")
            self._conn.execute("DELETE FROM meta WHERE key IN ('embed_model', 'embed_dim')")
        return n

    # -- document/audit read paths (Phase 1) -------------------------------------

    def get_document(self, doc_id: int) -> sqlite3.Row | None:
        row: sqlite3.Row | None = self._conn.execute(
            "SELECT * FROM documents WHERE id=?", (doc_id,)
        ).fetchone()
        return row

    def chunks_for_document(self, doc_id: int) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT id, ord, kind, locator, text FROM chunks WHERE document_id=? ORDER BY ord",
            (doc_id,),
        ).fetchall()

    def citation_target(self, doc_id: int, chunk_id: int) -> tuple[str, str, str, str] | None:
        """``(source, title, original path, chunk locator)`` for one cited pair, so a report
        can link a reference to the original vendor document (not the chunk). None if the
        chunk doesn't belong to the document (a mis-attributed citation)."""
        row = self._conn.execute(
            "SELECT d.source, d.title, d.path, c.locator "
            "FROM chunks c JOIN documents d ON c.document_id = d.id "
            "WHERE c.id=? AND d.id=?",
            (chunk_id, doc_id),
        ).fetchone()
        if row is None:
            return None
        return (str(row[0] or ""), str(row[1] or ""), str(row[2] or ""), str(row[3] or ""))

    def get_image(self, sha256: str) -> sqlite3.Row | None:
        row: sqlite3.Row | None = self._conn.execute(
            "SELECT * FROM images WHERE sha256=?", (sha256,)
        ).fetchone()
        return row

    def relations_out(self, doc_id: int) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT dst_doc, dst_raw, link_type FROM relations WHERE src_doc=? ORDER BY id",
            (doc_id,),
        ).fetchall()

    def relations_in(self, doc_id: int) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT src_doc, dst_raw, link_type FROM relations WHERE dst_doc=? ORDER BY id",
            (doc_id,),
        ).fetchall()

    def fmt_histogram(self) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT fmt, COUNT(*) FROM documents GROUP BY fmt ORDER BY fmt"
        ).fetchall()
        return {str(r[0]): int(r[1]) for r in rows}

    def documents_without_chunks(self) -> int:
        return int(
            self._conn.execute(
                "SELECT COUNT(*) FROM documents d "
                "WHERE NOT EXISTS (SELECT 1 FROM chunks c WHERE c.document_id = d.id)"
            ).fetchone()[0]
        )

    def documents_with_empty_audience(self) -> int:
        return int(
            self._conn.execute(
                "SELECT COUNT(*) FROM documents WHERE audience IS NULL OR audience IN ('', '[]')"
            ).fetchone()[0]
        )

    # -- search read path (Phase 1) ----------------------------------------------

    def bm25(self, match: str, limit: int) -> list[sqlite3.Row]:
        """Ranked BM25 rows for a *sanitized* FTS query (search.py owns the sanitizer).

        Ordered best-first with a deterministic tie-break on (doc id, chunk id) so an
        identical index + query always yields identical ranked results (R-SRCH-5).
        """
        return self._conn.execute(
            "SELECT c.document_id AS doc_id, c.id AS chunk_id, c.kind AS kind, "
            "c.locator AS locator, d.title AS title, d.path AS path, d.fmt AS fmt, "
            "d.audience AS audience, "
            "snippet(chunks_fts, 0, '', '', ' … ', 12) AS snippet, "
            "bm25(chunks_fts) AS bm25 "
            "FROM chunks_fts "
            "JOIN chunks c ON c.id = chunks_fts.rowid "
            "JOIN documents d ON d.id = c.document_id "
            "WHERE chunks_fts MATCH ? "
            "ORDER BY bm25(chunks_fts) ASC, c.document_id ASC, c.id ASC "
            "LIMIT ?",
            (match, limit),
        ).fetchall()

    def hydrate_chunks(self, chunk_ids: Sequence[int]) -> dict[int, sqlite3.Row]:
        """Fetch chunk + owning-document fields for a set of chunk ids (hybrid fusion)."""
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" * len(chunk_ids))
        rows = self._conn.execute(
            "SELECT c.id AS chunk_id, c.document_id AS doc_id, c.kind AS kind, "
            "c.locator AS locator, c.text AS text, d.title AS title, d.path AS path, "
            "d.fmt AS fmt, d.audience AS audience "
            "FROM chunks c JOIN documents d ON d.id = c.document_id "
            f"WHERE c.id IN ({placeholders})",
            tuple(chunk_ids),
        ).fetchall()
        return {int(r["chunk_id"]): r for r in rows}
