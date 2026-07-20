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

SCHEMA_VERSION = 7

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

# Migration 4 = structured STDF test-data tables (GATE 6, R-STDF-2). Alongside the searchable test
# chunks, numeric results + part touchdowns land in real columns so **non-AI tools** (plain SQL, a
# thin web UI) can query the data and drive the plot engine directly, unaided.
_SCHEMA_V4 = """
CREATE TABLE stdf_results (
    id         INTEGER PRIMARY KEY,
    doc_id     INTEGER REFERENCES documents(id),
    chunk_id   INTEGER REFERENCES chunks(id),
    test_num   INTEGER,
    test_txt   TEXT,
    result     REAL,
    units      TEXT,
    head       INTEGER,
    site       INTEGER,
    part_id    TEXT,
    insertion  TEXT,
    passed     INTEGER
);
CREATE INDEX idx_stdf_results_doc  ON stdf_results(doc_id);
CREATE INDEX idx_stdf_results_test ON stdf_results(test_num);
CREATE TABLE stdf_parts (
    id         INTEGER PRIMARY KEY,
    doc_id     INTEGER REFERENCES documents(id),
    part_id    TEXT,
    insertion  TEXT,
    lot        TEXT,
    sublot     TEXT,
    wafer      TEXT,
    x          INTEGER,
    y          INTEGER,
    head       INTEGER,
    site       INTEGER,
    hard_bin   INTEGER,
    soft_bin   INTEGER,
    passed     INTEGER
);
CREATE INDEX idx_stdf_parts_doc   ON stdf_parts(doc_id);
CREATE INDEX idx_stdf_parts_wafer ON stdf_parts(wafer);
"""

# Migration 5 = generic columnar data tables (Phase 10). STDF is just one thing a data store can
# hold; a CSV (or any table) lands here as named numeric columns + values so the SAME non-AI query
# and plot tools work on arbitrary data, not just STDF.
_SCHEMA_V5 = """
CREATE TABLE data_columns (
    id       INTEGER PRIMARY KEY,
    doc_id   INTEGER REFERENCES documents(id),
    dataset  TEXT,
    name     TEXT,
    kind     TEXT,               -- numeric | categorical | text
    units    TEXT,
    lo       REAL,
    hi       REAL,
    n        INTEGER
);
CREATE INDEX idx_data_columns_doc ON data_columns(doc_id);
CREATE TABLE data_values (
    col_id   INTEGER REFERENCES data_columns(id),
    row_idx  INTEGER,
    value    REAL,
    grp      TEXT
);
CREATE INDEX idx_data_values_col ON data_values(col_id);
"""

# Migration 6 = feedback as first-class, ranking-eligible data (Phase 8 / R-FB). A user's feedback is
# stored **per-user** (private to them) with an option to promote it to **global** (everyone). It
# carries a rating and an optional (doc_id, chunk_id) target so it can adjust search ranking and be
# reconciled against the source hierarchy (feedback > internal > vendor).
_SCHEMA_V6 = """
CREATE TABLE feedback (
    id       INTEGER PRIMARY KEY,
    ts       TEXT,
    author   TEXT,
    scope    TEXT,               -- 'user' (private to author) | 'global' (everyone)
    doc_id   INTEGER,
    chunk_id INTEGER,
    rating   INTEGER,            -- -1 (wrong/down) | 0 (note) | +1 (right/up)
    text     TEXT
);
CREATE INDEX idx_feedback_author ON feedback(author);
CREATE INDEX idx_feedback_target ON feedback(doc_id, chunk_id);
"""

# Migration 7 = structured source-code symbols (Phase 9 / GATE 9). A `store_type: code` store parses
# each file into functions/classes/methods; alongside the searchable per-symbol chunk, the symbol's
# structured fields land here so non-AI tools (the `code` CLI/MCP, the style-guide deriver) can query
# by language / kind / name without re-parsing.
_SCHEMA_V7 = """
CREATE TABLE code_symbols (
    id         INTEGER PRIMARY KEY,
    doc_id     INTEGER REFERENCES documents(id),
    chunk_id   INTEGER REFERENCES chunks(id),
    language   TEXT,
    kind       TEXT,               -- function | method | class | struct | interface | trait | enum | type
    name       TEXT,
    qualname   TEXT,
    signature  TEXT,
    docstring  TEXT,
    start_line INTEGER,
    end_line   INTEGER,
    parent     TEXT,
    path       TEXT
);
CREATE INDEX idx_code_symbols_doc  ON code_symbols(doc_id);
CREATE INDEX idx_code_symbols_kind ON code_symbols(language, kind);
CREATE INDEX idx_code_symbols_name ON code_symbols(name);
"""

_MIGRATIONS: tuple[tuple[int, str], ...] = (
    (1, _SCHEMA_V1),
    (2, _SCHEMA_V2),
    (3, _SCHEMA_V3),
    (4, _SCHEMA_V4),
    (5, _SCHEMA_V5),
    (6, _SCHEMA_V6),
    (7, _SCHEMA_V7),
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

    def add_flag(
        self,
        *,
        doc_id: int,
        chunk_id: int | None,
        kind: str,
        source: str = "",
        rule_id: str = "",
        note: str = "",
    ) -> int:
        """Record a ``flags`` row (R-ING-8 gotcha, discrepancy, …) — the UI/report-filterable
        half of dual-tagging. Returns the new id."""
        cur = self._conn.execute(
            "INSERT INTO flags(doc_id, chunk_id, kind, source, rule_id, note, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (doc_id, chunk_id, kind, source, rule_id, note, _utcnow_iso()),
        )
        self._maybe_commit()
        return int(cur.lastrowid or 0)

    def count_flags(self, kind: str | None = None) -> int:
        if kind is None:
            return int(self._conn.execute("SELECT COUNT(*) FROM flags").fetchone()[0])
        return int(
            self._conn.execute("SELECT COUNT(*) FROM flags WHERE kind=?", (kind,)).fetchone()[0]
        )

    def flags_for_document(self, doc_id: int) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT id, chunk_id, kind, source, rule_id, note FROM flags "
            "WHERE doc_id=? ORDER BY id",
            (doc_id,),
        ).fetchall()

    def clear_flags(self, kind: str) -> int:
        """Delete every flag of one ``kind`` — a discrepancy re-scan replaces its prior findings."""
        cur = self._conn.execute("DELETE FROM flags WHERE kind=?", (kind,))
        self._maybe_commit()
        return int(cur.rowcount)

    def duplicate_active_documents(self) -> list[tuple[str, list[tuple[int, str]]]]:
        """Groups of ACTIVE documents that share a content_hash (byte-identical files ingested
        under different paths/sources) — the duplicate half of the discrepancy scan (§17). Each
        group is ``(content_hash, [(doc_id, path), …])``; only hashes with ≥2 active docs."""
        rows = self._conn.execute(
            "SELECT content_hash, id, path FROM documents "
            "WHERE status='active' AND content_hash IN ("
            "  SELECT content_hash FROM documents WHERE status='active' "
            "  GROUP BY content_hash HAVING COUNT(*) > 1"
            ") ORDER BY content_hash, id"
        ).fetchall()
        groups: dict[str, list[tuple[int, str]]] = {}
        for h, did, path in rows:
            groups.setdefault(str(h), []).append((int(did), str(path)))
        return [(h, groups[h]) for h in sorted(groups)]

    def chunk_doc_map(self) -> dict[int, int]:
        """``{chunk_id: document_id}`` for every chunk — used to tell same-doc near-duplicates
        (uninteresting) from cross-document conflict candidates in the discrepancy scan."""
        return {
            int(r[0]): int(r[1])
            for r in self._conn.execute("SELECT id, document_id FROM chunks")
        }

    def flagged_chunks(self, kind: str, limit: int = 0) -> list[sqlite3.Row]:
        """Flagged (chunk, doc, rule) rows for one ``kind`` — for report filters and precision
        sampling at the gate. ``limit`` 0 means all, ordered deterministically by flag id."""
        sql = (
            "SELECT f.id AS flag_id, f.rule_id, f.note, d.id AS doc_id, d.path, d.title, "
            "c.id AS chunk_id, c.ord, c.locator, c.text "
            "FROM flags f JOIN documents d ON f.doc_id = d.id "
            "LEFT JOIN chunks c ON f.chunk_id = c.id "
            "WHERE f.kind=? ORDER BY f.id"
        )
        if limit > 0:
            sql += " LIMIT ?"
            return self._conn.execute(sql, (kind, limit)).fetchall()
        return self._conn.execute(sql, (kind,)).fetchall()

    # -- structured STDF data (GATE 6, R-STDF-2) — non-AI-queryable ----------------

    def add_stdf_results(self, rows: Sequence[tuple]) -> None:  # type: ignore[type-arg]
        """Bulk-insert numeric test results (doc_id, chunk_id, test_num, test_txt, result, units,
        head, site, part_id, insertion, passed) — the columns a non-AI tool / web UI queries."""
        if rows:
            self._conn.executemany(
                "INSERT INTO stdf_results(doc_id, chunk_id, test_num, test_txt, result, units, "
                "head, site, part_id, insertion, passed) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                rows,
            )
            self._maybe_commit()

    def add_stdf_parts(self, rows: Sequence[tuple]) -> None:  # type: ignore[type-arg]
        """Bulk-insert part touchdowns (doc_id, part_id, insertion, lot, sublot, wafer, x, y, head,
        site, hard_bin, soft_bin, passed)."""
        if rows:
            self._conn.executemany(
                "INSERT INTO stdf_parts(doc_id, part_id, insertion, lot, sublot, wafer, x, y, "
                "head, site, hard_bin, soft_bin, passed) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                rows,
            )
            self._maybe_commit()

    def stdf_test_list(self) -> list[sqlite3.Row]:
        """Distinct (test_num, test_txt, n) across the store — a web UI's test picker."""
        return self._conn.execute(
            "SELECT test_num, MAX(test_txt) AS test_txt, COUNT(*) AS n FROM stdf_results "
            "GROUP BY test_num ORDER BY test_num"
        ).fetchall()

    def stdf_results_query(
        self, *, test_num: int | None = None, insertion: str | None = None,
        doc_id: int | None = None, limit: int = 100000,
    ) -> list[sqlite3.Row]:
        """Numeric results with optional filters — the data behind a plot, queryable without AI."""
        clauses, params = [], []
        for col, val in (("r.test_num", test_num), ("r.insertion", insertion), ("r.doc_id", doc_id)):
            if val is not None:
                clauses.append(f"{col}=?")
                params.append(val)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)
        return self._conn.execute(
            "SELECT r.test_num, r.test_txt, r.result, r.units, r.head, r.site, r.part_id, "
            "r.insertion, r.passed FROM stdf_results r" + where + " ORDER BY r.id LIMIT ?",
            params,
        ).fetchall()

    def stdf_parts_all(self, *, doc_id: int | None = None) -> list[sqlite3.Row]:
        """All part touchdowns (optionally one doc) for yield/traceability, ordered by ingest order."""
        if doc_id is None:
            return self._conn.execute("SELECT * FROM stdf_parts ORDER BY id").fetchall()
        return self._conn.execute(
            "SELECT * FROM stdf_parts WHERE doc_id=? ORDER BY id", (doc_id,)
        ).fetchall()

    def count_stdf_results(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM stdf_results").fetchone()[0])

    # ---- structured source-code symbols (Phase 9): queryable by non-AI tools + the style deriver ---

    def add_code_symbols(self, rows: Sequence[tuple]) -> None:  # type: ignore[type-arg]
        """Bulk-insert code symbols (doc_id, chunk_id, language, kind, name, qualname, signature,
        docstring, start_line, end_line, parent, path)."""
        if rows:
            self._conn.executemany(
                "INSERT INTO code_symbols(doc_id, chunk_id, language, kind, name, qualname, "
                "signature, docstring, start_line, end_line, parent, path) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                rows,
            )
            self._maybe_commit()

    def code_symbols_query(
        self, *, language: str | None = None, kind: str | None = None, name_like: str | None = None,
        doc_id: int | None = None, limit: int = 100000,
    ) -> list[sqlite3.Row]:
        """Symbols with optional filters (language / kind / name glob / doc), in source order —
        the data behind ``code ls`` and the style-guide deriver, queryable without AI."""
        clauses, params = [], []
        for col, val in (("language", language), ("kind", kind), ("doc_id", doc_id)):
            if val is not None:
                clauses.append(f"{col}=?")
                params.append(val)
        if name_like:
            clauses.append("(name LIKE ? OR qualname LIKE ?)")
            params += [name_like, name_like]
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)
        return self._conn.execute(
            "SELECT doc_id, chunk_id, language, kind, name, qualname, signature, docstring, "
            "start_line, end_line, parent, path FROM code_symbols" + where
            + " ORDER BY doc_id, start_line LIMIT ?",
            params,
        ).fetchall()

    def count_code_symbols(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM code_symbols").fetchone()[0])

    # ---- generic columnar data (Phase 10): any CSV/table, queryable + plottable by non-AI tools ---

    def add_data_column(
        self, *, doc_id: int, dataset: str, name: str, kind: str, units: str,
        lo: float | None, hi: float | None, values: Sequence[float], groups: Sequence[str] = (),
    ) -> int:
        """Persist one column (metadata + its values) and return the new ``data_columns`` id.
        ``values``/``groups`` are stored row-per-value so plain SQL (or a thin web UI) can query them."""
        cur = self._conn.execute(
            "INSERT INTO data_columns(doc_id, dataset, name, kind, units, lo, hi, n) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (doc_id, dataset, name, kind, units, lo, hi, len(values)),
        )
        col_id = int(cur.lastrowid or 0)
        if values:
            grp = list(groups) if groups else [None] * len(values)
            self._conn.executemany(
                "INSERT INTO data_values(col_id, row_idx, value, grp) VALUES (?,?,?,?)",
                [(col_id, i, float(v), grp[i]) for i, v in enumerate(values)],
            )
        self._maybe_commit()
        return col_id

    def data_columns(self, *, doc_id: int | None = None) -> list[sqlite3.Row]:
        """Column catalog (id, doc_id, dataset, name, kind, units, lo, hi, n) — a data store's
        columns a non-AI tool / web UI lists to pick something to query or plot."""
        where = " WHERE doc_id = ?" if doc_id is not None else ""
        params = (doc_id,) if doc_id is not None else ()
        return self._conn.execute(
            "SELECT id, doc_id, dataset, name, kind, units, lo, hi, n FROM data_columns"
            + where + " ORDER BY id", params,
        ).fetchall()

    def data_values(self, *, column_id: int) -> list[sqlite3.Row]:
        """One column's ``(row_idx, value, grp)`` in order — the data behind a plot/query, no AI."""
        return self._conn.execute(
            "SELECT row_idx, value, grp FROM data_values WHERE col_id = ? ORDER BY row_idx",
            (column_id,),
        ).fetchall()

    def count_data_values(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM data_values").fetchone()[0])

    # ---- feedback (Phase 8): per-user + global, ranking-eligible -----------------------------

    def add_feedback(
        self, *, author: str, scope: str, text: str, doc_id: int | None = None,
        chunk_id: int | None = None, rating: int | None = None,
    ) -> int:
        """Record one piece of feedback and return its id. ``scope`` is ``user`` (private to
        ``author``) or ``global`` (visible to everyone). ``rating`` is clamped to the documented
        -1/0/+1 range so a single call can't inject an unbounded score (red-team #H2)."""
        if rating is not None:
            try:
                rating = max(-1, min(1, int(rating)))
            except (TypeError, ValueError):
                rating = None
        cur = self._conn.execute(
            "INSERT INTO feedback(ts, author, scope, doc_id, chunk_id, rating, text) "
            "VALUES (?,?,?,?,?,?,?)",
            (_utcnow_iso(), author, scope, doc_id, chunk_id, rating, text),
        )
        self._maybe_commit()
        return int(cur.lastrowid or 0)

    def feedback_entries(
        self, *, author: str | None = None, include_global: bool = True
    ) -> list[sqlite3.Row]:
        """Feedback **visible to ``author``**: their own private entries plus every global entry.
        With ``author=None`` and ``include_global`` False → nothing; None + global → global only."""
        clauses, params = [], []
        if author is not None:
            clauses.append("(author = ? AND scope = 'user')")
            params.append(author)
        if include_global:
            clauses.append("scope = 'global'")
        if not clauses:
            return []
        return self._conn.execute(
            "SELECT id, ts, author, scope, doc_id, chunk_id, rating, text FROM feedback "
            "WHERE " + " OR ".join(clauses) + " ORDER BY id",
            params,
        ).fetchall()

    def feedback_scores(self, *, author: str | None = None) -> dict[int, int]:
        """Net rating per **document** visible to ``author`` (own private + global) — the signal that
        nudges search ranking. Each **account** counts at most **once** per document (its latest
        rating), so no single account can spam a doc up or down (red-team #H2); a malformed rating row
        is skipped, never crashing ranking (red-team #L2)."""
        latest: dict[tuple[str, int], int] = {}  # (author, doc_id) -> that account's latest rating
        for r in self.feedback_entries(author=author, include_global=True):  # ordered by id asc
            did, rating, who = r["doc_id"], r["rating"], r["author"]
            if did is None or rating is None:
                continue
            try:
                rv = max(-1, min(1, int(rating)))
            except (TypeError, ValueError):
                continue
            latest[(str(who), int(did))] = rv
        scores: dict[int, int] = {}
        for (_who, did), rv in latest.items():
            scores[did] = scores.get(did, 0) + rv
        return scores

    def count_feedback(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0])

    def document_source_map(self, doc_ids: Sequence[int]) -> dict[int, str]:
        """``{doc_id: source label}`` for the given ids — used to resolve each hit's authority tier
        (the config maps a source name → tier) during feedback-aware re-ranking."""
        ids = list(dict.fromkeys(doc_ids))
        if not ids:
            return {}
        placeholders = ",".join("?" * len(ids))
        return {
            int(r["id"]): str(r["source"] or "")
            for r in self._conn.execute(
                f"SELECT id, source FROM documents WHERE id IN ({placeholders})", ids  # noqa: S608
            )
        }

    def stdf_documents(self) -> list[sqlite3.Row]:
        """Every STDF document with its SKU (the ``source`` label = part SKU/name it was filed under),
        lot, distinct insertions, and result/part counts — the file catalog behind ``stdf ls``.
        Glob / SKU filtering happens above this in the service layer."""
        return self._conn.execute(
            "SELECT d.id AS doc_id, d.path, d.title, d.source AS sku, d.status, "
            "(SELECT COUNT(*) FROM stdf_results r WHERE r.doc_id = d.id) AS n_results, "
            "(SELECT COUNT(*) FROM stdf_parts p WHERE p.doc_id = d.id) AS n_parts, "
            "(SELECT p.lot FROM stdf_parts p WHERE p.doc_id = d.id AND p.lot IS NOT NULL LIMIT 1) "
            "  AS lot, "
            "(SELECT GROUP_CONCAT(x.insertion) FROM "
            "  (SELECT DISTINCT p.insertion FROM stdf_parts p WHERE p.doc_id = d.id) x) AS insertions "
            "FROM documents d WHERE d.fmt = 'stdf' ORDER BY d.id"
        ).fetchall()

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

    def chunk_dedup_keys(self, chunk_ids: Sequence[int]) -> dict[int, tuple[str, int]]:
        """Map each chunk id to its cross-store identity ``(document content_hash, ord)`` — the key
        a federated search dedupes/fuses on, so the same logical chunk appearing in more than one
        store collapses to a single result (R-TEST-3). ``content_hash`` is empty only for a document
        ingested without one; the caller falls back to a store-local key in that case so distinct
        content never false-merges."""
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" * len(chunk_ids))
        rows = self._conn.execute(
            f"SELECT c.id, d.content_hash, c.ord FROM chunks c "
            f"JOIN documents d ON c.document_id = d.id WHERE c.id IN ({placeholders})",
            tuple(chunk_ids),
        ).fetchall()
        return {int(r[0]): (str(r[1] or ""), int(r[2])) for r in rows}

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
            # code_symbols reference chunks(id), so clear them before the chunks they point at
            c.execute("DELETE FROM code_symbols WHERE doc_id=?", (doc_id,))
            c.execute("DELETE FROM chunks WHERE document_id=?", (doc_id,))
            c.execute("DELETE FROM relations WHERE src_doc=?", (doc_id,))
            c.execute("UPDATE relations SET dst_doc=NULL WHERE dst_doc=?", (doc_id,))
            c.execute("DELETE FROM images WHERE doc_id=?", (doc_id,))
            c.execute("DELETE FROM stdf_results WHERE doc_id=?", (doc_id,))
            c.execute("DELETE FROM stdf_parts WHERE doc_id=?", (doc_id,))
            c.execute(
                "DELETE FROM data_values WHERE col_id IN "
                "(SELECT id FROM data_columns WHERE doc_id=?)", (doc_id,))
            c.execute("DELETE FROM data_columns WHERE doc_id=?", (doc_id,))
            c.execute("DELETE FROM feedback WHERE doc_id=?", (doc_id,))
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

    def images_for_chunk(self, doc_id: int, chunk_id: int) -> list[sqlite3.Row]:
        """Retained images associated with a cited chunk — for embedding a diagram directly in a
        report. An ``enrichment``/``image_ref`` chunk shares its ``locator`` with its image (both
        from the image's heading path), so map through ``(doc_id, locator)``. Returns
        ``(sha256, ext, alt, caption)`` rows; empty for non-image chunks."""
        row = self._conn.execute(
            "SELECT kind, locator FROM chunks WHERE document_id=? AND id=?", (doc_id, chunk_id)
        ).fetchone()
        if row is None or row["kind"] not in ("enrichment", "image_ref"):
            return []
        return self._conn.execute(
            "SELECT sha256, ext, alt, caption FROM images WHERE doc_id=? AND locator=?",
            (doc_id, row["locator"]),
        ).fetchall()

    def images_for_chunks(self, chunk_ids: Sequence[int]) -> dict[int, list[sqlite3.Row]]:
        """``images_for_chunk`` for a whole ranked result set in one query — search calls this on
        every hit, so it must not be N+1. Keyed by chunk id; chunks with no image are absent."""
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" * len(chunk_ids))
        rows = self._conn.execute(
            "SELECT c.id AS chunk_id, i.sha256, i.ext, i.alt, i.caption "
            "FROM chunks c JOIN images i ON i.doc_id = c.document_id AND i.locator = c.locator "
            f"WHERE c.id IN ({placeholders}) AND c.kind IN ('enrichment', 'image_ref') "
            "ORDER BY c.id, i.sha256",  # deterministic ordering (R-SRCH-5)
            tuple(chunk_ids),
        ).fetchall()
        out: dict[int, list[sqlite3.Row]] = {}
        for row in rows:
            out.setdefault(int(row["chunk_id"]), []).append(row)
        return out

    def count_images(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM images").fetchone()[0])

    # -- vision enrichment (enrich.vision_images) --------------------------------

    def images_needing_vision(
        self, *, limit: int | None = None, by_size: bool = False
    ) -> list[sqlite3.Row]:
        """Retained images with no vision result yet. Ordered by sha for determinism, or by
        **descending size** when ``by_size`` (largest first) so a capped run enriches the real
        diagrams before tiny icons — with ``sha256`` as the deterministic tie-break.

        ``vision_model IS NULL`` is the worklist marker, so a re-run only processes images
        that were never enriched (idempotent, and no double API spend)."""
        order = "bytes DESC, sha256" if by_size else "sha256"
        sql = (
            "SELECT sha256, ext, doc_id, locator, alt, caption FROM images "
            f"WHERE vision_model IS NULL ORDER BY {order}"
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

    def documents_needing_summary(self, limit: int = 0) -> list[tuple[int, str]]:
        """``(doc_id, title)`` for ACTIVE documents that have no AI summary yet — the summary
        worklist (§17). A summary is an ``enrichment`` chunk with locator ``summary``; its absence
        is the marker, so a re-run only summarizes new docs (idempotent, no double spend). Ordered
        by doc id for determinism."""
        sql = (
            "SELECT d.id, d.title FROM documents d WHERE d.status='active' AND NOT EXISTS ("
            "  SELECT 1 FROM chunks c WHERE c.document_id=d.id AND c.kind='enrichment' "
            "  AND c.locator='summary'"
            ") ORDER BY d.id"
        )
        if limit > 0:
            rows = self._conn.execute(sql + " LIMIT ?", (limit,)).fetchall()
        else:
            rows = self._conn.execute(sql).fetchall()
        return [(int(r[0]), str(r[1])) for r in rows]

    def document_ingest_text(self, doc_id: int) -> str:
        """A document's ingest-time text (body/code/table chunks, not prior enrichment), joined —
        the input to summarization. Ordered by ord so the text reads in document order."""
        rows = self._conn.execute(
            "SELECT text FROM chunks WHERE document_id=? AND kind NOT IN "
            "('enrichment', 'image_ref') ORDER BY ord",
            (doc_id,),
        ).fetchall()
        return "\n".join(str(r[0]) for r in rows)

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

    def related_documents(
        self, doc_id: int, direction: str = "both", depth: int = 1
    ) -> list[dict[str, object]]:
        """Documents reachable from ``doc_id`` over the resolved relations graph (R-ING-5, §17
        N-hop), walked up to ``depth`` hops via a recursive CTE. direction: ``out`` (docs this one
        links to), ``in`` (docs that link to this one), ``both``. Returns one row per reachable
        doc — its SHORTEST hop count, path, title, link_type, and a ``weight`` (how many times the
        source directly references it — the link-strength signal). Ranked by **(hops, −weight,
        doc_id)**: the most-referenced direct neighbours first, so on a densely-linked corpus the
        genuinely-related pages beat one-off nav links. Self excluded; the depth cap bounds the
        walk so cycles can't spin."""
        depth = max(1, depth)
        wanted = ("out", "in") if direction == "both" else (direction,)
        reach: dict[int, tuple[int, str]] = {}  # doc -> (shortest hops, direction)
        for d in wanted:
            nxt = "r.dst_doc" if d == "out" else "r.src_doc"
            edge = "r.src_doc = w.doc" if d == "out" else "r.dst_doc = w.doc"
            rows = self._conn.execute(
                "WITH RECURSIVE walk(doc, hops) AS ("
                "  SELECT ?, 0 "
                "  UNION "
                f"  SELECT {nxt}, w.hops + 1 FROM walk w JOIN relations r ON {edge} "
                f"  WHERE {nxt} IS NOT NULL AND w.hops < ?"
                ") SELECT doc, MIN(hops) FROM walk WHERE doc != ? GROUP BY doc",
                (doc_id, depth, doc_id),
            ).fetchall()
            for r in rows:
                doc, hops = int(r[0]), int(r[1])
                if doc in reach:
                    prev_hops, prev_dir = reach[doc]
                    reach[doc] = (min(prev_hops, hops), "both" if prev_dir != d else d)
                else:
                    reach[doc] = (hops, d)
        if not reach:
            return []
        # Direct-edge occurrence counts (link strength) + link_type, per neighbour+direction. Only
        # meaningful for 1-hop neighbours — multi-hop weight/link_type is ambiguous.
        weight: dict[tuple[int, str], int] = {}
        link_type: dict[tuple[int, str], str] = {}
        for row in self.relations_out(doc_id):
            if row["dst_doc"] is not None:
                key = (int(row["dst_doc"]), "out")
                weight[key] = weight.get(key, 0) + 1
                link_type.setdefault(key, str(row["link_type"] or ""))
        for row in self.relations_in(doc_id):
            if row["src_doc"] is not None:
                key = (int(row["src_doc"]), "in")
                weight[key] = weight.get(key, 0) + 1
                link_type.setdefault(key, str(row["link_type"] or ""))
        meta = {
            int(m["id"]): m
            for m in self._conn.execute(
                "SELECT id, path, title FROM documents WHERE id IN "
                f"({','.join('?' * len(reach))})",
                tuple(reach),
            ).fetchall()
        }

        def _weight(doc: int) -> int:
            hops, dirn = reach[doc]
            return weight.get((doc, dirn), 0) if hops == 1 else 0

        out: list[dict[str, object]] = []
        for doc in sorted(reach, key=lambda d: (reach[d][0], -_weight(d), d)):
            hops, dirn = reach[doc]
            m = meta.get(doc)
            out.append(
                {
                    "doc_id": doc,
                    "path": str(m["path"]) if m else "",
                    "title": str(m["title"]) if m else "",
                    "hops": hops,
                    "direction": dirn,
                    "link_type": link_type.get((doc, dirn), "") if hops == 1 else "",
                    "weight": _weight(doc),
                }
            )
        return out

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
