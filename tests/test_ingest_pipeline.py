"""End-to-end ingest orchestration (R-ING-1..6, §7): the pipeline into the store."""

from __future__ import annotations

from pathlib import Path

import pytest

from docusearch import config as cfg
from docusearch import ingest
from docusearch.store import Store


def _build_corpus(root: Path) -> None:
    (root / "guide").mkdir(parents=True)
    (root / "nav").mkdir()
    (root / "index.html").write_text(
        "<html><head><title>Home</title></head><body>"
        "<h1>Home</h1>"
        "<p>Welcome to the documentation catalog home page overview.</p>"
        '<p>See the <a href="guide/spi.html">SPI guide</a> for details.</p>'
        '<img src="pic.png" alt="home diagram">'
        "</body></html>",
        encoding="utf-8",
    )
    (root / "guide" / "spi.html").write_text(
        "<html><head><title>SPI</title></head><body>"
        "<h1>SPI</h1><h2>Timing</h2>"
        "<p>The SPI timing nonce ZQX7734 is configured here for the peripheral bus.</p>"
        "<pre><code>spi.configure(mode=0)</code></pre>"
        "<table><tr><th>Reg</th><th>Val</th></tr><tr><td>CTRL</td><td>0x1</td></tr></table>"
        '<p>Back to <a href="../index.html">home</a>.</p>'
        "</body></html>",
        encoding="utf-8",
    )
    (root / "nav" / "menu.html").write_text(
        "<body><nav>menu junk excluded by glob</nav></body>", encoding="utf-8"
    )
    (root / "tiny.html").write_text("<body><p>hi</p></body>", encoding="utf-8")
    (root / "notes.txt").write_text("not html, not included", encoding="utf-8")
    (root / "pic.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"fake-image-bytes")


def _config(root: Path, tmp: Path, *, min_chars: int = 25) -> cfg.Config:
    text = f"""
paths:
  staging_dir: "{(tmp / "staging").as_posix()}"
  db_path: "{(tmp / "catalog.db").as_posix()}"
  tmp_dir: "{(tmp / "tmp").as_posix()}"
sources:
  - name: docs
    location: "{root.as_posix()}"
    include: ["**/*.html"]
    exclude: ["**/nav/**"]
    content_selector: ""
    min_content_chars: {min_chars}
    audience: ["engineering"]
embed:
  model: "none"
"""
    path = tmp / "docusearch.yaml"
    path.write_text(text, encoding="utf-8")
    return cfg.load(path)


@pytest.fixture
def corpus(tmp_path: Path) -> tuple[Path, cfg.Config]:
    root = tmp_path / "corpus"
    root.mkdir()
    _build_corpus(root)
    return root, _config(root, tmp_path)


def _index_dump(store: Store) -> tuple[list, list]:
    docs = store._conn.execute(  # type: ignore[attr-defined]
        "SELECT id, path, title, content_hash, source_version FROM documents ORDER BY id"
    ).fetchall()
    chunks = store._conn.execute(  # type: ignore[attr-defined]
        "SELECT id, document_id, ord, kind, locator, text FROM chunks ORDER BY id"
    ).fetchall()
    return [tuple(r) for r in docs], [tuple(r) for r in chunks]


def test_parallel_parse_is_byte_identical_to_serial(tmp_path: Path) -> None:
    # The parse pool must never change the index: same doc/chunk ids, text, and order as a
    # serial run (R-SRCH-5). Writes are applied in file order regardless of worker count.
    root = tmp_path / "corpus"
    root.mkdir()
    _build_corpus(root)
    for i in range(40):  # enough files that a pool is meaningful
        (root / f"extra{i:02d}.html").write_text(
            f"<body><h1>Doc {i}</h1><p>content about topic {i} timing bus registers here</p>"
            f'<p>link <a href="index.html">home</a></p></body>',
            encoding="utf-8",
        )

    def ingest_with(workers: int, name: str) -> tuple[list, list]:
        run_dir = tmp_path / name
        run_dir.mkdir(parents=True, exist_ok=True)
        config = _config(root, run_dir)
        with Store.open(config.paths.db_path) as store:
            ingest.run_ingest(config, store, workers=workers)
            return _index_dump(store)

    serial = ingest_with(1, "serial")
    parallel = ingest_with(4, "parallel")
    assert parallel == serial


def test_ingest_counts(corpus: tuple[Path, cfg.Config]) -> None:
    root, config = corpus
    with Store.open(config.paths.db_path) as store:
        result = ingest.run_ingest(config, store)
        assert result.documents == 2  # index + guide/spi (tiny stripped, nav excluded)
        assert result.excluded_glob == 1  # nav/menu.html
        assert result.stripped_empty == 1  # tiny.html
        assert store.count_documents() == 2
        assert store.count_chunks() > 0


def test_ingest_needle_and_image_alt_are_searchable(corpus: tuple[Path, cfg.Config]) -> None:
    root, config = corpus
    with Store.open(config.paths.db_path) as store:
        ingest.run_ingest(config, store)
        assert store.chunk_ids_matching("ZQX7734")  # prose needle in a body chunk
        assert store.chunk_ids_matching("diagram")  # image alt text is findable
        assert store.chunk_ids_matching("CTRL")  # table cell linearized into a chunk


def test_ingest_code_block_stored_whole(corpus: tuple[Path, cfg.Config]) -> None:
    root, config = corpus
    with Store.open(config.paths.db_path) as store:
        ingest.run_ingest(config, store)
        rows = store._conn.execute("SELECT text FROM chunks WHERE kind='code'").fetchall()
        assert any("spi.configure(mode=0)" in r[0] for r in rows)


def test_ingest_relations_resolved(corpus: tuple[Path, cfg.Config]) -> None:
    root, config = corpus
    with Store.open(config.paths.db_path) as store:
        result = ingest.run_ingest(config, store)
        assert result.relations_total == 2
        assert result.relations_resolved == 2
        assert result.relations_unresolved == 0


def test_ingest_images_retained(corpus: tuple[Path, cfg.Config]) -> None:
    root, config = corpus
    with Store.open(config.paths.db_path) as store:
        result = ingest.run_ingest(config, store)
        assert result.images == 1
        staged = list((Path(config.paths.staging_dir) / "images").glob("*"))
        assert len(staged) == 1  # pic.png copied, keyed by sha256


def test_html_datauri_image_retained_and_missing_ref_surfaced(tmp_path: Path) -> None:
    # An inline data: image in HTML must be RETAINED (decoded like Markdown does, R-ING-6) — not
    # silently dropped — and a local image reference whose file is missing must be COUNTED
    # (images_unresolved), surfaced in the audit rather than dropped in silence.
    import base64

    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M8AAAMCAT/6JzFAAAAAAElFTkSuQmCC"
    )
    root = tmp_path / "docs"
    root.mkdir()
    b64 = base64.b64encode(png).decode()
    (root / "p.html").write_text(
        f'<body><h1>H</h1><img src="data:image/png;base64,{b64}" alt="inline">'
        '<img src="missing.png" alt="missing"></body>',
        encoding="utf-8",
    )
    path = tmp_path / "d.yaml"
    path.write_text(
        f'paths:\n  staging_dir: "{(tmp_path / "s").as_posix()}"\n'
        f'  db_path: "{(tmp_path / "c.db").as_posix()}"\n  tmp_dir: "{(tmp_path / "t").as_posix()}"\n'
        f'sources:\n  - name: d\n    location: "{root.as_posix()}"\n    min_content_chars: 1\n'
        'embed:\n  model: "none"\n',
        encoding="utf-8",
    )
    config = cfg.load(path)
    with Store.open(config.paths.db_path) as store:
        result = ingest.run_ingest(config, store)
        assert result.images == 1  # the data: image retained (was silently dropped before)
        assert result.images_unresolved == 1  # the missing local ref counted, not silent
        assert store._conn.execute("SELECT COUNT(*) FROM images").fetchone()[0] == 1


def test_ingest_incremental_skip(corpus: tuple[Path, cfg.Config]) -> None:
    root, config = corpus
    with Store.open(config.paths.db_path) as store:
        ingest.run_ingest(config, store)
        second = ingest.run_ingest(config, store)  # nothing changed
        assert second.documents == 0
        assert second.skipped_unchanged == 2


def test_ingest_reingests_changed_file(corpus: tuple[Path, cfg.Config]) -> None:
    root, config = corpus
    with Store.open(config.paths.db_path) as store:
        ingest.run_ingest(config, store)
        (root / "guide" / "spi.html").write_text(
            "<body><h1>SPI</h1><p>Rewritten content with a different nonce QQQ9.</p></body>",
            encoding="utf-8",
        )
        result = ingest.run_ingest(config, store)
        assert result.documents == 1
        assert result.skipped_unchanged == 1
        assert store.count_documents() == 2  # still 2 docs, one replaced
        assert store.chunk_ids_matching("QQQ9")
        assert not store.chunk_ids_matching("ZQX7734")  # old chunks gone


def test_ingest_force_reingests_all(corpus: tuple[Path, cfg.Config]) -> None:
    root, config = corpus
    with Store.open(config.paths.db_path) as store:
        ingest.run_ingest(config, store)
        result = ingest.run_ingest(config, store, force=True)
        assert result.documents == 2
        assert result.skipped_unchanged == 0
