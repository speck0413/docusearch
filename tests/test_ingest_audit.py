"""Audit-edge behaviors of the ingest pipeline (§7.8) — the loud reporting Gate 1 needs."""

from __future__ import annotations

from pathlib import Path

from docusearch import config as cfg
from docusearch import ingest
from docusearch.store import Store


def _load_config(tmp: Path, source_yaml: str) -> cfg.Config:
    text = f"""
paths:
  staging_dir: "{(tmp / "staging").as_posix()}"
  db_path: "{(tmp / "catalog.db").as_posix()}"
  tmp_dir: "{(tmp / "tmp").as_posix()}"
{source_yaml}
embed:
  model: "none"
"""
    path = tmp / "docusearch.yaml"
    path.write_text(text, encoding="utf-8")
    return cfg.load(path)


def test_missing_source_location_is_reported_not_fatal(tmp_path: Path) -> None:
    config = _load_config(
        tmp_path,
        f'sources:\n  - name: gone\n    location: "{(tmp_path / "nope").as_posix()}"\n',
    )
    with Store.open(config.paths.db_path) as store:
        result = ingest.run_ingest(config, store)
        assert result.documents == 0
        assert any("not found" in msg for _, msg in result.errors)


def test_content_selector_miss_counted_but_doc_kept(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    (root / "a.html").write_text(
        "<body><h1>Title</h1><p>Plenty of real content here to keep it.</p></body>",
        encoding="utf-8",
    )
    config = _load_config(
        tmp_path,
        f'sources:\n  - name: d\n    location: "{root.as_posix()}"\n'
        '    content_selector: "main.article"\n    min_content_chars: 5\n',
    )
    with Store.open(config.paths.db_path) as store:
        result = ingest.run_ingest(config, store)
        assert result.content_selector_misses == 1
        assert result.documents == 1  # fell back to <body>, still ingested
        assert store.chunk_ids_matching("content")


def test_external_link_stays_unresolved(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    (root / "a.html").write_text(
        '<body><h1>H</h1><p>See <a href="https://example.com/x">external</a> reference here.</p></body>',
        encoding="utf-8",
    )
    config = _load_config(
        tmp_path,
        f'sources:\n  - name: d\n    location: "{root.as_posix()}"\n    min_content_chars: 5\n',
    )
    with Store.open(config.paths.db_path) as store:
        result = ingest.run_ingest(config, store)
        assert result.relations_total == 1
        assert result.relations_unresolved == 1


def test_other_files_counted(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    (root / "a.html").write_text(
        "<body><p>real content that is long enough here</p></body>", "utf-8"
    )
    (root / "readme.txt").write_text("not html", encoding="utf-8")
    config = _load_config(
        tmp_path,
        f'sources:\n  - name: d\n    location: "{root.as_posix()}"\n'
        '    include: ["**/*.html"]\n    min_content_chars: 5\n',
    )
    with Store.open(config.paths.db_path) as store:
        result = ingest.run_ingest(config, store)
        assert result.other_files == 1  # readme.txt present but not selected


def test_untagged_audience_counted(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    (root / "a.html").write_text(
        "<body><p>real content that is long enough here</p></body>", "utf-8"
    )
    config = _load_config(
        tmp_path,
        f'sources:\n  - name: d\n    location: "{root.as_posix()}"\n'
        "    audience: []\n    min_content_chars: 5\n",
    )
    with Store.open(config.paths.db_path) as store:
        result = ingest.run_ingest(config, store)
        assert result.untagged_audience_docs == 1
