"""Catalog facade behaviors: selective source purge (remove_source)."""

from __future__ import annotations

from pathlib import Path

from docusearch import config as cfg
from docusearch.catalog import Catalog
from docusearch.store import Store


def _two_source_config(tmp: Path, keep: Path, drop: Path) -> cfg.Config:
    path = tmp / "docusearch.yaml"
    path.write_text(
        f'paths:\n  staging_dir: "{(tmp / "s").as_posix()}"\n'
        f'  db_path: "{(tmp / "c.db").as_posix()}"\n'
        f'  tmp_dir: "{(tmp / "t").as_posix()}"\n'
        "sources:\n"
        f'  - name: keep_me\n    location: "{keep.as_posix()}"\n    min_content_chars: 5\n'
        f'  - name: delete_me_next\n    location: "{drop.as_posix()}"\n    min_content_chars: 5\n'
        'embed:\n  model: "none"\n',
        encoding="utf-8",
    )
    return cfg.load(path)


def _docs(root: Path, n: int, tag: str) -> None:
    root.mkdir()
    for i in range(n):
        (root / f"{tag}{i}.html").write_text(
            f"<body><h1>{tag} {i}</h1><p>content about {tag} timing interfaces {i}</p></body>",
            encoding="utf-8",
        )


def test_remove_source_purges_only_that_label(tmp_path: Path) -> None:
    keep, drop = tmp_path / "keep", tmp_path / "drop"
    _docs(keep, 3, "keep")
    _docs(drop, 4, "del")
    config = _two_source_config(tmp_path, keep, drop)
    cat = Catalog(config)
    cat.ingest()
    with Store.open(config.paths.db_path) as store:
        assert store.count_documents() == 7

    removed = cat.remove_source("delete_me_next")

    assert removed == 4
    with Store.open(config.paths.db_path) as store:
        assert store.count_documents() == 3
        assert store.document_ids_for_source("delete_me_next") == []
        assert len(store.document_ids_for_source("keep_me")) == 3


def test_prune_missing_removes_docs_whose_files_are_gone(tmp_path: Path) -> None:
    keep, drop = tmp_path / "keep", tmp_path / "drop"
    _docs(keep, 2, "keep")
    _docs(drop, 2, "del")
    config = _two_source_config(tmp_path, keep, drop)
    cat = Catalog(config)
    cat.ingest()
    with Store.open(config.paths.db_path) as store:
        assert store.count_documents() == 4

    # simulate a moved/renamed folder: the 'drop' source files disappear from disk
    for f in drop.glob("*.html"):
        f.unlink()

    assert cat.prune_missing(apply=False) == 2  # dry run counts, removes nothing
    with Store.open(config.paths.db_path) as store:
        assert store.count_documents() == 4
    assert cat.prune_missing(apply=True) == 2  # now actually removes the orphans
    with Store.open(config.paths.db_path) as store:
        assert store.count_documents() == 2  # only the present-on-disk docs remain


def test_remove_unknown_source_is_noop(tmp_path: Path) -> None:
    keep, drop = tmp_path / "keep", tmp_path / "drop"
    _docs(keep, 2, "keep")
    _docs(drop, 1, "del")
    config = _two_source_config(tmp_path, keep, drop)
    cat = Catalog(config)
    cat.ingest()
    assert cat.remove_source("does-not-exist") == 0
    with Store.open(config.paths.db_path) as store:
        assert store.count_documents() == 3  # nothing removed
