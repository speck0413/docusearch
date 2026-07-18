"""Catalog facade behaviors: selective source purge (remove_source) + vision enrichment."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from docusearch import config as cfg
from docusearch import vision
from docusearch.catalog import Catalog
from docusearch.config import ConfigError
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


# --------------------------------------------------------------- vision enrichment


class _StubVision:
    model_id = "stub-vision-1"

    def describe(self, image_path, *, media_type, alt="", caption="", context=""):  # type: ignore[no-untyped-def]
        return vision.ImageInsight(text=f"OCR {context}", description="a block diagram", model=self.model_id)


class _StubEmbed:
    """A deterministic in-memory embed provider (no torch) for the embed-after-enrich path."""

    model_id = "stub-embed-1"
    dim = 4

    def embed(self, texts):  # type: ignore[no-untyped-def]
        return np.ones((len(texts), self.dim), dtype=np.float32)


def _vision_config(tmp: Path, root: Path, *, on: bool) -> cfg.Config:
    path = tmp / "docusearch.yaml"
    path.write_text(
        f'paths:\n  staging_dir: "{(tmp / "s").as_posix()}"\n'
        f'  db_path: "{(tmp / "c.db").as_posix()}"\n'
        f'  tmp_dir: "{(tmp / "t").as_posix()}"\n'
        "sources:\n"
        f'  - name: docs\n    location: "{root.as_posix()}"\n    min_content_chars: 5\n'
        f'embed:\n  model: "none"\nenrich:\n  vision_images: {"true" if on else "false"}\n',
        encoding="utf-8",
    )
    return cfg.load(path)


def _stage_one_image(config: cfg.Config) -> str:
    data = b"\x89PNG\r\n\x1a\n stub"
    sha = hashlib.sha256(data).hexdigest()
    images = Path(config.paths.staging_dir) / "images"
    images.mkdir(parents=True, exist_ok=True)
    (images / f"{sha}.png").write_bytes(data)
    with Store.open(config.paths.db_path) as store:
        doc_id = next(iter(store.document_path_to_id().values()))
        store.add_image(
            sha256=sha, ext="png", doc_id=doc_id, locator="Fig", alt="", caption="", num_bytes=len(data)
        )
    return sha


def test_enrich_vision_refuses_when_off(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    _docs(root, 1, "d")
    config = _vision_config(tmp_path, root, on=False)
    with pytest.raises(ConfigError):
        Catalog(config).enrich_vision()


def test_enrich_vision_enriches_and_embeds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "docs"
    _docs(root, 1, "d")
    config = _vision_config(tmp_path, root, on=True)
    cat = Catalog(config)
    cat.ingest()  # embed.model none -> 0 vectors so far
    _stage_one_image(config)
    monkeypatch.setattr(vision, "make_vision_provider", lambda e: _StubVision())
    monkeypatch.setattr(Catalog, "_provider", lambda self: _StubEmbed())

    result = cat.enrich_vision()

    assert result.enriched == 1
    with Store.open(config.paths.db_path) as store:
        assert store.chunk_ids_matching("diagram")  # enrichment chunk is BM25-searchable
        assert store.count_embeddings() >= 1  # and was embedded via the ingest path
        assert store.images_needing_vision() == []  # persisted, so re-runs skip it
