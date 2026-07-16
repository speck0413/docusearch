"""Embeddings at index time + provenance (R-EMB-2/3/6, R-CFG-4).

Uses a deterministic fake provider (no torch) so these stay fast and offline; the real
LocalProvider is covered in test_embed.py.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from docusearch import config as cfg
from docusearch import embed, ingest
from docusearch.store import Store


class FakeProvider:
    """Deterministic hash-based vectors — same text always maps to the same vector."""

    def __init__(self, model_id: str = "fake-v1", dim: int = 8) -> None:
        self._id = model_id
        self._dim = dim

    @property
    def model_id(self) -> str:
        return self._id

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vec = np.frombuffer(digest[: self._dim], dtype=np.uint8).astype(np.float32)
            out[i] = vec / (np.linalg.norm(vec) or 1.0)
        return out


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
    root = tmp_path / "docs"
    _corpus(root)
    config = _config(tmp_path, root)
    with Store.open(config.paths.db_path) as store:
        ingest.run_ingest(config, store, provider=FakeProvider("model-a"))
        with pytest.raises(embed.EmbedError, match="[Rr]e-index"):
            ingest.run_ingest(config, store, provider=FakeProvider("model-b"), force=True)


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
