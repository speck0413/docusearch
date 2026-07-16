"""Corpora tooling: registry integrity + pure helpers (§15.1).

The downloader is validation tooling (not shipped), loaded here by path. We don't hit
the network — we check the registry is well-formed and the pure helpers behave.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _load_download():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(
        "corpora_download", _ROOT / "corpora" / "download.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclass machinery looks the module up here
    spec.loader.exec_module(mod)
    return mod


dl = _load_download()


def test_registry_is_well_formed() -> None:
    assert dl.CORPORA, "expected at least one corpus"
    names = [c.name for c in dl.CORPORA]
    assert len(names) == len(set(names)), "corpus names must be unique"
    for c in dl.CORPORA:
        assert c.license, f"{c.name} missing license"
        assert c.note, f"{c.name} missing note"
        assert c.kind in {"tarball", "zip", "rustup", "git", "manual"}
        # Official channels only: http(s) sources must be TLS.
        if c.url:
            assert c.url.startswith("https://"), f"{c.name} url must be https"


def test_primary_corpus_is_php() -> None:
    # §15.1 names the PHP manual the primary vendor analog.
    assert dl.CORPORA[0].name == "php"


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    p = tmp_path / "blob.bin"
    p.write_bytes(b"docusearch corpora")
    assert dl._sha256_file(p) == hashlib.sha256(b"docusearch corpora").hexdigest()


def test_count_files_respects_glob(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "one.html").write_text("x", encoding="utf-8")
    (tmp_path / "a" / "two.html").write_text("x", encoding="utf-8")
    (tmp_path / "a" / "note.md").write_text("x", encoding="utf-8")
    assert dl._count_files(tmp_path, "**/*.html") == 2
    assert dl._count_files(tmp_path, "**/*.md") == 1


def test_render_manifest_lists_every_corpus() -> None:
    text = dl.render_manifest(dl._skeleton_entries())
    assert "MANIFEST" in text
    for c in dl.CORPORA:
        assert c.name in text
        assert c.license in text


def test_manifest_file_exists_and_lists_sources() -> None:
    manifest = _ROOT / "corpora" / "MANIFEST.md"
    assert manifest.exists(), "run: python corpora/download.py --write-skeleton"
    body = manifest.read_text(encoding="utf-8")
    for c in dl.CORPORA:
        assert c.name in body


def test_extract_rejects_unknown_archive(tmp_path: Path) -> None:
    bad = tmp_path / "corpus.rar"
    bad.write_bytes(b"nope")
    with pytest.raises(ValueError, match="unknown archive"):
        dl._extract(bad, tmp_path / "out")
