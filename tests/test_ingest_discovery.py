"""Source discovery (globs) + content hashing (R-ING-1, R-ING-3)."""

from __future__ import annotations

import hashlib
from pathlib import Path

from docusearch import ingest


def _make_tree(root: Path) -> None:
    (root / "a.html").write_text("<h1>A</h1>", encoding="utf-8")
    (root / "b.html").write_text("<h1>B</h1>", encoding="utf-8")
    (root / "notes.txt").write_text("plain", encoding="utf-8")
    (root / "nav").mkdir()
    (root / "nav" / "menu.html").write_text("<nav>x</nav>", encoding="utf-8")
    (root / "guide").mkdir()
    (root / "guide" / "intro.html").write_text("<h1>Intro</h1>", encoding="utf-8")


def _rels(root: Path, paths: list[Path]) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in paths)


def test_include_only_html(tmp_path: Path) -> None:
    _make_tree(tmp_path)
    found = list(ingest.iter_files(tmp_path, include=["**/*.html"], exclude=[]))
    assert _rels(tmp_path, found) == ["a.html", "b.html", "guide/intro.html", "nav/menu.html"]


def test_exclude_glob_removes_nav(tmp_path: Path) -> None:
    _make_tree(tmp_path)
    found = list(ingest.iter_files(tmp_path, include=["**/*.html"], exclude=["**/nav/**"]))
    assert _rels(tmp_path, found) == ["a.html", "b.html", "guide/intro.html"]


def test_exclude_does_not_overmatch_similar_names(tmp_path: Path) -> None:
    # "**/nav/**" must not exclude a "navigation.html" file (segment-aware matching).
    (tmp_path / "navigation.html").write_text("<h1>n</h1>", encoding="utf-8")
    found = list(ingest.iter_files(tmp_path, include=["**/*.html"], exclude=["**/nav/**"]))
    assert _rels(tmp_path, found) == ["navigation.html"]


def test_results_are_sorted_deterministically(tmp_path: Path) -> None:
    for name in ("zeta.html", "alpha.html", "mid.html"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    found = list(ingest.iter_files(tmp_path, include=["**/*.html"], exclude=[]))
    assert found == sorted(found)


def test_empty_include_matches_everything(tmp_path: Path) -> None:
    _make_tree(tmp_path)
    found = list(ingest.iter_files(tmp_path, include=[], exclude=[]))
    # every file, including the .txt
    assert "notes.txt" in _rels(tmp_path, found)


def test_content_hash_matches_hashlib(tmp_path: Path) -> None:
    p = tmp_path / "f.html"
    p.write_bytes(b"<h1>Hello</h1>")
    assert ingest.content_hash(p) == hashlib.sha256(b"<h1>Hello</h1>").hexdigest()


def test_content_hash_changes_with_content(tmp_path: Path) -> None:
    p = tmp_path / "f.html"
    p.write_bytes(b"one")
    h1 = ingest.content_hash(p)
    p.write_bytes(b"two")
    assert ingest.content_hash(p) != h1
