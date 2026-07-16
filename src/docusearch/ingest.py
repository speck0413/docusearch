"""Ingestion pipeline: filesystem -> extract -> chunk -> link/image -> index (§7).

The largest module by design (R-ARCH-3). It turns a source folder of documents into
rows in the store: discover files (globs), skip unchanged ones by content hash
(R-ING-3), strip boilerplate and extract structured text (R-ING-2, §7.3), chunk while
preserving code blocks (R-ING-4), capture links and images (R-ING-5/6), index into FTS5,
and emit a loud audit report (§7.8).

Public surface (grows through Phase 1):
    iter_files(location, include, exclude) -> Iterator[Path]   # source discovery
    content_hash(path) -> str                                  # SHA-256, incremental skip
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator, Sequence
from pathlib import Path

_HASH_BLOCK = 1 << 20


def _glob_to_regex(pattern: str) -> re.Pattern[str]:
    """Translate a path glob to a segment-aware regex.

    ``**`` spans any number of path segments, ``*`` matches within one segment, ``?``
    matches one non-separator character. Matching is done against a POSIX relative path.
    """
    out: list[str] = []
    i, n = 0, len(pattern)
    while i < n:
        if pattern[i : i + 3] == "**/":
            out.append("(?:.*/)?")
            i += 3
        elif pattern[i : i + 2] == "**":
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("^" + "".join(out) + "$")


def iter_files(
    location: Path | str,
    include: Sequence[str],
    exclude: Sequence[str],
) -> Iterator[Path]:
    """Yield files under ``location`` matching any include glob and no exclude glob.

    Results are sorted for deterministic ingest order (needle/audit reproducibility).
    An empty ``include`` list matches everything (R-ING-1).
    """
    root = Path(location)
    inc = [_glob_to_regex(p) for p in include] or [re.compile(".*")]
    exc = [_glob_to_regex(p) for p in exclude]
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if not any(r.match(rel) for r in inc):
            continue
        if any(r.match(rel) for r in exc):
            continue
        yield path


def content_hash(path: Path | str) -> str:
    """SHA-256 of a file's bytes — the incremental-ingest key (R-ING-3)."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(_HASH_BLOCK), b""):
            h.update(block)
    return h.hexdigest()
