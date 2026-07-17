"""Citations & link resolution (R-CIT-1, §11).

Every factual claim in agent output or a report ends with a tag: ``[GK]`` (general
knowledge — not found in the catalog) or ``[D:<doc_id>#<chunk_id>]`` (catalog-sourced).
This module parses those tags, resolves catalog citations to a working server URL, and
**verifies** that a text never cites a chunk outside its evidence set — the report
builder uses that to refuse hallucinated references.

Public surface:
    Citation                                    -- one parsed tag
    CitationError                               -- raised when a text cites outside evidence
    parse(text) -> list[Citation]
    resolve(citation, base_url) -> str | None    -- None for [GK]
    verify(text, allowed_chunk_ids) -> list[Citation]   -- offending citations (empty = ok)
    render_references(text, base_url) -> (body, refs)    -- numbered refs for reports
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# [GK] or [D:<digits>#<digits>]
_CITATION = re.compile(r"\[GK\]|\[D:(\d+)#(\d+)\]")


class CitationError(Exception):
    """A text cites a chunk that is not in the allowed evidence set (R-CIT-1)."""


@dataclass(frozen=True)
class Citation:
    """A parsed citation tag."""

    kind: str  # "GK" | "doc"
    raw: str  # the exact matched text, e.g. "[D:812#90312]"
    doc_id: int | None = None
    chunk_id: int | None = None


def parse(text: str) -> list[Citation]:
    """Return every citation tag in ``text``, in order of appearance."""
    out: list[Citation] = []
    for match in _CITATION.finditer(text):
        if match.group(0) == "[GK]":
            out.append(Citation(kind="GK", raw="[GK]"))
        else:
            out.append(
                Citation(
                    kind="doc",
                    raw=match.group(0),
                    doc_id=int(match.group(1)),
                    chunk_id=int(match.group(2)),
                )
            )
    return out


def resolve(citation: Citation, base_url: str) -> str | None:
    """The clickable server URL for a catalog citation, or None for ``[GK]`` (R-API-2)."""
    if citation.kind != "doc":
        return None
    base = base_url.rstrip("/")
    return f"{base}/v1/documents/{citation.doc_id}?chunk={citation.chunk_id}"


def verify(text: str, allowed: set[tuple[int, int]]) -> list[Citation]:
    """Return the catalog citations that reference a ``(doc_id, chunk_id)`` pair outside
    the evidence set.

    Checking the *pair* (not just the chunk) matters: a chunk id is globally unique, so a
    citation like ``[D:1#2]`` where chunk 2 actually belongs to a different document is a
    mis-attribution — it would resolve to the wrong source URL. Empty means every catalog
    citation is backed by an evidence hit. ``[GK]`` is never a violation. The report
    builder refuses to render when this is non-empty (R-CIT-1).
    """
    return [
        c
        for c in parse(text)
        if c.kind == "doc"
        and c.doc_id is not None
        and c.chunk_id is not None
        and (c.doc_id, c.chunk_id) not in allowed
    ]


def render_references(text: str, base_url: str) -> tuple[str, list[str]]:
    """Turn ``[D:...]`` tags into numbered markers ``[n]`` and build a references list.

    Distinct catalog citations are numbered once in order of first appearance and
    de-duplicated; ``[GK]`` is left inline. Returns ``(body, references)`` where each
    reference is ``"<n>. <url>"`` (R-CIT-2, §11).
    """
    numbering: dict[tuple[int, int], int] = {}
    ordered: list[Citation] = []
    for c in parse(text):
        if c.kind == "doc" and c.doc_id is not None and c.chunk_id is not None:
            key = (c.doc_id, c.chunk_id)
            if key not in numbering:
                numbering[key] = len(ordered) + 1
                ordered.append(c)

    body = text
    for (doc_id, chunk_id), number in numbering.items():
        body = body.replace(f"[D:{doc_id}#{chunk_id}]", f"[{number}]")

    references = [f"{i + 1}. {resolve(c, base_url)}" for i, c in enumerate(ordered)]
    return body, references
