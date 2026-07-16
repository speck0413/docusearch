"""Ingestion pipeline: filesystem -> extract -> chunk -> link/image -> index (§7).

The largest module by design (R-ARCH-3). It turns a source folder of documents into
rows in the store: discover files (globs), skip unchanged ones by content hash
(R-ING-3), strip boilerplate and extract structured text (R-ING-2, §7.3), chunk while
preserving code blocks (R-ING-4), capture links and images (R-ING-5/6), index into FTS5,
and emit a loud audit report (§7.8).

Public surface (grows through Phase 1):
    iter_files(location, include, exclude) -> Iterator[Path]   # source discovery
    content_hash(path) -> str                                  # SHA-256, incremental skip
    extract_html(html, *, content_selector, strip_selectors) -> ExtractedDoc  # §7.3
    ExtractedDoc / Segment / LinkRef / ImageRef               # extraction result types
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from selectolax.parser import HTMLParser, Node

_HASH_BLOCK = 1 << 20

_HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_TEXT_BLOCKS = {"p", "li", "dd", "dt", "blockquote", "figcaption", "caption", "address", "summary"}
_SKIP = {"script", "style", "template", "svg", "math", "noscript"}


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


# ----------------------------------------------------------------- HTML extraction


@dataclass
class Segment:
    """One ordered piece of a document's content, tagged with its heading path."""

    kind: str  # body | code | table
    text: str
    heading_path: str


@dataclass
class LinkRef:
    """A cross-reference as written in the source (resolved later, R-ING-5)."""

    target: str  # raw href
    anchor: str
    link_type: str = "html_href"


@dataclass
class ImageRef:
    """An image reference kept for retention + text findability (R-ING-6)."""

    src: str
    alt: str
    caption: str
    heading_path: str


@dataclass
class ExtractedDoc:
    title: str
    segments: list[Segment] = field(default_factory=list)
    links: list[LinkRef] = field(default_factory=list)
    images: list[ImageRef] = field(default_factory=list)
    content_selector_matched: bool = True

    @property
    def text_length(self) -> int:
        """Total visible text length — compared against ``min_content_chars`` (R-ING-2)."""
        return sum(len(s.text) for s in self.segments)


def _clean(text: str) -> str:
    """Collapse runs of whitespace (for prose/tables/headings, never for code)."""
    return " ".join(text.split())


def _heading_path(stack: list[tuple[int, str]]) -> str:
    return " > ".join(text for _, text in stack)


def _push_heading(stack: list[tuple[int, str]], level: int, text: str) -> None:
    while stack and stack[-1][0] >= level:
        stack.pop()
    stack.append((level, text))


def _linearize_table(node: Node) -> str:
    rows: list[str] = []
    for tr in node.css("tr"):
        cells = [_clean(cell.text()) for cell in tr.css("td, th")]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _add_link(node: Node, doc: ExtractedDoc) -> None:
    href = node.attributes.get("href")
    if href:
        doc.links.append(LinkRef(target=href, anchor=_clean(node.text())))


def _add_image(node: Node, heading_path: str, doc: ExtractedDoc, caption: str = "") -> None:
    src = node.attributes.get("src")
    if src:
        doc.images.append(
            ImageRef(
                src=src,
                alt=node.attributes.get("alt") or "",
                caption=caption,
                heading_path=heading_path,
            )
        )


def _collect_links_images(node: Node, heading_path: str, doc: ExtractedDoc) -> None:
    for a in node.css("a[href]"):
        _add_link(a, doc)
    for img in node.css("img"):
        _add_image(img, heading_path, doc)


def _walk(node: Node, stack: list[tuple[int, str]], doc: ExtractedDoc) -> None:
    tag = node.tag
    if not tag or tag.startswith("_") or tag == "-text":
        return
    tag = tag.lower()
    if tag in _SKIP:
        return

    if tag in _HEADINGS:
        text = _clean(node.text())
        _push_heading(stack, int(tag[1]), text)
        if text:
            doc.segments.append(Segment("body", text, _heading_path(stack)))
        return

    hp = _heading_path(stack)

    if tag == "pre":
        code = node.text().rstrip("\n")
        if code.strip():
            doc.segments.append(Segment("code", code, hp))
        _collect_links_images(node, hp, doc)
        return

    if tag == "table":
        text = _linearize_table(node)
        if text:
            doc.segments.append(Segment("table", text, hp))
        _collect_links_images(node, hp, doc)
        return

    if tag == "figure":
        cap_node = node.css_first("figcaption")
        caption = _clean(cap_node.text()) if cap_node else ""
        for img in node.css("img"):
            _add_image(img, hp, doc, caption=caption)
        if caption:
            doc.segments.append(Segment("body", caption, hp))
        return

    if tag == "img":
        _add_image(node, hp, doc)
        return

    if tag in _TEXT_BLOCKS:
        text = _clean(node.text())
        if text:
            doc.segments.append(Segment("body", text, hp))
        _collect_links_images(node, hp, doc)
        return

    if tag == "a":
        _add_link(node, doc)
        text = _clean(node.text())
        if text:
            doc.segments.append(Segment("body", text, hp))
        return

    for child in node.iter(include_text=False):
        _walk(child, stack, doc)


def extract_html(
    html: str,
    *,
    content_selector: str = "",
    strip_selectors: Sequence[str] = (),
) -> ExtractedDoc:
    """Extract structured content from one HTML document (§7.3).

    Applies ``strip_selectors`` then scopes to ``content_selector`` (falling back to the
    whole page, loudly, if the selector matches nothing). Preserves code blocks whole,
    linearizes tables with ``|``, and captures links + images with their heading path.
    """
    tree = HTMLParser(html)
    for selector in strip_selectors:
        for node in tree.css(selector):
            node.decompose()

    matched = True
    root: Node | None
    if content_selector:
        root = tree.css_first(content_selector)
        if root is None:
            matched = False
            root = tree.body
    else:
        root = tree.body
    if root is None:
        root = tree.root

    title = ""
    title_node = tree.css_first("title")
    if title_node is not None:
        title = _clean(title_node.text())
    if not title and root is not None:
        h1 = root.css_first("h1")
        if h1 is not None:
            title = _clean(h1.text())

    doc = ExtractedDoc(title=title, content_selector_matched=matched)
    if root is not None:
        stack: list[tuple[int, str]] = []
        for child in root.iter(include_text=False):
            _walk(child, stack, doc)
    return doc
