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
    chunk_document(doc, *, chunk_tokens, overlap) -> list[Chunk]   # §7.6
    Chunk                                                     # a ready-to-index chunk
    run_ingest(config, store, *, force) -> IngestResult       # the whole pipeline (§7)
    IngestResult                                              # per-run audit counts
    render_ingest_audit(result, *, run_id) -> str             # §7.8 report
    render_store_audit(store) -> str                          # `docusearch audit`
"""

from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from selectolax.parser import HTMLParser, Node

from . import runlog
from .config import Config, SourceConfig
from .store import Store

_HASH_BLOCK = 1 << 20

_HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_SKIP = {"script", "style", "template", "svg", "math", "noscript"}
# Block-level tags: an element with NO block-level element child is a "leaf block" whose
# text is emitted whole (with a separator so inline children never glue); an element WITH
# a block child is a container we recurse into (buffering its own inline/text runs).
# Everything not listed here is treated as inline. pre/table/figure/img are special-cased.
_BLOCK = _HEADINGS | {
    "html",
    "body",
    "pre",
    "table",
    "figure",
    "img",
    "div",
    "section",
    "article",
    "main",
    "header",
    "footer",
    "aside",
    "nav",
    "p",
    "ul",
    "ol",
    "dl",
    "li",
    "dd",
    "dt",
    "blockquote",
    "form",
    "fieldset",
    "details",
    "summary",
    "address",
    "hr",
    "figcaption",
    "tbody",
    "thead",
    "tfoot",
    "tr",
    "td",
    "th",
    "caption",
}


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


def _classify_files(
    location: Path | str,
    include: Sequence[str],
    exclude: Sequence[str],
) -> tuple[list[Path], int, int]:
    """Sort files under ``location`` into (included, excluded_by_glob, other) for the audit.

    A file matching any exclude glob is excluded; otherwise it is included if it matches
    any include glob (an empty include list matches everything, R-ING-1); anything left
    is "other" (present but not selected). Included files come back deterministically
    sorted for reproducible ingest order.
    """
    root = Path(location)
    inc = [_glob_to_regex(p) for p in include] or [re.compile(".*")]
    exc = [_glob_to_regex(p) for p in exclude]
    included: list[Path] = []
    excluded_glob = 0
    other = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if any(r.match(rel) for r in exc):
            excluded_glob += 1
        elif any(r.match(rel) for r in inc):
            included.append(path)
        else:
            other += 1
    return included, excluded_glob, other


def iter_files(
    location: Path | str,
    include: Sequence[str],
    exclude: Sequence[str],
) -> Iterator[Path]:
    """Yield files under ``location`` matching any include glob and no exclude glob."""
    return iter(_classify_files(location, include, exclude)[0])


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
    caption = node.css_first("caption")
    if caption is not None:
        cap = _clean(caption.text(separator=" "))
        if cap:
            rows.append(cap)
    for tr in node.css("tr"):
        cells = [_clean(cell.text(separator=" ")) for cell in tr.css("td, th")]
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


def _has_block_child(node: Node) -> bool:
    for child in node.iter(include_text=False):
        ct = child.tag
        if ct and ct.lower() in _BLOCK:
            return True
    return False


def _emit_body(doc: ExtractedDoc, text: str, heading_path: str) -> None:
    if text:
        doc.segments.append(Segment("body", text, heading_path))


def _walk(node: Node, stack: list[tuple[int, str]], doc: ExtractedDoc) -> None:
    tag = node.tag
    if not tag or tag.startswith("_") or tag == "-text":
        return
    tag = tag.lower()
    if tag in _SKIP:
        return

    if tag in _HEADINGS:
        text = _clean(node.text(separator=" "))
        _push_heading(stack, int(tag[1]), text)
        _emit_body(doc, text, _heading_path(stack))
        return

    if tag == "pre":  # code: preserve exact whitespace (no separator injection)
        code = node.text().rstrip("\n")
        if code.strip():
            doc.segments.append(Segment("code", code, _heading_path(stack)))
        _collect_links_images(node, _heading_path(stack), doc)
        return

    if tag == "table":
        text = _linearize_table(node)
        if text:
            doc.segments.append(Segment("table", text, _heading_path(stack)))
        _collect_links_images(node, _heading_path(stack), doc)
        return

    if tag == "figure":
        cap_node = node.css_first("figcaption")
        caption = _clean(cap_node.text(separator=" ")) if cap_node else ""
        for img in node.css("img"):
            _add_image(img, _heading_path(stack), doc, caption=caption)
        _emit_body(doc, caption, _heading_path(stack))
        return

    if tag == "img":
        _add_image(node, _heading_path(stack), doc)
        return

    # Leaf block: no block-level element child -> emit its whole text with a separator so
    # inline children (span/code/strong/a) never glue together (red-team Finding 1).
    if not _has_block_child(node):
        _emit_body(doc, _clean(node.text(separator=" ")), _heading_path(stack))
        _collect_links_images(node, _heading_path(stack), doc)
        return

    # Container with block children: emit its own inline/text runs, recurse block children.
    parts: list[str] = []

    def _flush() -> None:
        _emit_body(doc, _clean(" ".join(parts)), _heading_path(stack))
        parts.clear()

    for child in node.iter(include_text=True):
        ctag = child.tag
        if ctag == "-text":
            parts.append(child.text() or "")
        elif not ctag or ctag.startswith("_") or ctag.lower() in _SKIP:
            continue
        elif ctag.lower() in _BLOCK:
            _flush()
            _walk(child, stack, doc)
        else:  # inline element: keep its text with the current run, capture its links/images
            parts.append(child.text(separator=" "))
            if ctag.lower() == "a":
                _add_link(child, doc)
            _collect_links_images(child, _heading_path(stack), doc)
    _flush()


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
        _walk(root, [], doc)
    return doc


# ----------------------------------------------------------------- chunking


@dataclass
class Chunk:
    """A ready-to-index unit of a document, with its heading-path locator (R-ING-4)."""

    ord: int
    kind: str  # body | code
    locator: str
    text: str


def chunk_document(doc: ExtractedDoc, *, chunk_tokens: int, overlap: int) -> list[Chunk]:
    """Split a document into chunks (§7.6).

    Body/table text is grouped by heading path and packed toward ``chunk_tokens`` words
    with a sliding ``overlap``; a **code** segment always becomes one whole chunk, never
    split. Each chunk records its heading-path locator (R-ING-4).
    """
    ov = min(max(overlap, 0), max(chunk_tokens - 1, 0))
    chunks: list[Chunk] = []
    buf: list[str] = []
    hp: str | None = None
    emitted = False  # has this heading group produced a chunk yet?
    fresh = False  # has new (non-overlap) content been added since the last emit?

    def emit() -> None:
        nonlocal emitted
        if buf:
            chunks.append(Chunk(len(chunks), "body", hp or "", " ".join(buf)))
            emitted = True

    def close_group() -> None:
        nonlocal buf, fresh
        if buf and (fresh or not emitted):
            emit()
        buf = []
        fresh = False

    for seg in doc.segments:
        if seg.kind == "code":
            close_group()
            emitted = False
            chunks.append(Chunk(len(chunks), "code", seg.heading_path, seg.text))
            hp = None
            continue

        if hp is None:
            hp, emitted, fresh = seg.heading_path, False, False
        elif seg.heading_path != hp:
            close_group()
            hp, emitted = seg.heading_path, False

        for word in seg.text.split():
            buf.append(word)
            fresh = True
            if len(buf) >= chunk_tokens:
                emit()
                buf = buf[-ov:] if ov > 0 else []
                fresh = False

    close_group()
    return chunks


# ----------------------------------------------------------------- orchestration

_EXTERNAL_PREFIXES = ("http://", "https://", "ftp://", "mailto:", "javascript:", "tel:", "data:")


@dataclass
class IngestResult:
    """Counts and audit data for one ingest run (§7.8) — the input to Gate 1."""

    files_found: int = 0
    included: int = 0
    excluded_glob: int = 0
    other_files: int = 0
    skipped_unchanged: int = 0
    stripped_empty: int = 0
    parse_errors: int = 0
    documents: int = 0
    chunks: int = 0
    images: int = 0
    relations_total: int = 0
    relations_resolved: int = 0
    relations_unresolved: int = 0
    content_selector_misses: int = 0
    zero_chunk_docs: int = 0
    untagged_audience_docs: int = 0
    per_extension: dict[str, int] = field(default_factory=dict)
    errors: list[tuple[str, str]] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)


def _resolve_local(raw: str, src_path: Path) -> Path | None:
    """Resolve a raw href/src to a local path, or None if it is an external/data URL."""
    target = raw.split("#", 1)[0].split("?", 1)[0].strip()
    if not target or target.startswith("//") or target.lower().startswith(_EXTERNAL_PREFIXES):
        return None
    return (src_path.parent / target).resolve()


def _stage_images(
    doc: ExtractedDoc,
    src_path: Path,
    staging_dir: Path,
    store: Store,
    doc_id: int,
    base_ord: int,
    result: IngestResult,
) -> int:
    """Retain image originals (keyed by sha256) and add searchable image_ref chunks."""
    images_dir = staging_dir / "images"
    added = 0
    for img in doc.images:
        local = _resolve_local(img.src, src_path)
        if local is not None and local.is_file():
            data = local.read_bytes()
            sha = hashlib.sha256(data).hexdigest()
            ext = local.suffix.lstrip(".").lower() or "bin"
            images_dir.mkdir(parents=True, exist_ok=True)
            dest = images_dir / f"{sha}.{ext}"
            if not dest.exists():
                dest.write_bytes(data)
            store.add_image(
                sha256=sha,
                ext=ext,
                doc_id=doc_id,
                locator=img.heading_path,
                alt=img.alt,
                caption=img.caption,
                num_bytes=len(data),
            )
            result.images += 1
        text = " ".join(t for t in (img.alt, img.caption, img.heading_path) if t).strip()
        if text:
            store.add_chunk(
                document_id=doc_id,
                ord=base_ord + added,
                text=text,
                kind="image_ref",
                locator=img.heading_path,
            )
            added += 1
    return added


def _ingest_file(
    path: Path,
    source: SourceConfig,
    config: Config,
    store: Store,
    *,
    force: bool,
    result: IngestResult,
) -> None:
    ext = path.suffix.lstrip(".").lower() or "html"
    result.per_extension[ext] = result.per_extension.get(ext, 0) + 1
    abspath = path.resolve().as_posix()

    file_hash = content_hash(path)
    existing = store.document_content_hash(abspath)
    if existing is not None and existing == file_hash and not force:
        result.skipped_unchanged += 1
        return
    if existing is not None:
        doc_id_old = store.document_id_for_path(abspath)
        if doc_id_old is not None:
            store.delete_document(doc_id_old)

    try:
        html = path.read_bytes().decode("utf-8", errors="replace")
        doc = extract_html(
            html,
            content_selector=source.content_selector,
            strip_selectors=source.strip_selectors,
        )
    except Exception as err:  # a genuinely unparseable file — reported, not fatal
        result.parse_errors += 1
        result.errors.append((abspath, f"{type(err).__name__}: {err}"))
        return

    if not doc.content_selector_matched:
        result.content_selector_misses += 1
    if doc.text_length < source.min_content_chars:
        result.stripped_empty += 1
        return
    if not source.audience:
        result.untagged_audience_docs += 1

    doc_id = store.add_document(
        path=abspath,
        source=source.name,
        title=doc.title,
        content_hash=file_hash,
        content_type="documentation",
        fmt=ext,
        audience=source.audience,
        mtime=path.stat().st_mtime,
        status="active",
    )
    result.documents += 1

    chunks = chunk_document(
        doc, chunk_tokens=config.index.chunk_tokens, overlap=config.index.chunk_overlap
    )
    for chunk in chunks:
        store.add_chunk(
            document_id=doc_id,
            ord=chunk.ord,
            text=chunk.text,
            kind=chunk.kind,
            locator=chunk.locator,
        )
    image_chunks = _stage_images(
        doc, path, Path(config.paths.staging_dir), store, doc_id, len(chunks), result
    )
    total_chunks = len(chunks) + image_chunks
    result.chunks += total_chunks
    if total_chunks == 0:
        result.zero_chunk_docs += 1

    for link in doc.links:
        store.add_relation(src_doc=doc_id, dst_raw=link.target, link_type=link.link_type)


def _resolve_links(store: Store) -> None:
    """Post-pass: resolve raw link targets to document ids now that all docs exist (R-ING-5)."""
    path_to_id = store.document_path_to_id()
    for rel_id, src_path, dst_raw in store.unresolved_relations():
        local = _resolve_local(dst_raw, Path(src_path))
        if local is not None:
            dst = path_to_id.get(local.as_posix())
            if dst is not None:
                store.set_relation_dst(rel_id, dst)


def run_ingest(config: Config, store: Store, *, force: bool = False) -> IngestResult:
    """Ingest every configured source into the store and return an audit result (§7)."""
    result = IngestResult()
    start = time.perf_counter()
    for source in config.sources:
        root = Path(source.location)
        if not root.is_dir():
            result.errors.append((source.location, "source location not found"))
            continue
        included, excluded_glob, other = _classify_files(root, source.include, source.exclude)
        result.files_found += len(included) + excluded_glob + other
        result.included += len(included)
        result.excluded_glob += excluded_glob
        result.other_files += other
        for path in included:
            _ingest_file(path, source, config, store, force=force, result=result)

    _resolve_links(store)
    result.relations_total = store.count_relations()
    result.relations_resolved = store.count_resolved_relations()
    result.relations_unresolved = result.relations_total - result.relations_resolved
    result.timings_ms["total"] = (time.perf_counter() - start) * 1000.0
    runlog.log(
        "ingest.done",
        documents=result.documents,
        chunks=result.chunks,
        images=result.images,
        skipped_unchanged=result.skipped_unchanged,
        stripped_empty=result.stripped_empty,
        parse_errors=result.parse_errors,
        relations_unresolved=result.relations_unresolved,
        took_ms=round(result.timings_ms["total"], 1),
    )
    return result


# ----------------------------------------------------------------- audit rendering


def render_ingest_audit(result: IngestResult, *, run_id: str = "") -> str:
    """Render the ingest audit report (§7.8) — the input to Gate 1.

    Loudly surfaces everything that was skipped or is suspect (nothing dropped
    silently): glob exclusions, too-short strips, parse errors, content-selector
    misses, zero-chunk docs, untagged audiences, and unresolved links.
    """
    took = result.timings_ms.get("total", 0.0)
    lines = [
        "# Ingest audit",
        "",
        f"run_id: `{run_id}`  ·  elapsed: {took / 1000:.2f}s",
        "",
        "## Discovery",
        f"- files found: **{result.files_found}**",
        f"- included (passed globs): **{result.included}**",
        f"- excluded by glob: **{result.excluded_glob}**",
        f"- other (present, not selected): **{result.other_files}**",
        "",
        "## Documents & chunks",
        f"- documents ingested this run: **{result.documents}**",
        f"- skipped (unchanged hash): **{result.skipped_unchanged}**",
        f"- chunks written: **{result.chunks}**",
        f"- images retained: **{result.images}**",
        "",
        "## Relations (cross-references)",
        f"- total: **{result.relations_total}**",
        f"- resolved: **{result.relations_resolved}**",
        f"- unresolved (external / broken — kept for audit): **{result.relations_unresolved}**",
        "",
        "## ⚠️ Skips & anomalies (review every non-zero line)",
        f"- stripped as too short (< min_content_chars): **{result.stripped_empty}**",
        f"- content_selector matched nothing (fell back to body): **{result.content_selector_misses}**",
        f"- zero-chunk documents: **{result.zero_chunk_docs}**",
        f"- documents with no audience tag: **{result.untagged_audience_docs}**",
        f"- parse errors: **{result.parse_errors}**",
        "",
        "## Per-extension",
    ]
    if result.per_extension:
        lines += [f"- `{ext}`: {n}" for ext, n in sorted(result.per_extension.items())]
    else:
        lines.append("- (none)")
    if result.errors:
        lines += ["", "## Errors", *[f"- `{path}` — {msg}" for path, msg in result.errors[:200]]]
        if len(result.errors) > 200:
            lines.append(f"- … and {len(result.errors) - 200} more")
    lines.append("")
    return "\n".join(lines)


def render_store_audit(store: Store) -> str:
    """Render the current index state for ``docusearch audit`` (spot-check counts)."""
    resolved = store.count_resolved_relations()
    total_rel = store.count_relations()
    histogram = store.fmt_histogram()
    fmt_lines = [f"- `{fmt}`: {n}" for fmt, n in histogram.items()] or ["- (none)"]
    lines = [
        "# Index audit",
        "",
        f"- documents: **{store.count_documents()}**",
        f"- chunks: **{store.count_chunks()}**",
        f"- images: **{store.count_images()}**",
        f"- relations: **{total_rel}** ({resolved} resolved, {total_rel - resolved} unresolved)",
        "",
        "## By format",
        *fmt_lines,
        "",
        "## ⚠️ Anomalies",
        f"- documents with zero chunks: **{store.documents_without_chunks()}**",
        f"- documents with empty audience: **{store.documents_with_empty_audience()}**",
        "",
    ]
    return "\n".join(lines)
