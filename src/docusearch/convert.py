"""Derived-corpus builder for format-layering validation (Phase 4, §17).

Converts a baseline HTML corpus to another format so the golden + needle suites can prove that
retrieval survives the conversion — i.e. that the format's extractor (Phase 4a: PyMuPDF)
recovers the content that HTML ingest indexed. This is **harness-only**: ``reportlab`` is lazy-
imported and lives in the ``[dev]`` extra, never the runtime deps (ingesting real PDFs needs
only ``[pdf]``/PyMuPDF).

The conversion preserves the document's *tokens* (every needle, identifier, and word), not its
visual formatting — a PDF's text layer carries no code/table structure anyway, so the point of
the derived corpus is to check that content, not styling, makes the round trip.

Public surface:
    html_to_pdf_bytes(html) -> bytes                 # one HTML doc's text -> a PDF
    convert_corpus(src_dir, dst_dir, fmt="pdf") -> ConvertResult
"""

from __future__ import annotations

import io
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING
from xml.sax.saxutils import escape

from .ingest import extract_html

if TYPE_CHECKING:
    from .config import SourceConfig

_SUPPORTED = ("pdf", "docx")


def html_to_pdf_bytes(
    html: str, *, content_selector: str = "", strip_selectors: Sequence[str] = ()
) -> bytes:
    """Render one HTML document's extracted text to a single-column PDF (reportlab).

    Every segment becomes a wrapped paragraph (heading paths as sub-headings), so all text
    tokens survive; PyMuPDF then recovers them at ingest. XML-special characters are escaped so
    reportlab's paragraph markup can't drop code containing ``<``/``>``/``&``.

    ``content_selector``/``strip_selectors`` mirror the source's ingest config so the derived
    PDF carries the same **cleaned** article text the HTML store indexes — not framework chrome.
    """
    from reportlab.lib.pagesizes import letter  # lazy: [dev] harness dependency
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    doc = extract_html(html, content_selector=content_selector, strip_selectors=list(strip_selectors))
    styles = getSampleStyleSheet()
    story: list[object] = []
    if doc.title:
        story.append(Paragraph(escape(doc.title), styles["Title"]))
    last_heading: str | None = None
    for seg in doc.segments:
        if seg.heading_path and seg.heading_path != last_heading:
            story.append(Paragraph(escape(seg.heading_path), styles["Heading2"]))
            last_heading = seg.heading_path
        # newlines -> spaces so code lines don't merge into an unbroken token run
        story.append(Paragraph(escape(seg.text.replace("\n", " ")), styles["BodyText"]))
        story.append(Spacer(1, 6))
    # Image alt/caption text carries needles too (§15.2 hides 10 nonces there); HTML ingest
    # indexes it as an image_ref chunk, and a faithful render of a broken <img> shows its alt.
    # Emit the same alt+caption string so those tokens survive the HTML -> PDF -> extract trip.
    for img in doc.images:
        caption = " ".join(t for t in (img.alt, img.caption) if t).strip()
        if caption:
            story.append(Paragraph(escape(caption), styles["Italic"]))
            story.append(Spacer(1, 6))
    if not story:  # never emit an empty PDF — keep the doc present for the audit counts
        story.append(Paragraph(escape(doc.title or "(no extractable text)"), styles["BodyText"]))
    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=letter, title=doc.title or "").build(story)
    return buf.getvalue()


def _add_docx_table(out: object, linearized: str) -> None:
    """Rebuild a real DOCX table from a ``a | b`` / newline-per-row linearized segment, so rows
    stay distinct (a flattened paragraph merges them) and ``extract_docx``'s table path is
    exercised end-to-end by the derived corpus (§15.4)."""
    grid = [row.split(" | ") for row in linearized.split("\n") if row.strip()]
    if not grid:
        return
    ncols = max(len(cells) for cells in grid)
    table = out.add_table(rows=len(grid), cols=ncols)  # type: ignore[attr-defined]
    for ri, cells in enumerate(grid):
        for ci, value in enumerate(cells):
            table.rows[ri].cells[ci].text = value


def html_to_docx_bytes(
    html: str, *, content_selector: str = "", strip_selectors: Sequence[str] = ()
) -> bytes:
    """Render one HTML document's extracted text to a DOCX (python-docx).

    Mirrors :func:`html_to_pdf_bytes`: heading paths become ``Heading`` paragraphs, every segment
    (body/code/table) becomes a paragraph, and image alt/caption text is emitted as a paragraph so
    the image-placement needles survive (§15.2/§15.4). ``extract_docx`` recovers all of it at
    ingest. ``content_selector``/``strip_selectors`` mirror the source's ingest config so the
    derived DOCX carries the same cleaned content the HTML store indexes.
    """
    from docx import Document  # lazy: python-docx ([docx] extra / [dev])

    doc = extract_html(html, content_selector=content_selector, strip_selectors=list(strip_selectors))
    out = Document()
    if doc.title:
        out.core_properties.title = doc.title
        out.add_heading(doc.title, level=1)
    last_heading: str | None = None
    for seg in doc.segments:
        if seg.heading_path and seg.heading_path != last_heading:
            out.add_heading(seg.heading_path, level=2)
            last_heading = seg.heading_path
        if seg.kind == "table":
            _add_docx_table(out, seg.text)  # a real DOCX table, so row boundaries survive
        else:
            # newlines -> spaces so code lines don't merge into an unbroken token run
            out.add_paragraph(seg.text.replace("\n", " "))
    for img in doc.images:
        caption = " ".join(t for t in (img.alt, img.caption) if t).strip()
        if caption:
            out.add_paragraph(caption)
    if not doc.segments and not doc.images:  # never emit an empty DOCX (keep it in audit counts)
        out.add_paragraph(doc.title or "(no extractable text)")
    buf = io.BytesIO()
    out.save(buf)
    return buf.getvalue()


def _render_bytes(
    html: str, fmt: str, *, content_selector: str = "", strip_selectors: Sequence[str] = ()
) -> bytes:
    """Dispatch to the format's HTML->bytes renderer (the derived-corpus analog of the ingest
    extractor dispatch). A new format adds one branch here plus its ``html_to_<fmt>_bytes``."""
    if fmt == "pdf":
        return html_to_pdf_bytes(html, content_selector=content_selector, strip_selectors=strip_selectors)
    if fmt == "docx":
        return html_to_docx_bytes(html, content_selector=content_selector, strip_selectors=strip_selectors)
    raise ValueError(f"unsupported target format {fmt!r}; supported: {_SUPPORTED}")


@dataclass
class ConvertResult:
    converted: int = 0
    errors: list[tuple[str, str]] = field(default_factory=list)  # (path, message)


def convert_corpus(src_dir: Path | str, dst_dir: Path | str, *, fmt: str = "pdf") -> ConvertResult:
    """Convert every HTML file under ``src_dir`` to ``fmt`` at the mirrored relative path under
    ``dst_dir`` (e.g. ``a/b.html`` -> ``a/b.pdf``). Returns audit counts; a per-file failure is
    recorded and the run continues."""
    if fmt not in _SUPPORTED:
        raise ValueError(f"unsupported target format {fmt!r}; supported: {_SUPPORTED}")
    src, dst = Path(src_dir), Path(dst_dir)
    result = ConvertResult()
    for html_path in sorted(src.rglob("*.htm*")):
        rel = html_path.relative_to(src).with_suffix(f".{fmt}")
        out = dst / rel
        try:
            html = html_path.read_bytes().decode("utf-8", errors="replace")
            data = _render_bytes(html, fmt)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(data)
            result.converted += 1
        except Exception as err:  # noqa: BLE001 - one bad file must not abort the batch
            result.errors.append((str(html_path), f"{type(err).__name__}: {err}"))
    return result


def convert_source(
    source: SourceConfig,
    dst_dir: Path | str,
    *,
    fmt: str = "pdf",
    progress: Callable[[int, int], None] | None = None,
) -> ConvertResult:
    """Convert exactly the files a source ingests — ``iter_files`` applies the same include/
    exclude globs (R-REUSE-2) — into ``fmt`` under ``dst_dir``, mirroring relative paths and
    applying the source's ``content_selector``/``strip_selectors`` so the derived corpus carries
    the same cleaned content the HTML store indexes. This is how a real ingestion is *altered
    for the format under test* (§15.4 / R-PROC-8). Per-file failures are recorded, not fatal.
    """
    if fmt not in _SUPPORTED:
        raise ValueError(f"unsupported target format {fmt!r}; supported: {_SUPPORTED}")
    from .ingest import iter_files  # lazy: keeps convert import light

    src_root, dst = Path(source.location), Path(dst_dir)
    files = list(iter_files(source.location, source.include, source.exclude))
    total = len(files)
    result = ConvertResult()
    for i, path in enumerate(files, 1):
        try:
            rel = path.relative_to(src_root).with_suffix(f".{fmt}")
            out = dst / rel
            html = path.read_bytes().decode("utf-8", errors="replace")
            data = _render_bytes(
                html,
                fmt,
                content_selector=source.content_selector,
                strip_selectors=source.strip_selectors,
            )
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(data)
            result.converted += 1
        except Exception as err:  # noqa: BLE001 - one bad file must not abort the batch
            result.errors.append((str(path), f"{type(err).__name__}: {err}"))
        if progress is not None:
            progress(i, total)
    return result
