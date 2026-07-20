"""Multi-format report OUTPUT (Phase 6 / GATE 6).

Render a cited report to **any format docusearch can read, except STDF** — PDF, DOCX, PPTX, XLSX
(``md``/``html`` stay in :mod:`report`). Same citation discipline: a citation outside the evidence
set is refused (R-CIT-1) before a byte is written. Reuses the format libraries' write side
(python-docx / python-pptx / openpyxl / reportlab) — all pip, no LibreOffice.
"""

from __future__ import annotations

import io
import re
from collections.abc import Mapping, Sequence
from typing import Any

from . import citations

EXPORT_FORMATS = ("pdf", "docx", "pptx", "xlsx")

# plotly needs a tick to draw before the print snapshot is taken
_PDF_SETTLE_MS = 250


class ExportDependencyError(RuntimeError):
    """A format's writer is not installed. The message names the exact command to fix it."""

# Control chars that are illegal in OOXML/XML (keep tab, LF, CR) — a NUL in a title/body otherwise
# crashes python-docx/openpyxl with a raw traceback (red-team M1).
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def xml_safe(text: str) -> str:
    """Strip control characters that OOXML writers reject."""
    return _CONTROL.sub("", text)


def xlsx_cell(value: str) -> str:
    """Sanitize a value for a spreadsheet cell: strip control chars AND neutralize **formula
    injection** — a leading ``= + - @`` is prefixed with ``'`` so the cell stays TEXT, never a live
    formula in whoever opens the file (CWE-1236, red-team H3)."""
    v = xml_safe(value)
    return "'" + v if v[:1] in ("=", "+", "-", "@") else v

_AI_WARNING = "AI-generated — verify every claim against the cited sources before relying on it."


def _sections(
    sections: Sequence[Mapping[str, str]] | None, body: str
) -> list[tuple[str, str]]:
    if sections:
        return [(str(s.get("heading", "")), str(s.get("body", ""))) for s in sections]
    return [("", body)] if body else []


def _paragraphs(body: str) -> list[str]:
    return [xml_safe(ln.strip()) for ln in body.splitlines() if ln.strip()]


def export_report(
    *,
    title: str,
    sections: Sequence[Mapping[str, str]] | None = None,
    body: str = "",
    subtitle: str = "",
    evidence: set[tuple[int, int]],
    fmt: str,
    request: str = "",
    requested_by: str = "",
    model: str = "",
    classification: str = "Confidential",
    ref_targets: Mapping[tuple[int, int], tuple[str, str]] | None = None,
    html: str = "",  # the rendered HTML report - required for fmt="pdf", ignored otherwise
    pptx_template: str = "",  # a .pptx/.potx whose theme + layouts the deck inherits
) -> bytes:
    """Render the report to ``fmt`` bytes. Raises ``CitationError`` if any citation references a
    ``(doc_id, chunk_id)`` outside ``evidence`` (R-CIT-1)."""
    fmt = fmt.lower()
    if fmt not in EXPORT_FORMATS:
        raise ValueError(f"unknown export format {fmt!r}; expected one of {EXPORT_FORMATS} (or md/html via report.render_report)")
    secs = _sections(sections, body)
    surface = "\n".join([title, subtitle, *(b for _, b in secs)])
    violations = citations.verify(surface, evidence)
    if violations:
        raise citations.CitationError(
            f"report cites {len(violations)} chunk(s) outside its evidence set: "
            + ", ".join(f"D:{c.doc_id}#{c.chunk_id}" for c in violations[:10])
        )
    secs, refs = _numbered(title, subtitle, secs, ref_targets, evidence)
    meta = [x for x in (f"Request: {request}" if request else "",
                        f"For: {requested_by}" if requested_by else "",
                        f"Model: {model}" if model else "",
                        f"Classification: {classification}") if x]
    if fmt == "docx":
        return _to_docx(title, subtitle, secs, meta, refs)
    if fmt == "pptx":
        return _to_pptx(title, subtitle, secs, meta, refs, template=pptx_template)
    if fmt == "xlsx":
        return _to_xlsx(title, secs, meta, refs)
    if not html:
        raise ValueError(
            "pdf export renders the HTML report - pass html=render_report(..., fmt='html')"
        )
    return html_to_pdf(html)


def _numbered(
    title: str,
    subtitle: str,
    secs: list[tuple[str, str]],
    ref_targets: Mapping[tuple[int, int], tuple[str, str]] | None,
    evidence: set[tuple[int, int]],
    base_url: str = "",
) -> tuple[list[tuple[str, str]], list[str]]:
    """Bodies with inline citations turned into ``[1]`` markers, plus the numbered reference list.

    Reuses the HTML/Markdown renderer's numbering so all six formats number identically
    (R-REUSE-2). A reference names the real document — ``store - title - heading`` — and never a
    bare ``D:<doc>#<chunk>``: that id is an internal key, and a report that shows one, or whose
    references are ids instead of documents, is not acceptable output.
    """
    from . import report

    numbering, ordered = report._collect_numbering([title, subtitle, *(b for _h, b in secs)])
    bodies = [(head, report._cite_md(body, numbering)) for head, body in secs]
    refs = [label for _href, label in report._references(ordered, base_url, ref_targets)]
    # evidence that was supplied but never cited still belongs in the list, after the cited ones
    cited = {(c.doc_id, c.chunk_id) for c in ordered}
    for key in sorted(evidence - cited):
        if ref_targets and key in ref_targets:
            refs.append(ref_targets[key][1])
    return bodies, [xml_safe(r) for r in refs]


def _styled(paragraph: object, *names: str) -> None:
    """Apply the first BUILT-IN style that exists in this document.

    Reports are styled by NAME only — no hardcoded fonts, sizes or colours — so a corporate
    .dotx or theme restyles the whole report on drop-in. A template need not define every
    built-in and python-docx raises on an unknown style, so fall through and finally leave the
    default rather than failing an export over cosmetics."""
    for name in names:
        try:
            paragraph.style = name  # type: ignore[attr-defined]
            return
        except (KeyError, ValueError):
            continue


def _to_docx(
    title: str, subtitle: str, secs: list[tuple[str, str]], meta: list[str], refs: list[str]
) -> bytes:
    from docx import Document

    doc = Document()
    doc.add_heading(xml_safe(title), 0)
    if subtitle:
        _styled(doc.add_paragraph(xml_safe(subtitle)), "Subtitle", "Body Text")
    _styled(doc.add_paragraph(_AI_WARNING), "Intense Quote", "Quote", "Body Text")
    for line in meta:
        _styled(doc.add_paragraph(xml_safe(line)), "Caption", "Body Text")
    for heading, bdy in secs:
        if heading:
            doc.add_heading(xml_safe(heading), level=1)
        for para in _paragraphs(bdy):
            _styled(doc.add_paragraph(para), "Body Text")
    if refs:
        doc.add_heading("References", level=1)
        for i, r in enumerate(refs, 1):
            _styled(doc.add_paragraph(f"{i}. {r}"), "Body Text")
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _layout(prs: Any, *names: str, fallback: int) -> Any:
    """A slide layout by NAME, falling back to an index.

    Every template orders its layouts differently, so taking ``slide_layouts[1]`` and hoping is
    how a themed deck comes out wrong. Match the standard names first."""
    wanted = {n.lower() for n in names}
    for layout in prs.slide_layouts:
        if layout.name.lower() in wanted:
            return layout
    return prs.slide_layouts[fallback if fallback < len(prs.slide_layouts) else 0]


def _body_placeholder(slide: Any) -> Any:
    """The slide's content placeholder, found by TYPE not index.

    ``placeholders[1]`` is the body only by convention; in a real template that index may be a
    picture, a date, or absent entirely."""
    from pptx.enum.shapes import PP_PLACEHOLDER

    holders = [p for p in slide.placeholders if p != slide.shapes.title]
    for kind in (PP_PLACEHOLDER.BODY, PP_PLACEHOLDER.OBJECT, PP_PLACEHOLDER.SUBTITLE):
        for holder in holders:
            if holder.placeholder_format.type == kind:
                return holder
    return holders[0] if holders else None


def _to_pptx(
    title: str, subtitle: str, secs: list[tuple[str, str]], meta: list[str], refs: list[str],
    *, template: str = "",
) -> bytes:
    """Build the deck from a template's LAYOUTS and PLACEHOLDERS only.

    Nothing here sets a font, colour or position, so pointing ``reports.pptx_template`` at any
    deck — or applying a different theme in PowerPoint afterwards — restyles the whole thing.
    With no template configured, python-pptx's default gives a clean Office look out of the box."""
    from pptx import Presentation

    prs = Presentation(template) if template else Presentation()
    cover = prs.slides.add_slide(_layout(prs, "Title Slide", fallback=0))
    if cover.shapes.title is not None:
        cover.shapes.title.text = xml_safe(title)
    sub = _body_placeholder(cover)
    if sub is not None:
        sub.text = xml_safe(subtitle) or _AI_WARNING
    body_layout = _layout(prs, "Title and Content", "Title and Body", fallback=1)
    for heading, bdy in secs:
        slide = prs.slides.add_slide(body_layout)
        if slide.shapes.title is not None:
            slide.shapes.title.text = xml_safe(heading) or xml_safe(title)
        holder = _body_placeholder(slide)
        if holder is not None:
            holder.text = "\n".join(_paragraphs(bdy)) or " "
    if refs:
        slide = prs.slides.add_slide(body_layout)
        if slide.shapes.title is not None:
            slide.shapes.title.text = "References"
        holder = _body_placeholder(slide)
        if holder is not None:
            holder.text = "\n".join(f"{i}. {r}" for i, r in enumerate(refs, 1))
    buf = io.BytesIO()
    prs.save(buf)
    return buf.getvalue()


def _to_xlsx(title: str, secs: list[tuple[str, str]], meta: list[str], refs: list[str]) -> bytes:
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "Report"
    ws.append([xlsx_cell(title)])
    ws.append([_AI_WARNING])
    for line in meta:
        ws.append([xlsx_cell(line)])
    for heading, bdy in secs:
        ws.append([])
        if heading:
            ws.append([xlsx_cell(heading)])
        for para in _paragraphs(bdy):
            ws.append([xlsx_cell(para)])
    if refs:
        ws.append([])
        ws.append(["References"])
        for r in refs:
            ws.append([xlsx_cell(r)])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def html_to_pdf(html: str) -> bytes:
    """Print a rendered HTML report to PDF in headless chromium.

    PDF is the one output format with no structured writer of its own, deliberately: rendering
    the real HTML means there is ONE layout to maintain instead of two that drift, and because
    chromium executes the page's JS, interactive plotly charts appear in the PDF. A direct
    writer (reportlab) or a JS-less converter (WeasyPrint) would silently drop them.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as err:  # actionable, one line (operability contract)
        raise ExportDependencyError(
            "PDF export needs the [export] extra: pip install 'docusearch[export]' "
            "&& python -m playwright install chromium"
        ) from err
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            try:
                page = browser.new_page()
                # The report is self-contained (inlined CSS/JS, data: images) so there is no
                # network to wait on, but plotly needs a tick to draw before the print snapshot.
                page.set_content(html, wait_until="domcontentloaded")
                page.wait_for_timeout(_PDF_SETTLE_MS)
                return bytes(
                    page.pdf(
                        format="Letter",
                        print_background=True,  # keep the report's banner/card styling
                        margin={"top": "0.6in", "bottom": "0.6in",
                                "left": "0.5in", "right": "0.5in"},
                    )
                )
            finally:
                browser.close()
    except ExportDependencyError:
        raise
    except Exception as err:  # a missing browser download is the common case
        raise ExportDependencyError(
            f"PDF export could not drive chromium ({type(err).__name__}). Run: "
            "python -m playwright install chromium"
        ) from err
