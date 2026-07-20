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
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
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

# What a good report looks like in each format. The renderer can lay out whatever it is given,
# but it cannot invent structure that is not there: a section written as four dense paragraphs
# becomes four dense bullets on a slide. So the AUTHOR has to know the target before writing,
# and this is the single place that contract is stated — the MCP tool, help(), and the skill all
# read from here so they cannot drift.
FORMAT_GUIDANCE: dict[str, str] = {
    "md": "Prose. Full paragraphs, fenced code blocks, markdown lists. No length limit.",
    "html": "Prose, same as md — the richest layout. Cards render per section `kind`.",
    "html-slide": (
        "A DECK. One idea per section; 4-6 short bullets, each under ~15 words. Lead with the "
        "point, not the preamble. Put a long procedure in several sections rather than one. "
        "Code belongs in its own section, trimmed to the lines that matter."
    ),
    "pptx": (
        "A DECK. Same as html-slide: 4-6 short bullets per section, each under ~15 words, one "
        "idea per section. Write bullets as markdown list items ('- ') so they render as real "
        "bullets. Long sections are split across continuation slides and the full prose is kept "
        "in the speaker notes, but a section written as paragraphs still reads as a wall of "
        "text — write it short."
    ),
    "xlsx": (
        "A GRID, one row per point. Write each section as a markdown list where every item is "
        "one self-contained fact; use nested items ('  - ') for supporting detail. Avoid long "
        "paragraphs: a paragraph becomes one enormous cell."
    ),
    "docx": (
        "A DOCUMENT. Prose with headings; short paragraphs beat one long one. Use markdown "
        "lists for steps and enumerations so they render as real Word lists."
    ),
    "pdf": "A DOCUMENT — identical to docx guidance; PDF is printed from the HTML rendering.",
}


def guidance(fmt: str) -> str:
    """Authoring guidance for a target format (empty for an unknown one)."""
    return FORMAT_GUIDANCE.get(fmt.lower(), "")


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



# A slide holds a handful of points before it becomes a wall of text. Past this the section
# continues on another slide rather than overflowing off the bottom.
_SLIDE_BULLETS = 6
_SLIDE_CHARS = 160  # a "point" longer than this is prose, not a bullet


def _points(body: str) -> list[tuple[int, str]]:
    """A body as ``(indent level, text)`` bullet points.

    Markdown list markers become real bullet levels; a fenced code block keeps its lines
    verbatim. Prose that is not a list still yields one point per paragraph, so a document-shaped
    section degrades to readable bullets instead of one unbroken blob."""
    points: list[tuple[int, str]] = []
    in_code = False
    for raw in body.splitlines():
        line = raw.rstrip()
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if not line.strip():
            continue
        if in_code:
            points.append((1, xml_safe(line.strip())))
            continue
        indent = (len(line) - len(line.lstrip())) // 2
        text = line.strip()
        if text[:2] in ("- ", "* ") or (text[:1].isdigit() and text[1:3] in (". ", ") ")):
            text = text.split(" ", 1)[1] if " " in text else text
            points.append((min(indent, 4), xml_safe(text)))
        else:
            points.append((0, xml_safe(text)))
    return points


def _slide_chunks(points: list[tuple[int, str]]) -> list[list[tuple[int, str]]]:
    """Split points across slides, keeping long prose points on lighter slides."""
    chunks: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] = []
    budget = 0
    for level, text in points:
        cost = 2 if len(text) > _SLIDE_CHARS else 1
        if current and budget + cost > _SLIDE_BULLETS:
            chunks.append(current)
            current, budget = [], 0
        current.append((level, text))
        budget += cost
    if current:
        chunks.append(current)
    return chunks or [[]]


def _fill(holder: Any, points: list[tuple[int, str]]) -> None:
    """Write real bullet PARAGRAPHS, not one newline-joined blob.

    Setting ``.text`` with newlines produces a single paragraph that neither indents nor bullets
    and simply overflows the placeholder — the reason decks looked like dumped documents."""
    frame = holder.text_frame
    frame.clear()
    frame.word_wrap = True
    for i, (level, text) in enumerate(points or [(0, " ")]):
        para = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        para.text = text
        para.level = level


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

    # Exclude the title by placeholder TYPE, not by object identity: python-pptx hands back a new
    # wrapper each time you touch `shapes.title`, so `p != slide.shapes.title` can be true for the
    # title itself — and the title would then be filled with body text.
    titles = (PP_PLACEHOLDER.TITLE, PP_PLACEHOLDER.CENTER_TITLE)
    holders = [p for p in slide.placeholders if p.placeholder_format.type not in titles]
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
        chunks = _slide_chunks(_points(bdy))
        for n, chunk in enumerate(chunks):
            slide = prs.slides.add_slide(body_layout)
            head = xml_safe(heading) or xml_safe(title)
            if slide.shapes.title is not None:
                slide.shapes.title.text = head if n == 0 else f"{head} (cont.)"
            holder = _body_placeholder(slide)
            if holder is not None:
                _fill(holder, chunk)
            # the full prose lives in the speaker notes, so shortening the slide loses nothing
            if n == 0 and bdy.strip():
                with suppress(AttributeError, KeyError):  # a template without a notes master
                    slide.notes_slide.notes_text_frame.text = xml_safe(bdy.strip())
    if refs:
        numbered = [(0, f"{i}. {r}") for i, r in enumerate(refs, 1)]
        for n, chunk in enumerate(_slide_chunks(numbered)):
            slide = prs.slides.add_slide(body_layout)
            if slide.shapes.title is not None:
                slide.shapes.title.text = "References" if n == 0 else "References (cont.)"
            holder = _body_placeholder(slide)
            if holder is not None:
                _fill(holder, chunk)
    buf = io.BytesIO()
    prs.save(buf)
    return buf.getvalue()


def _to_xlsx(title: str, secs: list[tuple[str, str]], meta: list[str], refs: list[str]) -> bytes:
    """A spreadsheet is a GRID, so emit rows and columns — not a document poured down column A.

    Section / Point / Detail lets you filter and sort by section, which is the only reason to
    want a report as a workbook. Provenance and references get their own sheets rather than
    interrupting the data."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font

    wb = Workbook()
    ws = wb.active
    ws.title = "Report"
    ws.append(["Section", "Point", "Detail"])
    for cell in ws[1]:
        cell.font = Font(bold=True)
    for heading, bdy in secs:
        points = _points(bdy)
        for level, text in points:
            ws.append([
                xlsx_cell(heading),
                xlsx_cell(text) if level == 0 else "",
                xlsx_cell(text) if level > 0 else "",
            ])
    ws.freeze_panes = "A2"  # the header stays put while you scroll
    ws.auto_filter.ref = ws.dimensions  # filter/sort by section out of the box
    for col, width in (("A", 28), ("B", 90), ("C", 90)):
        ws.column_dimensions[col].width = width
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)

    info = wb.create_sheet("About")
    info.append(["Title", xlsx_cell(title)])
    info.append(["Notice", _AI_WARNING])
    for line in meta:
        key, _, value = line.partition(": ")
        info.append([xlsx_cell(key), xlsx_cell(value)])
    info.column_dimensions["A"].width = 18
    info.column_dimensions["B"].width = 90

    if refs:
        sheet = wb.create_sheet("References")
        sheet.append(["#", "Source"])
        for cell in sheet[1]:
            cell.font = Font(bold=True)
        for i, r in enumerate(refs, 1):
            sheet.append([i, xlsx_cell(r)])
        sheet.freeze_panes = "A2"
        sheet.column_dimensions["A"].width = 5
        sheet.column_dimensions["B"].width = 110

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
    def render() -> bytes:
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

    try:
        # Playwright's SYNC api refuses to run inside a running asyncio loop, and the server is
        # async — calling it on the request thread fails with a bare "Error" that says nothing.
        # A worker thread has no loop of its own, so this works from both the async server and a
        # plain synchronous caller like the CLI.
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(render).result()
    except ExportDependencyError:
        raise
    except Exception as err:  # a missing browser download is the common case
        raise ExportDependencyError(
            f"PDF export could not drive chromium ({type(err).__name__}). Run: "
            "python -m playwright install chromium"
        ) from err
