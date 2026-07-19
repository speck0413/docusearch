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

from . import citations

EXPORT_FORMATS = ("pdf", "docx", "pptx", "xlsx")

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
    refs = _reference_lines(evidence, ref_targets)
    meta = [x for x in (f"Request: {request}" if request else "",
                        f"For: {requested_by}" if requested_by else "",
                        f"Model: {model}" if model else "",
                        f"Classification: {classification}") if x]
    if fmt == "docx":
        return _to_docx(title, subtitle, secs, meta, refs)
    if fmt == "pptx":
        return _to_pptx(title, subtitle, secs, meta, refs)
    if fmt == "xlsx":
        return _to_xlsx(title, secs, meta, refs)
    return _to_pdf(title, subtitle, secs, meta, refs)


def _reference_lines(
    evidence: set[tuple[int, int]], ref_targets: Mapping[tuple[int, int], tuple[str, str]] | None
) -> list[str]:
    out: list[str] = []
    for (doc_id, chunk_id) in sorted(evidence):
        label = ""
        if ref_targets and (doc_id, chunk_id) in ref_targets:
            label = ref_targets[(doc_id, chunk_id)][1]
        out.append(f"[D:{doc_id}#{chunk_id}] {label}".rstrip())
    return out


def _to_docx(
    title: str, subtitle: str, secs: list[tuple[str, str]], meta: list[str], refs: list[str]
) -> bytes:
    from docx import Document

    doc = Document()
    doc.add_heading(xml_safe(title), 0)
    if subtitle:
        doc.add_paragraph(xml_safe(subtitle))
    doc.add_paragraph(_AI_WARNING)
    for line in meta:
        doc.add_paragraph(xml_safe(line))
    for heading, bdy in secs:
        if heading:
            doc.add_heading(xml_safe(heading), level=1)
        for para in _paragraphs(bdy):
            doc.add_paragraph(para)
    if refs:
        doc.add_heading("References", level=1)
        for r in refs:
            doc.add_paragraph(xml_safe(r), style="List Bullet")
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _to_pptx(
    title: str, subtitle: str, secs: list[tuple[str, str]], meta: list[str], refs: list[str]
) -> bytes:
    from pptx import Presentation

    prs = Presentation()
    cover = prs.slides.add_slide(prs.slide_layouts[0])
    cover.shapes.title.text = xml_safe(title)
    if cover.slide_layout.placeholders and len(cover.placeholders) > 1:
        cover.placeholders[1].text = xml_safe(subtitle) or _AI_WARNING
    for heading, bdy in secs:
        slide = prs.slides.add_slide(prs.slide_layouts[1])
        slide.shapes.title.text = xml_safe(heading) or xml_safe(title)
        slide.placeholders[1].text = "\n".join(_paragraphs(bdy)) or " "
    if refs:
        ref_slide = prs.slides.add_slide(prs.slide_layouts[1])
        ref_slide.shapes.title.text = "References"
        ref_slide.placeholders[1].text = "\n".join(xml_safe(r) for r in refs)
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


def _to_pdf(
    title: str, subtitle: str, secs: list[tuple[str, str]], meta: list[str], refs: list[str]
) -> bytes:
    from html import escape

    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    styles = getSampleStyleSheet()
    flow = [Paragraph(escape(xml_safe(title)), styles["Title"])]
    if subtitle:
        flow.append(Paragraph(escape(xml_safe(subtitle)), styles["Italic"]))
    flow.append(Paragraph(f"<i>{escape(_AI_WARNING)}</i>", styles["BodyText"]))
    for line in meta:
        flow.append(Paragraph(escape(xml_safe(line)), styles["BodyText"]))
    flow.append(Spacer(1, 12))
    for heading, bdy in secs:
        if heading:
            flow.append(Paragraph(escape(xml_safe(heading)), styles["Heading1"]))
        for para in _paragraphs(bdy):
            flow.append(Paragraph(escape(para), styles["BodyText"]))
        flow.append(Spacer(1, 8))
    if refs:
        flow.append(Paragraph("References", styles["Heading1"]))
        for r in refs:
            flow.append(Paragraph(escape(xml_safe(r)), styles["BodyText"]))
    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=letter, title=title).build(flow)
    return buf.getvalue()
