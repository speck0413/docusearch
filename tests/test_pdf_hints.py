"""PDF font pre-analysis → config hints (task #34): analyse a sample of PDFs' font sizes and report
how their headings will be inferred at ingest, so a user (or the bootstrap command) can see it."""

from __future__ import annotations

from pathlib import Path

from docusearch import ingest


def _multi_font_pdf(path: Path) -> None:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=letter)
    c.setFont("Helvetica-Bold", 22)
    c.drawString(72, 740, "Chapter One")
    c.setFont("Helvetica-Bold", 15)
    c.drawString(72, 700, "A Section")
    c.setFont("Helvetica", 11)
    for i, y in enumerate(range(660, 400, -20)):  # lots of body text at 11pt = the dominant size
        c.drawString(72, y, f"Body paragraph line {i} at the ordinary reading size here.")
    c.showPage()
    c.save()


def _flat_pdf(path: Path) -> None:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=letter)
    c.setFont("Helvetica", 11)
    for i, y in enumerate(range(740, 500, -20)):
        c.drawString(72, y, f"Uniform body line {i}, no heading sizes at all here.")
    c.showPage()
    c.save()


def test_font_profile_detects_headings(tmp_path: Path) -> None:
    p = tmp_path / "doc.pdf"
    _multi_font_pdf(p)
    prof = ingest.pdf_font_profile([p.read_bytes()])
    assert prof.sampled == 1 and prof.pages == 1
    assert prof.body_size == 11.0
    assert prof.detected
    # 22pt -> H1, 15pt -> H2 (ranked descending); body 11pt is not a heading
    assert prof.levels[22.0] == 1 and prof.levels[15.0] == 2 and 11.0 not in prof.levels
    assert prof.coverage[11.0] > prof.coverage[22.0]  # body dominates the histogram


def test_font_profile_flat_document(tmp_path: Path) -> None:
    p = tmp_path / "flat.pdf"
    _flat_pdf(p)
    prof = ingest.pdf_font_profile([p.read_bytes()])
    assert prof.body_size == 11.0 and not prof.detected and prof.levels == {}


def test_font_profile_multiple_pdfs_and_bad_bytes(tmp_path: Path) -> None:
    good = tmp_path / "a.pdf"
    _multi_font_pdf(good)
    prof = ingest.pdf_font_profile([good.read_bytes(), b"not a pdf at all", good.read_bytes()])
    assert prof.sampled == 2  # the garbage bytes are skipped, not fatal
    assert prof.detected and prof.pages == 2
