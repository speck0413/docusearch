"""Multi-format report OUTPUT (GATE 6): a cited report renders to PDF/DOCX/PPTX/XLSX, round-tripping
its content back through our own extractors, and refuses citations outside the evidence set."""

from __future__ import annotations

from pathlib import Path

import pytest

from docusearch import citations, ingest, report_export

_SPEC = {
    "title": "PA Control Overview",
    "subtitle": "how PA works",
    "sections": [
        {"heading": "What is PA", "body": "Protocol Aware drives serialized buses [D:10#5].\nIt uses ports."},
        {"heading": "Setup", "body": "Create a port then run frames [D:10#6]."},
    ],
    "evidence": {(10, 5), (10, 6)},
    "request": "overview of PA",
    "requested_by": "Stephen",
    "model": "claude-sonnet-5",
}


@pytest.mark.parametrize("fmt,ext", [("docx", "docx"), ("pptx", "pptx"), ("xlsx", "xlsx")])
def test_export_roundtrips_through_our_extractor(tmp_path: Path, fmt: str, ext: str) -> None:
    data = report_export.export_report(fmt=fmt, **_SPEC)  # type: ignore[arg-type]
    assert isinstance(data, bytes) and len(data) > 500
    p = tmp_path / f"report.{ext}"
    p.write_bytes(data)
    doc = ingest.extract_document(p, ext)
    text = "\n".join(s.text for s in doc.segments)
    assert "Protocol Aware" in text  # body content survived
    assert "PA Control Overview" in text or doc.title == "PA Control Overview"  # title present
    assert "Setup" in text  # second section heading/content present


def test_export_pdf_is_valid_and_extractable(tmp_path: Path) -> None:
    data = report_export.export_report(fmt="pdf", **_SPEC)  # type: ignore[arg-type]
    assert data.startswith(b"%PDF")  # a real PDF
    p = tmp_path / "report.pdf"
    p.write_bytes(data)
    doc = ingest.extract_document(p, "pdf")
    text = "\n".join(s.text for s in doc.segments)
    assert "Protocol Aware" in text


def test_export_refuses_citation_outside_evidence() -> None:
    bad = dict(_SPEC)
    bad["sections"] = [{"heading": "x", "body": "claim with a bad cite [D:99#99]."}]
    with pytest.raises(citations.CitationError):
        report_export.export_report(fmt="docx", **bad)  # type: ignore[arg-type]


def test_export_rejects_unknown_format() -> None:
    with pytest.raises(ValueError, match="export format"):
        report_export.export_report(fmt="rtf", **_SPEC)  # type: ignore[arg-type]
