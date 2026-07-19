"""Format-equivalence writers (GATE 6): HTML → PPTX/XLSX for the derived corpus. A needle in the
HTML must survive conversion + re-extraction (the format-equivalence channel)."""

from __future__ import annotations

from pathlib import Path

import pytest

from docusearch import convert, ingest

_HTML = (
    "<html><head><title>SPI Guide</title></head><body>"
    "<main><h1>SPI</h1><h2>Timing</h2>"
    "<p>The SPI nonce ZQX7734 configures the peripheral bus.</p>"
    "<table><tr><th>Reg</th><th>Val</th></tr><tr><td>CTRL</td><td>0x1</td></tr></table>"
    "</main></body></html>"
)


@pytest.mark.parametrize("fmt", ["pptx", "xlsx"])
def test_html_to_office_preserves_needle_and_table(tmp_path: Path, fmt: str) -> None:
    data = convert._render_bytes(_HTML, fmt, content_selector="main")
    assert isinstance(data, bytes) and len(data) > 500
    p = tmp_path / f"doc.{fmt}"
    p.write_bytes(data)
    doc = ingest.extract_document(p, fmt)
    text = "\n".join(s.text for s in doc.segments)
    assert "ZQX7734" in text  # prose needle survived HTML -> fmt -> extract
    assert "CTRL" in text  # table cell survived


def test_convert_supports_new_formats() -> None:
    assert "pptx" in convert._SUPPORTED and "xlsx" in convert._SUPPORTED
