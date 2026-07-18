"""Phase 4a — PDF format layering: PyMuPDF extractor, format dispatch, and needles-through-
conversion (§7.3, §15.4). PDFs are generated with reportlab (a [dev] harness dep); extraction
uses PyMuPDF (the [pdf] runtime extra)."""

from __future__ import annotations

from pathlib import Path

from docusearch import config
from docusearch.catalog import Catalog
from docusearch.convert import convert_corpus, html_to_pdf_bytes
from docusearch.ingest import extract_document, extract_pdf
from docusearch.store import Store


def _make_pdf(path: Path, pages: list[list[str]], *, link: str | None = None) -> None:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=letter)
    for i, lines in enumerate(pages):
        y = 720.0
        for ln in lines:
            c.drawString(72, y, ln)
            y -= 18
        if i == 0 and link:
            c.linkURL(link, (72, 700, 300, 720))
        c.showPage()
    c.save()


def test_extract_pdf_text_page_locators_and_links(tmp_path: Path) -> None:
    p = tmp_path / "doc.pdf"
    _make_pdf(
        p,
        [["SPI Protocol Overview", "The nonce ZQX-7734-FRB configures the bus."],
         ["Page two body with needle KLM-9021-END."]],
        link="https://example.com/spec",
    )
    doc = extract_pdf(p.read_bytes())
    assert [s.heading_path for s in doc.segments] == ["page 1", "page 2"]
    assert "ZQX-7734-FRB" in doc.segments[0].text
    assert "KLM-9021-END" in doc.segments[1].text
    assert all(s.kind == "body" for s in doc.segments)
    assert any(lk.target == "https://example.com/spec" and lk.link_type == "pdf_link" for lk in doc.links)
    assert doc.title  # title falls back to the first line when metadata has none


def test_extract_document_dispatch(tmp_path: Path) -> None:
    pdf = tmp_path / "a.pdf"
    _make_pdf(pdf, [["hello PDF world"]])
    html = tmp_path / "b.html"
    html.write_text("<body><h1>T</h1><p>hello HTML world</p></body>", encoding="utf-8")
    assert "hello PDF world" in " ".join(s.text for s in extract_document(pdf, "pdf").segments)
    assert "hello HTML world" in " ".join(s.text for s in extract_document(html, "html").segments)


def test_html_to_pdf_roundtrip_preserves_needles(tmp_path: Path) -> None:
    html = (
        "<body><h1>SPI</h1>"
        "<p>The SPI timing nonce ZQX-7734-FRB configures the peripheral bus.</p>"
        "<pre><code>def frame(addr, data): return (addr &lt;&lt; 8) | data  # CODE-NEEDLE-77</code></pre>"
        "</body>"
    )
    doc = extract_pdf(html_to_pdf_bytes(html))
    text = "\n".join(s.text for s in doc.segments)
    assert "ZQX-7734-FRB" in text  # prose needle survives HTML -> PDF -> extract
    assert "CODE-NEEDLE-77" in text  # code needle survives too


def _pdf_corpus_config(tmp_path: Path, pdf_dir: Path) -> config.Config:
    path = tmp_path / "pdf.yaml"
    path.write_text(
        f'paths:\n  staging_dir: "{(tmp_path / "s").as_posix()}"\n'
        f'  db_path: "{(tmp_path / "pdf.db").as_posix()}"\n'
        f'  tmp_dir: "{(tmp_path / "t").as_posix()}"\n'
        "sources:\n"
        f'  - name: pdf-corpus\n    location: "{pdf_dir.as_posix()}"\n'
        '    include: ["**/*.pdf"]\n    min_content_chars: 5\n'
        'embed:\n  model: "none"\n',
        encoding="utf-8",
    )
    return config.load(path)


def test_needle_survives_pdf_conversion_and_ingest(tmp_path: Path) -> None:
    # baseline HTML corpus with a needle -> convert to PDF -> ingest the PDFs -> search finds it
    src = tmp_path / "html"
    (src / "sub").mkdir(parents=True)
    (src / "spi.html").write_text(
        "<body><h1>SPI</h1><p>The nonce ZQX-7734-FRB configures the peripheral bus.</p></body>",
        encoding="utf-8",
    )
    (src / "sub" / "match.html").write_text(
        "<body><h1>Match</h1><p>A match loop uses the token MLP-4242-XY per site.</p></body>",
        encoding="utf-8",
    )
    pdf_dir = tmp_path / "pdf"
    result = convert_corpus(src, pdf_dir, fmt="pdf")
    assert result.converted == 2 and not result.errors
    assert (pdf_dir / "spi.pdf").is_file() and (pdf_dir / "sub" / "match.pdf").is_file()

    cfg = _pdf_corpus_config(tmp_path, pdf_dir)
    cat = Catalog(cfg)
    ingest_result = cat.ingest()
    assert ingest_result.documents == 2
    # needles survive the conversion and are findable via the real (sanitized) search path
    for needle, source in (("ZQX-7734-FRB", "spi.pdf"), ("MLP-4242-XY", "match.pdf")):
        hits = cat.search(needle)
        assert hits, f"needle {needle} not recovered from the PDF"
        assert any(source in h.path for h in hits)
    # PDFs carry page-based locators
    with Store.open(cfg.paths.db_path) as store:
        rows = store.chunks_for_document(1)
        assert any(r["locator"] == "page 1" for r in rows)
