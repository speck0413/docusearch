"""Phase 4a — PDF format layering: PyMuPDF extractor, format dispatch, and needles-through-
conversion (§7.3, §15.4). PDFs are generated with reportlab (a [dev] harness dep); extraction
uses PyMuPDF (the [pdf] runtime extra)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from docusearch import config
from docusearch.catalog import Catalog
from docusearch.convert import convert_corpus, html_to_pdf_bytes
from docusearch.ingest import extract_document, extract_pdf
from docusearch.store import Store

_ROOT = Path(__file__).resolve().parents[1]


def _load_compare():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(
        "harness_compare", _ROOT / "harness" / "compare.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _corpus_cfg(tmp_path: Path, name: str, location: Path, include: str) -> config.Config:
    path = tmp_path / f"{name}.yaml"
    path.write_text(
        f'paths:\n  staging_dir: "{(tmp_path / name / "s").as_posix()}"\n'
        f'  db_path: "{(tmp_path / name / "c.db").as_posix()}"\n'
        f'  tmp_dir: "{(tmp_path / name / "t").as_posix()}"\n'
        f'sources:\n  - name: {name}\n    location: "{location.as_posix()}"\n'
        f'    include: ["{include}"]\n    min_content_chars: 5\n'
        'embed:\n  model: "none"\n',
        encoding="utf-8",
    )
    return config.load(path)


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


def test_html_to_pdf_preserves_image_alt_needles(tmp_path: Path) -> None:
    # The needle suite hides 10 nonces in image alt/caption text. A faithful HTML->PDF render
    # of a (deliberately broken) <img> shows its alt text, so the nonce must survive the round
    # trip — otherwise those needles are lost in conversion (a §15.4 extractor/converter defect).
    html = (
        "<body><h1>Diagram</h1>"
        "<p>Ordinary body prose PROSE-1000-XX here.</p>"
        '<img src="missing-x.png" alt="calibration diagram IMG-5150-ALT for the gyroscope">'
        "</body>"
    )
    doc = extract_pdf(html_to_pdf_bytes(html))
    text = "\n".join(s.text for s in doc.segments)
    assert "PROSE-1000-XX" in text  # prose still survives
    assert "IMG-5150-ALT" in text  # and so does the image-alt needle


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


def _tiny_png() -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (123, 200, 50)).save(buf, format="PNG")
    return buf.getvalue()


def test_extract_pdf_retains_embedded_image(tmp_path: Path) -> None:
    # R-ING-6 for PDF: an image embedded in the PDF is retained with its bytes (for vision).
    import io

    from reportlab.lib.pagesizes import letter
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    p = tmp_path / "img.pdf"
    c = canvas.Canvas(str(p), pagesize=letter)
    c.drawString(72, 720, "a page with an embedded diagram")
    c.drawImage(ImageReader(io.BytesIO(_tiny_png())), 72, 600, width=64, height=64)
    c.showPage()
    c.save()
    doc = extract_pdf(p.read_bytes())
    assert doc.images, "embedded PDF image should be retained (R-ING-6)"
    assert doc.images[0].data and doc.images[0].heading_path == "page 1"


def test_pdf_embedded_image_retained_on_ingest(tmp_path: Path) -> None:
    import io

    from reportlab.lib.pagesizes import letter
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    pdf_dir = tmp_path / "pdf"
    pdf_dir.mkdir()
    c = canvas.Canvas(str(pdf_dir / "d.pdf"), pagesize=letter)
    c.drawString(72, 720, "SPI block diagram overview with enough text to index")
    c.drawImage(ImageReader(io.BytesIO(_tiny_png())), 72, 600, width=64, height=64)
    c.showPage()
    c.save()
    cfg = _pdf_corpus_config(tmp_path, pdf_dir)
    Catalog(cfg).ingest()
    with Store.open(cfg.paths.db_path) as store:
        assert store.count_images() >= 1  # the embedded image reached the images table


def test_convert_embeds_real_image_pdf(tmp_path: Path) -> None:
    # a real <img> file must be EMBEDDED into the converted PDF (not just its alt text), so the
    # image survives HTML->PDF and can be vision-enriched (R-ING-6, §15.4).
    (tmp_path / "diagram.png").write_bytes(_tiny_png())
    (tmp_path / "page.html").write_text(
        '<body><h1>D</h1><p>see the block diagram below</p>'
        '<img src="diagram.png" alt="SPI block diagram"></body>',
        encoding="utf-8",
    )
    dst = tmp_path / "pdf"
    assert convert_corpus(tmp_path, dst, fmt="pdf").converted == 1
    doc = extract_pdf((dst / "page.pdf").read_bytes())
    assert doc.images, "the real image should be embedded + retained through conversion"


def test_convert_embeds_tall_image_without_layout_error(tmp_path: Path) -> None:
    # a very tall diagram must be scaled to fit the page, not abort the whole PDF build
    # (reportlab LayoutError "Flowable too large") — the real cause of dropped ACME block diagrams.
    from PIL import Image

    Image.new("RGB", (300, 1000), (50, 120, 200)).save(tmp_path / "tall.png", format="PNG")
    (tmp_path / "page.html").write_text(
        '<body><h1>D</h1><p>see</p><img src="tall.png" alt="tall block diagram"></body>',
        encoding="utf-8",
    )
    dst = tmp_path / "pdf"
    result = convert_corpus(tmp_path, dst, fmt="pdf")
    assert result.converted == 1 and not result.errors  # no LayoutError, file converted
    assert extract_pdf((dst / "page.pdf").read_bytes()).images  # and the image is retained


def test_compare_formats_pdf_matches_html_baseline(tmp_path: Path) -> None:
    # §15.4 format-equivalence: the PDF suite store must agree with the HTML baseline store —
    # same logical doc retrieved for each query (overlap@10, top-1 logical match, MRR).
    cmp = _load_compare()
    src = tmp_path / "html" / "library"
    src.mkdir(parents=True)
    docs = {
        "spi.html": "<body><h1>SPI</h1><p>The PA nWire SPI peripheral bus drives the MOSI "
        "and SCLK signals to the DUT.</p></body>",
        "match.html": "<body><h1>Match</h1><p>A match loop compares captured values per site "
        "during a functional test pattern.</p></body>",
        "timing.html": "<body><h1>Timing</h1><p>Clock edge timing with setup and hold windows "
        "around the strobe reference.</p></body>",
    }
    for name, html in docs.items():
        (src / name).write_text(html, encoding="utf-8")
    html_root, pdf_root = tmp_path / "html", tmp_path / "pdf"
    assert convert_corpus(html_root, pdf_root, fmt="pdf").converted == 3

    base_cfg = _corpus_cfg(tmp_path, "base", html_root, "**/*.html")
    suite_cfg = _corpus_cfg(tmp_path, "suite", pdf_root, "**/*.pdf")
    Catalog(base_cfg).ingest()
    Catalog(suite_cfg).ingest()

    queries = [
        "PA nWire SPI MOSI SCLK bus signals",
        "match loop compare values per site",
        "clock edge setup hold strobe timing",
    ]
    with Store.open(base_cfg.paths.db_path) as bstore, Store.open(suite_cfg.paths.db_path) as sstore:
        comps = cmp.compare_formats(
            queries,
            baseline_search=cmp.make_searcher(bstore),
            suite_search=cmp.make_searcher(sstore),
            baseline_root=html_root,
            suite_root=pdf_root,
        )
    summary = cmp.summarize(comps)
    assert summary.mean_overlap_at_k >= 0.7  # §15.4 default gate
    assert summary.top1_match_rate >= 0.8  # §15.4 default gate
    assert summary.mrr >= 0.8
    # logical keys map across formats (library/spi in both stores)
    assert comps[0].suite[0] == "library/spi"
    text = cmp.render_format_compare(comps, summary)
    assert "overlap@10" in text and "top-1" in text and "MRR" in text


def test_logical_key_unmapped_paths_do_not_collide() -> None:
    # red-team M1: when a stored path doesn't resolve under the given root, the fallback must
    # NOT collapse unrelated documents to the same bare-stem key (12 index.html would merge).
    cmp = _load_compare()
    root = Path("/corpus/html")
    a = cmp._logical_key("/somewhere/else/library/index.html", root)
    b = cmp._logical_key("/completely/different/tutorial/index.html", root)
    assert a != b  # distinct documents -> distinct keys, even on the unmapped fallback


def test_convert_source_honors_selector_and_excludes(tmp_path: Path) -> None:
    # Altering a real ingestion for the PDF format: the derived PDF must carry the CLEAN
    # content_selector text (not chrome), and honor the source's include/exclude globs.
    from docusearch.convert import convert_source

    src = tmp_path / "site"
    (src / "keep").mkdir(parents=True)
    (src / "skip").mkdir(parents=True)
    (src / "keep" / "page.html").write_text(
        "<html><body><nav>BREADCRUMB-CHROME</nav>"
        "<article><h1>Real</h1><p>ARTICLE-NONCE-4242 the calibration procedure.</p></article>"
        "<footer>FOOTER-CHROME</footer></body></html>",
        encoding="utf-8",
    )
    (src / "skip" / "junk.html").write_text("<article><p>EXCLUDED-9999</p></article>", "utf-8")

    source = config.SourceConfig(
        type="fs", name="s", version="", location=str(src),
        include=["keep/**/*.html"], exclude=["skip/**"],
        content_selector="article", strip_selectors=["nav", "footer"],
        min_content_chars=5, audience=[],
    )
    dst = tmp_path / "pdf"
    result = convert_source(source, dst, fmt="pdf")
    assert result.converted == 1 and not result.errors  # only the included file
    assert (dst / "keep" / "page.pdf").is_file()
    assert not (dst / "skip" / "junk.pdf").exists()  # exclude honored

    doc = extract_pdf((dst / "keep" / "page.pdf").read_bytes())
    text = "\n".join(s.text for s in doc.segments)
    assert "ARTICLE-NONCE-4242" in text  # clean article content survives
    assert "CHROME" not in text  # nav/footer chrome was stripped, not baked into the PDF


def test_summarize_empty_is_not_a_pass() -> None:
    # red-team M2: zero comparisons must never report PASS (an empty --queries file verifies
    # nothing, so compare.py must not exit 0 / print PASS).
    cmp = _load_compare()
    summary = cmp.summarize([])
    assert summary.queries == 0
    assert summary.passes is False
