"""Phase 4c — Markdown format layering: markdown-it-py extractor, format dispatch, and needles-
through-conversion (§7.3, §15.4)."""

from __future__ import annotations

from pathlib import Path

from docusearch.convert import convert_corpus, html_to_md_bytes
from docusearch.ingest import extract_document, extract_md

_MD = """---
title: SPI Doc
audience: engineering
---

# SPI Protocol Overview

The nonce ZQX-7734-FRB configures the peripheral bus. See [the spec](https://ex.com/spec).

## Timing

```python
CONST = "CODE-2000-YY"  # timing
```

| Marker | Note |
|--------|------|
| MLP-4242-XY | per site |

![calibration diagram IMG-4000-WW](diagram.png)
"""


def test_extract_md_headings_code_table_links_images() -> None:
    doc = extract_md(_MD.encode("utf-8"))
    text = "\n".join(s.text for s in doc.segments)
    assert "ZQX-7734-FRB" in text  # prose (front matter stripped, not indexed as body)
    assert "front matter" not in text.lower()
    # heading paths carry structure
    hpaths = {s.heading_path for s in doc.segments}
    assert any("SPI Protocol Overview" in h for h in hpaths)
    assert any("Timing" in h for h in hpaths)
    # fenced code preserved as its own kind, needle intact
    code = [s for s in doc.segments if s.kind == "code"]
    assert code and "CODE-2000-YY" in code[0].text
    # GFM table linearized, needle intact
    tbl = [s for s in doc.segments if s.kind == "table"]
    assert tbl and "MLP-4242-XY" in tbl[0].text and "|" in tbl[0].text
    # link + image captured
    assert any(lk.target == "https://ex.com/spec" and lk.link_type == "md_link" for lk in doc.links)
    assert any(im.src == "diagram.png" and "IMG-4000-WW" in im.alt for im in doc.images)


def test_extract_document_dispatch_md(tmp_path: Path) -> None:
    for ext in ("md", "markdown"):
        p = tmp_path / f"a.{ext}"
        p.write_text("# T\n\nhello MD world ZQX-7734-FRB here.\n", encoding="utf-8")
        segs = extract_document(p, ext).segments
        assert "ZQX-7734-FRB" in " ".join(s.text for s in segs)


def test_html_to_md_roundtrip_preserves_needles_all_placements() -> None:
    html = (
        "<body><h1>SPI</h1>"
        "<p>Prose needle PRO-1000-XX configures the bus.</p>"
        "<pre><code>CONST = 'CODE-2000-YY'  # timing</code></pre>"
        "<table><tr><td>TAB-3000-ZZ</td><td>per site</td></tr></table>"
        '<img src="missing.png" alt="diagram IMG-4000-WW for the gyroscope">'
        "</body>"
    )
    doc = extract_md(html_to_md_bytes(html))
    text = "\n".join(s.text for s in doc.segments)
    for nonce in ("PRO-1000-XX", "CODE-2000-YY", "TAB-3000-ZZ", "IMG-4000-WW"):
        assert nonce in text, f"{nonce} lost in HTML->MD->extract"


def test_html_to_md_escapes_underscore_identifiers() -> None:
    # red-team H1: underscore-wrapped identifiers must NOT be eaten as markdown emphasis.
    html = "<body><h1>API</h1><p>Override the __init__ and _missing_ and __dunder__ methods.</p></body>"
    doc = extract_md(html_to_md_bytes(html))
    text = "\n".join(s.text for s in doc.segments)
    for ident in ("__init__", "_missing_", "__dunder__"):
        assert ident in text, f"{ident} was lost to markdown emphasis"


def test_html_to_md_multi_image_locators_distinct(tmp_path: Path) -> None:
    # red-team M2: two images under different headings keep their own heading-path locators
    # (not both collapsed onto the last heading).
    from PIL import Image

    Image.new("RGB", (8, 8), (1, 2, 3)).save(tmp_path / "a.png", format="PNG")
    Image.new("RGB", (8, 8), (9, 9, 9)).save(tmp_path / "b.png", format="PNG")
    (tmp_path / "page.html").write_text(
        "<body><h1>Doc</h1>"
        '<h2>Alpha</h2><p>text</p><img src="a.png" alt="alpha diagram">'
        '<h2>Beta</h2><p>text</p><img src="b.png" alt="beta diagram">'
        "</body>",
        encoding="utf-8",
    )
    doc = extract_md(html_to_md_bytes((tmp_path / "page.html").read_text(), base_path=tmp_path / "page.html"))
    img_headings = {im.heading_path for im in doc.images}
    assert any("Alpha" in h for h in img_headings)
    assert any("Beta" in h for h in img_headings)  # not both under the last heading


def test_convert_corpus_md_embeds_real_image_retainable(tmp_path: Path) -> None:
    from PIL import Image

    Image.new("RGB", (8, 8), (10, 20, 200)).save(tmp_path / "diagram.png", format="PNG")
    (tmp_path / "page.html").write_text(
        '<body><h1>D</h1><p>see</p><img src="diagram.png" alt="block diagram"></body>',
        encoding="utf-8",
    )
    dst = tmp_path / "md"
    result = convert_corpus(tmp_path, dst, fmt="md")
    assert result.converted == 1 and not result.errors
    assert (dst / "page.md").is_file()
    # the real image was embedded as a data URI and is recovered with inline bytes (retainable)
    doc = extract_md((dst / "page.md").read_bytes())
    assert doc.images and doc.images[0].data is not None and doc.images[0].ext == "png"
