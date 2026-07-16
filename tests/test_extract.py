"""HTML extraction + boilerplate stripping (R-ING-2, §7.3)."""

from __future__ import annotations

from docusearch import ingest

SAMPLE = """<html><head><title>Doc Title</title></head><body>
<nav>navigation junk here</nav>
<main class="article">
  <h1>Interfaces</h1>
  <h2>SPI</h2>
  <h3>Timing</h3>
  <p>Configure the <a href="clock.html">clock</a> before use of SPI.</p>
  <pre><code>def foo():
    return 1</code></pre>
  <table>
    <tr><th>Register</th><th>Value</th></tr>
    <tr><td>CTRL</td><td>0x1</td></tr>
  </table>
  <figure><img src="diagram.png" alt="the diagram"><figcaption>Figure 1</figcaption></figure>
  <script>var x = 1;</script>
</main>
<footer>footer junk</footer>
</body></html>"""


def _bodies(doc: ingest.ExtractedDoc) -> str:
    return "\n".join(s.text for s in doc.segments if s.kind == "body")


def test_title_from_title_tag() -> None:
    doc = ingest.extract_html(SAMPLE)
    assert doc.title == "Doc Title"


def test_title_falls_back_to_h1() -> None:
    doc = ingest.extract_html("<body><h1>Only Heading</h1><p>x</p></body>")
    assert doc.title == "Only Heading"


def test_heading_path_locator() -> None:
    doc = ingest.extract_html(SAMPLE, content_selector="main.article")
    para = next(s for s in doc.segments if s.kind == "body" and "Configure the clock" in s.text)
    assert para.heading_path == "Interfaces > SPI > Timing"


def test_code_block_preserved_whole_and_unsplit() -> None:
    doc = ingest.extract_html(SAMPLE, content_selector="main.article")
    code = [s for s in doc.segments if s.kind == "code"]
    assert len(code) == 1
    assert code[0].text == "def foo():\n    return 1"  # indentation + newline preserved


def test_table_linearized_with_pipes() -> None:
    doc = ingest.extract_html(SAMPLE, content_selector="main.article")
    table = next(s for s in doc.segments if s.kind == "table")
    assert "Register | Value" in table.text
    assert "CTRL | 0x1" in table.text


def test_links_captured_raw() -> None:
    doc = ingest.extract_html(SAMPLE, content_selector="main.article")
    targets = [link.target for link in doc.links]
    assert "clock.html" in targets
    assert all(link.link_type == "html_href" for link in doc.links)


def test_images_captured_with_alt_and_heading() -> None:
    doc = ingest.extract_html(SAMPLE, content_selector="main.article")
    assert len(doc.images) == 1
    img = doc.images[0]
    assert img.src == "diagram.png"
    assert img.alt == "the diagram"
    assert img.caption == "Figure 1"
    assert img.heading_path == "Interfaces > SPI > Timing"


def test_content_selector_scopes_extraction() -> None:
    doc = ingest.extract_html(SAMPLE, content_selector="main.article")
    assert doc.content_selector_matched is True
    text = _bodies(doc)
    assert "navigation junk" not in text
    assert "footer junk" not in text
    assert "Configure the clock" in text


def test_content_selector_missing_falls_back_to_body() -> None:
    doc = ingest.extract_html(SAMPLE, content_selector="div#nonexistent")
    assert doc.content_selector_matched is False
    # fell back to the whole body, so real content is still present
    assert "Configure the clock" in _bodies(doc)


def test_strip_selectors_remove_chrome() -> None:
    doc = ingest.extract_html(SAMPLE, strip_selectors=["nav", "footer", "script"])
    text = _bodies(doc)
    assert "navigation junk" not in text
    assert "footer junk" not in text


def test_script_and_style_ignored() -> None:
    doc = ingest.extract_html(SAMPLE, content_selector="main.article")
    assert "var x" not in _bodies(doc)


def test_text_length_reflects_visible_content() -> None:
    doc = ingest.extract_html(SAMPLE, content_selector="main.article")
    assert doc.text_length > 0
    empty = ingest.extract_html("<body><div></div></body>")
    assert empty.text_length == 0


def test_whitespace_collapsed_in_body() -> None:
    doc = ingest.extract_html("<body><p>a   lot\n\nof   space</p></body>")
    body = next(s for s in doc.segments if s.kind == "body")
    assert body.text == "a lot of space"
