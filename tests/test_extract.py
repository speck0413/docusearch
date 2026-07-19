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


def test_semantic_nav_excluded_from_body_and_relations_without_selector() -> None:
    # Stephen 2026-07-18: a cross-link counts only if it's in the BODY text we index. Semantic
    # navigation chrome (nav/aside/footer) must be excluded from BOTH indexed text and relations
    # even with NO content_selector — a nav-pane link must not become a relation.
    html = (
        "<body>"
        '<nav><a href="navlink.html">nav to somewhere</a> navigation junk</nav>'
        '<aside><a href="asidelink.html">aside link</a> sidebar junk</aside>'
        '<main><h1>Doc</h1><p>Real content with a <a href="bodylink.html">body link</a>.</p></main>'
        '<footer><a href="footlink.html">footer link</a> footer junk</footer>'
        "</body>"
    )
    doc = ingest.extract_html(html)  # no content_selector
    text = _bodies(doc)
    assert "Real content" in text
    assert "navigation junk" not in text and "sidebar junk" not in text and "footer junk" not in text
    targets = [link.target for link in doc.links]
    assert targets == ["bodylink.html"]  # ONLY the body link is a relation


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


# --- regression tests for the red-team Finding 1 (content loss/gluing) ---


def _all_body(doc: ingest.ExtractedDoc) -> str:
    return " ".join(s.text for s in doc.segments if s.kind == "body")


def test_inline_synopsis_div_text_is_captured() -> None:
    # PHP method-synopsis markup: a <div> whose children are only inline <span>/<strong>.
    html = (
        "<body><h1>Method</h1>"
        '<div class="methodsynopsis dc-description">'
        '<span class="modifier">public</span> <span class="modifier">function</span> '
        '<span class="methodname"><strong>Gmagick::getimagegamma</strong></span>(): '
        '<span class="type"><a href="float.html">float</a></span></div></body>'
    )
    text = _all_body(ingest.extract_html(html))
    assert "public" in text
    assert "function" in text
    assert "Gmagick::getimagegamma" in text
    assert "float" in text


def test_no_gluing_across_element_boundaries() -> None:
    # Nested TOC list: an <a> text immediately followed by a nested <ul> must not fuse.
    html = (
        "<body><ul><li><a href='ref.html'>GnuPG Functions</a>"
        "<ul><li><a href='a.html'>gnupg_adddecryptkey</a> — Add a key for decryption</li>"
        "<li><a href='b.html'>gnupg_addencryptkey</a> — Add a key for encryption</li>"
        "</ul></li></ul></body>"
    )
    text = _all_body(ingest.extract_html(html))
    assert "Functionsgnupg" not in text  # not glued
    assert "decryptiongnupg" not in text  # not glued
    assert "gnupg_adddecryptkey" in text
    assert "gnupg_addencryptkey" in text


def test_mixed_inline_and_block_direct_text_captured() -> None:
    html = "<body><div>intro words here <p>a paragraph</p> trailing words here</div></body>"
    text = _all_body(ingest.extract_html(html))
    assert "intro words here" in text
    assert "a paragraph" in text
    assert "trailing words here" in text


def test_bare_div_and_inline_code_captured() -> None:
    html = (
        "<body><div>just some bare text</div>"
        '<div class="classsynopsisinfo"><code>readonly</code> <code>public</code> '
        "int <var>Foo::endcolumn</var></div></body>"
    )
    text = _all_body(ingest.extract_html(html))
    assert "just some bare text" in text
    assert "readonly" in text
    assert "Foo::endcolumn" in text
