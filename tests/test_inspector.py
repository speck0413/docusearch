"""Document-shape inspector: suggest content_selector / strip_selectors from samples."""

from __future__ import annotations

from docusearch import inspector

_CHROME = '<header id="top">Home</header><nav class="wh_breadcrumb">Home &gt; X</nav>'
_CONTENT = (
    '<article role="article"><h1>Title</h1>'
    "<p>Real article content here with plenty of words so coverage is meaningful.</p></article>"
)
_FOOTER = "<footer>copyright</footer><script>hljs();</script>"


def _doc(body: str) -> str:
    return f"<html><head><title>T</title></head><body>{body}</body></html>"


def test_suggests_body_container_and_strips_chrome() -> None:
    docs = [_doc(_CHROME + _CONTENT + _FOOTER) for _ in range(10)]
    res = inspector.inspect_html(docs)
    assert res.sampled == 10
    assert res.content_selector == "article"  # the body container, not the whole page
    assert {"script", "nav", "header", "footer"} <= set(res.strip_selectors)  # semantic chrome
    assert any("breadcrumb" in s for s in res.strip_selectors)  # keyword-detected chrome
    # the suggested candidate leads the ranked list
    assert res.content_candidates[0][0] == "article"


def test_empty_input_is_safe() -> None:
    res = inspector.inspect_html([])
    assert res.sampled == 0
    assert res.content_selector == "" and res.strip_selectors == []


def test_no_common_container_leaves_selector_empty() -> None:
    # plain pages with no recognizable body wrapper -> don't force a content_selector
    docs = [_doc("<p>just a bare paragraph of text with several words</p>") for _ in range(5)]
    res = inspector.inspect_html(docs)
    assert res.content_selector == ""
