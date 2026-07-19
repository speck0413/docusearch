"""Scripted report generation with citation rendering (R-CIT-2, §12)."""

from __future__ import annotations

import pytest

from docusearch import citations, report


def _kwargs(**over: object) -> dict[str, object]:
    base: dict[str, object] = {
        "title": "SPI timing summary",
        "body": "SPI runs at 1 MHz [D:1#2]. The sky is blue [GK]. It uses a clock [D:1#3].",
        "evidence": {(1, 2), (1, 3)},
        "base_url": "http://host:8321",
        "run_id": "RUN-XYZ",
        "audience": ["engineering"],
        "embed_model": "bge-large",
        "sources": ["vendor-html"],
    }
    base.update(over)
    return base


def test_markdown_has_banner_body_and_references() -> None:
    out = report.render_report(fmt="md", **_kwargs())  # type: ignore[arg-type]
    assert "# SPI timing summary" in out
    assert "RUN-XYZ" in out and "bge-large" in out and "engineering" in out
    assert "## References" in out
    assert "http://host:8321/v1/documents/1?chunk=2" in out
    assert "[GK]" in out  # general-knowledge tag left inline
    assert "[D:1#2]" not in out  # replaced by a numbered marker


def test_references_are_numbered_and_deduped() -> None:
    body = "A [D:1#2]. B [D:1#2]. C [D:5#7]."
    out = report.render_report(fmt="md", **_kwargs(body=body, evidence={(1, 2), (5, 7)}))  # type: ignore[arg-type]
    refs = out.split("## References")[1]
    assert "1. [" in refs and "http://host:8321/v1/documents/1?chunk=2" in refs
    assert "http://host:8321/v1/documents/5?chunk=7" in refs
    assert "3. " not in refs  # deduped -> only 2 refs


def test_refuses_citation_outside_evidence() -> None:
    # (9, 999) is not in the evidence set -> hallucinated ref, refuse to render (R-CIT-1)
    with pytest.raises(citations.CitationError):
        report.render_report(fmt="md", **_kwargs(body="X [D:9#999].", evidence={(1, 2)}))  # type: ignore[arg-type]


def test_refuses_citation_hidden_in_banner_metadata() -> None:
    # red-team R1: a citation-shaped token in sources/audience must not slip through
    with pytest.raises(citations.CitationError):
        report.render_report(
            fmt="md",
            **_kwargs(body="ok [GK].", sources=["manual [D:9#999]"], evidence={(1, 2)}),  # type: ignore[arg-type]
        )


def test_refuses_citation_hidden_in_the_title() -> None:
    # red-team H1: a fabricated citation in the TITLE must be refused, not rendered verbatim
    with pytest.raises(citations.CitationError):
        report.render_report(
            fmt="md",
            **_kwargs(title="Sneaky [D:9#999]", body="ok [GK].", evidence={(1, 2)}),  # type: ignore[arg-type]
        )


def test_html_output_is_valid_ish() -> None:
    out = report.render_report(fmt="html", **_kwargs())  # type: ignore[arg-type]
    assert "<html" in out.lower() and "</html>" in out.lower()
    assert "SPI timing summary" in out
    assert 'href="http://host:8321/v1/documents/1?chunk=2"' in out


def test_html_sections_render_as_cards_with_inline_citations() -> None:
    sections = [
        {"heading": "Overview", "kind": "overview", "body": "PA drives the bus [D:101#2]."},
        {
            "heading": "Example",
            "kind": "code",
            "body": "Run it [D:318#2]:\n```\npa.frame('WRITE')\n```\n",
        },
    ]
    out = report.render_report(
        fmt="html",
        title="Protocol Aware",
        sections=sections,  # type: ignore[arg-type]
        evidence={(101, 2), (318, 2)},
        base_url="http://host:8321",
    )
    assert 'class="card kind-overview"' in out and 'class="card kind-code"' in out
    assert "<h2>Overview</h2>" in out and "<h2>Example</h2>" in out
    # inline citation -> superscript link to the numbered reference anchor
    assert '<sup class="cite"><a href="#ref-1">1</a></sup>' in out
    assert 'id="ref-1"' in out and 'id="ref-2"' in out
    assert '<pre class="code">' in out  # fenced code became a code block
    assert "midnight" not in out.lower()  # (theme is via CSS vars, not literal text)
    assert "#0a1730" in out  # the midnight-blue background is present in the inlined CSS


def test_header_provenance_and_ai_warning() -> None:
    out = report.render_report(
        fmt="html",
        request="how do I control PA",
        requested_by="Stephen Peck",
        model="claude-haiku-4-5",
        classification="Internal — Example Corp",
        **_kwargs(),  # type: ignore[arg-type]
    )
    assert "Internal — Example Corp" in out  # ribbon renders whatever classification is passed
    assert "AI-generated" in out and "double-checked" in out  # the warning
    assert "how do I control PA" in out  # the exact request
    assert "Stephen Peck" in out and "claude-haiku-4-5" in out  # who/what generated it
    assert "vendor-html" in out  # document store name in the banner


def test_references_link_to_original_documents() -> None:
    # ref_targets overrides the chunk URL with a file link labelled store — title — heading
    targets = {
        (1, 2): ("file:///docs/acme/spi.html", "FOR_DEBUG — SPI Overview — Timing"),
        (1, 3): ("file:///docs/acme/spi.html", "FOR_DEBUG — SPI Overview"),
    }
    out = report.render_report(
        fmt="html",
        ref_targets=targets,
        **_kwargs(),  # type: ignore[arg-type]
    )
    assert 'href="file:///docs/acme/spi.html"' in out
    assert "FOR_DEBUG — SPI Overview — Timing" in out
    assert "v1/documents/1?chunk=2" not in out  # the chunk URL is gone


def test_generation_log_is_collapsible_and_not_citation_verified() -> None:
    trace = {
        "prompt": "effort: high",
        "queries": ["controlling PA", "PA engines"],
        "retrieved": ["[D:9#999] some doc — snippet"],  # a chunk NOT in evidence is fine here
        "reasoning": "grouped into cards by subsystem",
    }
    out = report.render_report(fmt="html", trace=trace, **_kwargs())  # type: ignore[arg-type]
    assert "<details" in out and "Generation log" in out  # collapsible section present
    assert "<details" in out and " open" not in out.split("Generation log")[0][-40:]  # collapsed
    assert "controlling PA" in out and "grouped into cards" in out
    # the trace's [D:9#999] must NOT be treated as a citation (it's a log, not a claim)
    assert "some doc" in out


def test_sections_citation_outside_evidence_is_refused() -> None:
    with pytest.raises(citations.CitationError):
        report.render_report(
            fmt="html",
            title="x",
            sections=[{"heading": "H", "kind": "code", "body": "bad [D:9#999]"}],  # type: ignore[arg-type]
            evidence={(1, 2)},
            base_url="http://h",
        )


def test_deterministic() -> None:
    a = report.render_report(fmt="md", **_kwargs())  # type: ignore[arg-type]
    b = report.render_report(fmt="md", **_kwargs())  # type: ignore[arg-type]
    assert a == b


def test_images_embedded_as_links() -> None:
    out = report.render_report(fmt="md", **_kwargs(images=["abc123"]))  # type: ignore[arg-type]
    assert "http://host:8321/v1/images/abc123" in out


def test_embedded_images_render_as_data_uri_html_and_md() -> None:
    # a cited diagram is embedded inline (base64 data URI) so the figure is self-contained.
    data_uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="
    imgs = [(data_uri, "UltraVS256-HP block diagram")]
    html = report.render_report(fmt="html", embedded_images=imgs, **_kwargs())  # type: ignore[arg-type]
    assert data_uri in html and "UltraVS256-HP block diagram" in html and "<figcaption>" in html
    md = report.render_report(fmt="md", embedded_images=imgs, **_kwargs())  # type: ignore[arg-type]
    assert f"![UltraVS256-HP block diagram]({data_uri})" in md


def test_evidence_images_embeds_cited_diagram(tmp_path):  # type: ignore[no-untyped-def]
    # end-to-end: a cited enrichment chunk -> its retained image -> a base64 data URI.
    import hashlib
    import io

    from PIL import Image

    from docusearch.cli import _evidence_images
    from docusearch.store import Store

    png = io.BytesIO()
    Image.new("RGB", (8, 8), (200, 50, 50)).save(png, format="PNG")
    data = png.getvalue()
    sha = hashlib.sha256(data).hexdigest()
    staging = tmp_path / "staging"
    (staging / "images").mkdir(parents=True)
    (staging / "images" / f"{sha}.png").write_bytes(data)
    db = tmp_path / "c.db"
    with Store.open(db) as store:
        doc = store.add_document(path="/a.pdf")
        store.add_image(sha256=sha, ext="png", doc_id=doc, locator="page 1",
                        alt="block diagram", caption="", num_bytes=len(data))
        cid = store.add_enrichment_chunk(doc, "Block diagram: a DAC drives the output.", "page 1")

    imgs = _evidence_images(str(db), str(staging), {(doc, cid)})
    assert imgs, "cited image chunk should yield an embedded figure"
    assert imgs[0][0].startswith("data:image/png;base64,")
    assert "block diagram" in imgs[0][1]
    # a non-image (body) chunk yields nothing
    assert _evidence_images(str(db), str(staging), {(doc, 999)}) == []
