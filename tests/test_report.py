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
    assert "1. http://host:8321/v1/documents/1?chunk=2" in out
    assert "2. http://host:8321/v1/documents/5?chunk=7" in out
    assert "3. " not in out.split("## References")[1]  # deduped -> only 2 refs


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


def test_deterministic() -> None:
    a = report.render_report(fmt="md", **_kwargs())  # type: ignore[arg-type]
    b = report.render_report(fmt="md", **_kwargs())  # type: ignore[arg-type]
    assert a == b


def test_images_embedded_as_links() -> None:
    out = report.render_report(fmt="md", **_kwargs(images=["abc123"]))  # type: ignore[arg-type]
    assert "http://host:8321/v1/images/abc123" in out
