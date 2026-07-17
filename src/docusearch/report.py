"""Scripted report generation (R-CIT-2, §12).

Reports are produced by code, never freehand by a model, so the output is consistent:
a banner (generated-at, run id, audience, embed model, sources), the claim body with
citation tags rendered as numbered markers, a References section of clickable server
URLs, and any embedded images. The builder **verifies** every ``[D:]`` tag against the
evidence set and refuses to render if a citation points outside it (R-CIT-1) — this is
what prevents hallucinated references.

Public surface:
    render_report(*, title, body, evidence_chunk_ids, base_url, fmt="md", ...) -> str
"""

from __future__ import annotations

import html as _html
from collections.abc import Sequence
from datetime import UTC, datetime

from . import citations
from ._version import __version__


def _banner_lines(
    *,
    title: str,
    run_id: str,
    generated_at: str,
    audience: Sequence[str],
    embed_model: str,
    sources: Sequence[str],
) -> list[str]:
    return [
        f"generated: {generated_at}",
        f"run_id: {run_id}",
        f"docusearch: v{__version__}",
        f"audience: {', '.join(audience) or '—'}",
        f"embed_model: {embed_model}",
        f"sources: {', '.join(sources) or '—'}",
    ]


def render_report(
    *,
    title: str,
    body: str,
    evidence: set[tuple[int, int]],
    base_url: str,
    fmt: str = "md",
    run_id: str = "",
    generated_at: str | None = None,
    audience: Sequence[str] = (),
    embed_model: str = "none",
    sources: Sequence[str] = (),
    images: Sequence[str] = (),
) -> str:
    """Render a report to ``md`` or ``html``. Raises ``CitationError`` if the title OR the
    body cites a ``(doc_id, chunk_id)`` outside the evidence set (R-CIT-1).

    ``evidence`` is the set of ``(doc_id, chunk_id)`` pairs from the search hits the report
    is built on.
    """
    # verify the WHOLE rendered surface — title, body, and banner metadata — so a
    # citation-shaped token can't hide anywhere (red-team H1 + R1)
    surface = "\n".join([title, body, *audience, *sources])
    violations = citations.verify(surface, evidence)
    if violations:
        bad = ", ".join(v.raw for v in violations)
        raise citations.CitationError(
            f"report cites sources outside the evidence set: {bad}. Refusing to render."
        )

    stamp = generated_at or datetime.now(UTC).isoformat(timespec="seconds")
    banner = _banner_lines(
        title=title,
        run_id=run_id,
        generated_at=stamp,
        audience=audience,
        embed_model=embed_model,
        sources=sources,
    )
    # number title + body citations together, then split back so numbering is consistent
    rendered_combined, references = citations.render_references(f"{title}\n{body}", base_url)
    title, _, rendered_body = rendered_combined.partition("\n")
    image_urls = [f"{base_url.rstrip('/')}/v1/images/{sha}" for sha in images]

    if fmt == "html":
        return _render_html(title, banner, rendered_body, references, image_urls)
    return _render_markdown(title, banner, rendered_body, references, image_urls)


def _render_markdown(
    title: str,
    banner: list[str],
    body: str,
    references: list[str],
    image_urls: list[str],
) -> str:
    out = [f"# {title}", "", "> " + "  ·  ".join(banner), "", body, ""]
    if image_urls:
        out += ["## Images", *[f"![image]({url})" for url in image_urls], ""]
    out += ["## References", *(references or ["(none)"]), ""]
    return "\n".join(out)


def _render_html(
    title: str,
    banner: list[str],
    body: str,
    references: list[str],
    image_urls: list[str],
) -> str:
    esc = _html.escape
    # numbered reference markers [n] become superscripts; URLs become links
    refs_html = "".join(f"<li>{_link(ref)}</li>" for ref in references) or "<li>(none)</li>"
    imgs_html = "".join(f'<img src="{esc(url)}" alt="image">' for url in image_urls)
    body_html = esc(body)
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{esc(title)}</title></head><body>"
        f"<h1>{esc(title)}</h1>"
        f"<blockquote>{' &middot; '.join(esc(b) for b in banner)}</blockquote>"
        f"<div class='report-body'><p>{body_html}</p></div>"
        + (f"<h2>Images</h2>{imgs_html}" if image_urls else "")
        + f"<h2>References</h2><ol>{refs_html}</ol>"
        + "</body></html>"
    )


def _link(reference: str) -> str:
    """Turn a "<n>. <url>" reference into an HTML link."""
    number, _, url = reference.partition(". ")
    url = url.strip()
    if url.startswith("http"):
        return f'{number}. <a href="{_html.escape(url)}">{_html.escape(url)}</a>'
    return _html.escape(reference)
