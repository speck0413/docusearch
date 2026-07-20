"""Scripted report generation (R-CIT-2, §12).

Reports are produced by code, never freehand by a model, so the output is consistent and
citations are trustworthy. A report is a **banner** + a set of **section cards** (overview,
code, hardware, test-program, procedure, … — whatever the lookup surfaced) + a **references**
list. Citations appear inline inside each card as little superscript number links that jump
to the reference; the reference list resolves each to a clickable server URL.

The builder **verifies** every ``[D:]`` tag against the evidence set and refuses to render if
a citation points outside it (R-CIT-1) — this is what prevents hallucinated references.

Public surface:
    render_report(*, title, sections|body, evidence, base_url, fmt="md", ...) -> str
"""

from __future__ import annotations

import html as _html
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

from . import citations
from ._version import __version__

# Same grammar as citations._CITATION, kept local for inline rendering.
_CITE_RE = re.compile(r"\[GK\]|\[D:(\d+)#(\d+)\]")

# section kind -> (emoji, css class). Unknown kinds fall back to a generic card.
_KINDS: dict[str, tuple[str, str]] = {
    "overview": ("📋", "kind-overview"),
    "summary": ("📋", "kind-overview"),
    "procedure": ("🪜", "kind-procedure"),
    "steps": ("🪜", "kind-procedure"),
    "code": ("💻", "kind-code"),
    "example": ("💻", "kind-code"),
    "hardware": ("🔧", "kind-hardware"),
    "config": ("⚙️", "kind-config"),
    "settings": ("⚙️", "kind-config"),
    "test": ("🧪", "kind-test"),
    "test-program": ("🧪", "kind-test"),
    "warning": ("⚠️", "kind-warning"),
    "gotcha": ("⚠️", "kind-warning"),
    "reference": ("📎", "kind-reference"),
}


def _norm_sections(
    sections: Sequence[Mapping[str, str]] | None, body: str
) -> list[tuple[str, str, str]]:
    """Return ``[(heading, kind, body)]``. Falls back to one 'overview' card from ``body``."""
    if sections:
        return [
            (str(s.get("heading", "")), str(s.get("kind", "overview")), str(s.get("body", "")))
            for s in sections
        ]
    return [("", "overview", body)]


_AI_WARNING = (
    "AI-generated — every claim must be double-checked against the cited source "
    "documents before it is relied upon."
)


def render_report(
    *,
    title: str,
    body: str = "",
    sections: Sequence[Mapping[str, str]] | None = None,
    subtitle: str = "",
    evidence: set[tuple[int, int]],
    base_url: str,
    fmt: str = "md",
    run_id: str = "",
    generated_at: str | None = None,
    audience: Sequence[str] = (),
    embed_model: str = "none",
    sources: Sequence[str] = (),
    images: Sequence[str] = (),
    embedded_images: Sequence[tuple[str, str]] = (),
    request: str = "",
    requested_by: str = "",
    model: str = "",
    classification: str = "Confidential",
    ref_targets: Mapping[tuple[int, int], tuple[str, str]] | None = None,
    trace: Mapping[str, object] | None = None,
) -> str:
    """Render a report to ``md`` or ``html``. Raises ``CitationError`` if any citation (in
    the title, subtitle, or any section) references a ``(doc_id, chunk_id)`` outside the
    evidence set (R-CIT-1). Pass ``sections`` (a list of ``{heading, kind, body}``) for the
    card layout, or the legacy single ``body``.

    Header provenance: ``classification`` (confidentiality banner), ``request`` (the exact
    prompt this answers), ``requested_by`` (the user it's for), ``model`` (which model
    generated it), plus the generated date and the document stores (``sources``).
    """
    secs = _norm_sections(sections, body)
    # Verify the WHOLE surface — nothing citation-shaped may hide anywhere (red-team H1/R1).
    surface = "\n".join(
        [
            title,
            subtitle,
            request,
            requested_by,
            model,
            classification,
            *[h for h, _k, _b in secs],
            *[b for _h, _k, b in secs],
            *audience,
            *sources,
        ]
    )
    violations = citations.verify(surface, evidence)
    if violations:
        bad = ", ".join(v.raw for v in violations)
        raise citations.CitationError(
            f"report cites sources outside the evidence set: {bad}. Refusing to render."
        )

    stamp = generated_at or datetime.now(UTC).isoformat(timespec="seconds")
    # Number citations across the whole report, in order of first appearance.
    numbering, ordered = _collect_numbering([title, subtitle, *[b for _h, _k, b in secs]])
    image_urls = [f"{base_url.rstrip('/')}/v1/images/{sha}" for sha in images]
    meta = _meta(stamp, requested_by, model, embed_model, audience, sources, run_id)
    header = (classification, request)
    refs = _references(ordered, base_url, ref_targets)

    if fmt == "html-slide":
        return _render_slides(
            title, subtitle, header, meta, secs, numbering, refs, image_urls, embedded_images, trace
        )
    if fmt == "html":
        return _render_html(
            title, subtitle, header, meta, secs, numbering, refs, image_urls, embedded_images, trace
        )
    return _render_markdown(
        title, subtitle, header, meta, secs, numbering, refs, image_urls, embedded_images, trace
    )


def _trace_html(trace: Mapping[str, object] | None) -> str:
    """A collapsed <details> log of how the report was produced — prompt, searches,
    retrieved chunks, and the model's reasoning. Escaped (not citation-linked): it's a log,
    not claims, so it is not citation-verified."""
    if not trace:
        return ""
    esc = _html.escape
    parts: list[str] = []

    def _block(title: str, value: object) -> None:
        if value in (None, "", [], {}):
            return
        if isinstance(value, (list, tuple)):
            items = "".join(f"<li>{esc(str(v))}</li>" for v in value)
            parts.append(f"<h3>{esc(title)}</h3><ol class='trace-list'>{items}</ol>")
        else:
            parts.append(f"<h3>{esc(title)}</h3><pre class='trace-pre'>{esc(str(value))}</pre>")

    _block("Prompt / effort", trace.get("prompt"))
    _block("Searches run", trace.get("queries"))
    _block("Retrieved (candidate evidence)", trace.get("retrieved"))
    _block("Model reasoning", trace.get("reasoning"))
    if not parts:
        return ""
    return (
        '<details class="trace"><summary>🔍 Generation log — how this report was produced'
        f'</summary><div class="trace-body">{"".join(parts)}</div></details>'
    )


def _trace_md(trace: Mapping[str, object] | None) -> list[str]:
    if not trace:
        return []
    out = ["<details>", "<summary>🔍 Generation log — how this report was produced</summary>", ""]
    for title, key in (
        ("Prompt / effort", "prompt"),
        ("Searches run", "queries"),
        ("Retrieved (candidate evidence)", "retrieved"),
        ("Model reasoning", "reasoning"),
    ):
        value = trace.get(key)
        if not value:
            continue
        out.append(f"**{title}**")
        out.append("")
        if isinstance(value, (list, tuple)):
            out += [f"- {v}" for v in value]
        else:
            out.append(str(value))
        out.append("")
    out += ["</details>", ""]
    return out


def _references(
    ordered: list[citations.Citation],
    base_url: str,
    ref_targets: Mapping[tuple[int, int], tuple[str, str]] | None,
) -> list[tuple[str, str]]:
    """``[(href, label)]`` per reference. Prefer the original vendor document (store — title
    — locator, linked to the source file) when ``ref_targets`` provides it; otherwise fall
    back to the server chunk URL."""
    out: list[tuple[str, str]] = []
    for c in ordered:
        assert c.doc_id is not None and c.chunk_id is not None
        target = (ref_targets or {}).get((c.doc_id, c.chunk_id))
        if target is not None:
            out.append(target)
        else:
            label = f"document {c.doc_id}, chunk {c.chunk_id}"
            out.append((citations.resolve(c, base_url) or "", label))
    return out


def _meta(
    stamp: str,
    requested_by: str,
    model: str,
    embed_model: str,
    audience: Sequence[str],
    sources: Sequence[str],
    run_id: str,
) -> list[tuple[str, str]]:
    rows = [("📅", stamp), ("📂", ", ".join(sources) or "—")]
    if requested_by:
        rows.append(("👤", f"requested by {requested_by}"))
    if model:
        rows.append(("🤖", f"generated by {model}"))
    rows += [
        ("🧠", f"embeddings: {embed_model}"),
        ("👥", ", ".join(audience) or "—"),
        ("🔖", run_id or "—"),
        ("⬦", f"docusearch v{__version__}"),
    ]
    return rows


def _collect_numbering(
    texts: Sequence[str],
) -> tuple[dict[tuple[int, int], int], list[citations.Citation]]:
    numbering: dict[tuple[int, int], int] = {}
    ordered: list[citations.Citation] = []
    for text in texts:
        for c in citations.parse(text):
            if c.kind == "doc" and c.doc_id is not None and c.chunk_id is not None:
                key = (c.doc_id, c.chunk_id)
                if key not in numbering:
                    numbering[key] = len(ordered) + 1
                    ordered.append(c)
    return numbering, ordered


# ------------------------------------------------------------------ inline citation markers


def _cite_md(text: str, numbering: dict[tuple[int, int], int]) -> str:
    def repl(m: re.Match[str]) -> str:
        if m.group(0) == "[GK]":
            return "[GK]"
        key = (int(m.group(1)), int(m.group(2)))
        n = numbering.get(key)
        return f"[{n}]" if n is not None else m.group(0)

    return _CITE_RE.sub(repl, text)


def _cite_html(escaped: str, numbering: dict[tuple[int, int], int]) -> str:
    """Replace citation tags (which survive HTML-escaping intact) with superscript links."""

    def repl(m: re.Match[str]) -> str:
        if m.group(0) == "[GK]":
            return '<sup class="gk" title="general knowledge">GK</sup>'
        key = (int(m.group(1)), int(m.group(2)))
        n = numbering.get(key)
        if n is None:
            return m.group(0)
        return f'<sup class="cite"><a href="#ref-{n}">{n}</a></sup>'

    return _CITE_RE.sub(repl, escaped)


# ---------------------------------------------------------------------- lightweight markdown


def _inline(text: str, numbering: dict[tuple[int, int], int]) -> str:
    """Escape a line, then apply citations, **bold**, and `code` (in that safe order)."""
    out = _html.escape(text)
    out = _cite_html(out, numbering)
    out = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", out)
    out = re.sub(r"`([^`]+)`", r"<code>\1</code>", out)
    return out


def _blocks_html(body: str, numbering: dict[tuple[int, int], int]) -> str:
    """Render a section body: fenced code blocks verbatim, plus paragraphs and lists."""
    parts = re.split(r"```[^\n]*\n(.*?)```", body, flags=re.S)
    html_parts: list[str] = []
    for i, seg in enumerate(parts):
        if i % 2 == 1:  # captured code block
            html_parts.append(f'<pre class="code"><code>{_html.escape(seg.rstrip())}</code></pre>')
        elif seg.strip():
            html_parts.append(_prose_html(seg, numbering))
    return "".join(html_parts)


def _prose_html(text: str, numbering: dict[tuple[int, int], int]) -> str:
    out: list[str] = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        if all(re.match(r"[-*]\s+", ln) for ln in lines):
            items = "".join(f"<li>{_inline(ln[2:].strip(), numbering)}</li>" for ln in lines)
            out.append(f"<ul>{items}</ul>")
        elif all(re.match(r"\d+[.)]\s+", ln) for ln in lines):
            stripped = [re.sub(r"^\d+[.)]\s+", "", ln) for ln in lines]
            items = "".join(f"<li>{_inline(ln, numbering)}</li>" for ln in stripped)
            out.append(f"<ol>{items}</ol>")
        else:
            out.append(f"<p>{_inline(' '.join(lines), numbering)}</p>")
    return "".join(out)


# --------------------------------------------------------------------------------- renderers


def _render_markdown(
    title: str,
    subtitle: str,
    header: tuple[str, str],
    meta: list[tuple[str, str]],
    secs: list[tuple[str, str, str]],
    numbering: dict[tuple[int, int], int],
    refs: list[tuple[str, str]],
    image_urls: list[str],
    embedded_images: Sequence[tuple[str, str]],
    trace: Mapping[str, object] | None,
) -> str:
    classification, request = header
    out: list[str] = []
    if classification:
        out += [f"**{classification.upper()}**", ""]
    out.append(f"# {title}")
    if subtitle:
        out += ["", f"_{subtitle}_"]
    out += ["", f"> ⚠️ **{_AI_WARNING}**", ""]
    if request:
        out += [f'**Request:** "{request}"', ""]
    out += ["> " + "  ·  ".join(f"{v}" for _icon, v in meta), ""]
    for heading, _kind, sec_body in secs:
        if heading:
            out.append(f"## {heading}")
            out.append("")
        out.append(_cite_md(sec_body.strip(), numbering))
        out.append("")
    if embedded_images or image_urls:
        out += ["## Figures"]
        # cited images are embedded inline (base64 data URI) so they render even if the
        # original file disappears; server URLs (live-server case) follow.
        out += [f"![{cap}]({src})" for src, cap in embedded_images]
        out += [f"![image]({u})" for u in image_urls]
        out += [""]
    ref_lines = [f"{i}. [{label}]({href})" for i, (href, label) in enumerate(refs, 1)]
    out += ["## References", *(ref_lines or ["(none)"]), ""]
    out += _trace_md(trace)
    return "\n".join(out)


def _render_html(
    title: str,
    subtitle: str,
    header: tuple[str, str],
    meta: list[tuple[str, str]],
    secs: list[tuple[str, str, str]],
    numbering: dict[tuple[int, int], int],
    refs: list[tuple[str, str]],
    image_urls: list[str],
    embedded_images: Sequence[tuple[str, str]],
    trace: Mapping[str, object] | None,
) -> str:
    esc = _html.escape
    classification, request = header
    chips = "".join(f'<span class="chip">{icon}&nbsp;{esc(v)}</span>' for icon, v in meta)
    cards: list[str] = []
    for heading, kind, sec_body in secs:
        icon, cls = _KINDS.get(kind, ("📄", "kind-overview"))
        head = (
            f'<div class="card-head"><span class="ic">{icon}</span><h2>{esc(heading)}</h2></div>'
            if heading
            else ""
        )
        cards.append(
            f'<section class="card {cls}">{head}'
            f'<div class="card-body">{_blocks_html(sec_body, numbering)}</div></section>'
        )
    if embedded_images or image_urls:
        # cited images are embedded inline as base64 data URIs, so the figure survives even if
        # the original file is moved/deleted (self-contained HTML). Server URLs follow.
        figs = "".join(
            f'<figure><img src="{esc(src)}" alt="{esc(cap)}">'
            f'<figcaption>{esc(cap)}</figcaption></figure>'
            for src, cap in embedded_images
        )
        figs += "".join(f'<img src="{esc(u)}" alt="figure">' for u in image_urls)
        cards.append(
            '<section class="card kind-reference"><div class="card-head">'
            '<span class="ic">🖼️</span><h2>Figures</h2></div>'
            f'<div class="card-body figures">{figs}</div></section>'
        )
    ref_items = (
        "".join(
            f'<li id="ref-{i}"><a href="{esc(href)}">{esc(label)}</a></li>'
            for i, (href, label) in enumerate(refs, 1)
        )
        or "<li>(none)</li>"
    )
    refs_card = (
        '<section class="card kind-reference"><div class="card-head">'
        '<span class="ic">📎</span><h2>References</h2></div>'
        f'<div class="card-body"><ol class="refs">{ref_items}</ol></div></section>'
    )
    subtitle_html = f'<p class="subtitle">{esc(subtitle)}</p>' if subtitle else ""
    ribbon = f'<div class="ribbon">{esc(classification)}</div>' if classification else ""
    request_html = (
        f'<div class="request"><span>Request</span> “{esc(request)}”</div>' if request else ""
    )
    warning = f'<div class="ai-warning"><span class="wic">⚠️</span>{esc(_AI_WARNING)}</div>'
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{esc(title)}</title><style>{_CSS}</style></head><body>"
        '<div class="report">'
        f"{ribbon}"
        '<header class="banner"><div class="eyebrow">docusearch report</div>'
        f"<h1>{esc(title)}</h1>{subtitle_html}{request_html}"
        f'<div class="meta">{chips}</div></header>'
        f"{warning}"
        f"<main>{''.join(cards)}{refs_card}</main>"
        f"{_trace_html(trace)}"
        '<footer class="foot">Generated by docusearch — every claim is '
        '<span class="cite-legend">[n]</span> a cited source or '
        '<span class="gk">GK</span> general knowledge.</footer>'
        "</div></body></html>"
    )


_CSS = """
:root{--bg:#0a1730;--card:#0f2547;--card2:#123059;--border:#1d3f6e;
--accent:#7fdbff;--accent2:#48cae4;--accent-dim:#2b6f97;--text:#e8f2ff;
--muted:#9fb6d6;--code:#081426;}
*{box-sizing:border-box}
body{margin:0;color:var(--text);
background:linear-gradient(160deg,#08122a,#0a1730 45%,#0b1d3a);
font:16px/1.62 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;}
.report{max-width:940px;margin:0 auto;padding:20px 20px 72px;}
a{color:var(--accent2);}
.ribbon{text-align:center;letter-spacing:.22em;text-transform:uppercase;font-size:12px;
font-weight:700;color:#ffd7a8;background:linear-gradient(90deg,#3a1d12,#5a2c17,#3a1d12);
border:1px solid #7a3f22;border-radius:8px;padding:7px 12px;margin-bottom:16px;}
.banner{background:linear-gradient(135deg,#0c2350,#0e2c63);border:1px solid var(--border);
border-left:4px solid var(--accent);border-radius:16px;padding:26px 32px;
box-shadow:0 10px 34px rgba(0,0,0,.4);}
.request{margin:.6em 0 1em;padding:10px 14px;border-left:3px solid var(--accent2);
background:rgba(127,219,255,.06);border-radius:0 8px 8px 0;color:#d6e8ff;font-style:italic;}
.request span{font-style:normal;font-weight:700;color:var(--accent2);text-transform:uppercase;
font-size:11px;letter-spacing:.12em;margin-right:8px;}
.ai-warning{display:flex;align-items:center;gap:10px;margin-top:16px;padding:12px 18px;
color:#ffe1b0;background:linear-gradient(90deg,rgba(120,70,20,.35),rgba(90,55,20,.25));
border:1px solid #8a5a2a;border-left:4px solid #ffb703;border-radius:12px;font-size:13.5px;}
.ai-warning .wic{font-size:17px;}
.eyebrow{letter-spacing:.2em;font-size:11.5px;color:var(--accent2);text-transform:uppercase;}
.banner h1{margin:.25em 0 .1em;font-size:30px;line-height:1.2;color:var(--accent);
letter-spacing:-.01em;}
.subtitle{margin:.15em 0 1.1em;color:var(--muted);font-size:16.5px;}
.meta{display:flex;flex-wrap:wrap;gap:8px;}
.chip{background:#0a1c3a;border:1px solid var(--border);border-radius:999px;
padding:4px 12px;font-size:12.5px;color:var(--muted);white-space:nowrap;}
.card{background:linear-gradient(180deg,var(--card),var(--card2));border:1px solid var(--border);
border-left:4px solid var(--accent-dim);border-radius:14px;margin-top:20px;overflow:hidden;
box-shadow:0 5px 20px rgba(0,0,0,.28);}
.card-head{display:flex;align-items:center;gap:11px;padding:15px 24px;
border-bottom:1px solid var(--border);background:rgba(127,219,255,.045);}
.card-head .ic{font-size:19px;line-height:1;}
.card-head h2{margin:0;font-size:17.5px;color:var(--accent);}
.card-body{padding:18px 24px;}
.card-body p{margin:.35em 0 .85em;}
.card-body p:last-child{margin-bottom:.2em;}
.card-body ul,.card-body ol{margin:.3em 0 .9em 1.25em;padding:0;}
.card-body li{margin:.3em 0;}
.card-body strong{color:#eaf4ff;}
pre.code{background:var(--code);border:1px solid var(--border);border-radius:10px;
padding:14px 16px;overflow-x:auto;margin:.4em 0 1em;
font:13.5px/1.55 "SF Mono",SFMono-Regular,Menlo,Consolas,monospace;color:#cfe8ff;}
code{background:#0a1c3a;border:1px solid var(--border);border-radius:5px;padding:.06em .36em;
font:13px "SF Mono",SFMono-Regular,Menlo,Consolas,monospace;color:#bfe6ff;}
pre.code code{background:none;border:0;padding:0;}
sup.cite{font-size:.7em;font-weight:700;}
sup.cite a{color:var(--accent2);text-decoration:none;padding:0 1px;}
sup.cite a:hover{text-decoration:underline;}
sup.gk,.gk{color:var(--muted);font-size:.72em;font-weight:600;}
.refs{margin:0;padding-left:1.4em;}
.refs li{margin:.4em 0;color:var(--muted);word-break:break-all;}
.figures img{max-width:100%;border:1px solid var(--border);border-radius:8px;margin:.4em 0;}
.trace{margin-top:22px;background:#0a1b36;border:1px solid var(--border);border-radius:12px;
overflow:hidden;}
.trace>summary{cursor:pointer;list-style:none;padding:13px 22px;color:var(--accent2);
font-size:13.5px;font-weight:600;user-select:none;}
.trace>summary::-webkit-details-marker{display:none;}
.trace>summary::before{content:"▸ ";color:var(--accent-dim);}
.trace[open]>summary::before{content:"▾ ";}
.trace-body{padding:4px 22px 18px;border-top:1px solid var(--border);}
.trace-body h3{font-size:12px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);
margin:16px 0 6px;}
.trace-list{margin:.2em 0 .6em 1.2em;padding:0;color:#cfe0f5;font-size:13.5px;}
.trace-list li{margin:.2em 0;word-break:break-word;}
.trace-pre{white-space:pre-wrap;background:var(--code);border:1px solid var(--border);
border-radius:8px;padding:10px 12px;color:#cfe8ff;font-size:12.5px;overflow-x:auto;}
.foot{margin-top:30px;text-align:center;color:var(--muted);font-size:12.5px;}
.cite-legend{color:var(--accent2);font-weight:700;}
.kind-overview{border-left-color:var(--accent);}
.kind-procedure{border-left-color:#64dfdf;}
.kind-code{border-left-color:#4cc9f0;}
.kind-hardware{border-left-color:#48cae4;}
.kind-config{border-left-color:#56cfe1;}
.kind-test{border-left-color:#80ffdb;}
.kind-warning{border-left-color:#ffb703;}
.kind-reference{border-left-color:#5aa9e6;}
"""

# Analytics/diff styling layered on the shared theme (cards, data grids, Beyond-Compare colors).
_ANALYTICS_CSS = """
.acard{background:var(--card);border:1px solid var(--border);border-left:4px solid var(--accent);
border-radius:14px;padding:18px 22px;margin:18px 0;box-shadow:0 8px 24px rgba(0,0,0,.3);}
.acard h2{color:var(--accent);margin:.1em 0 .5em;font-size:21px;}
.acard h3{color:var(--accent2);margin:.2em 0 .5em;font-size:16px;}
.acard p.stats{color:var(--muted);font-size:13px;margin:.4em 0;}
.acard img.plot{max-width:100%;height:auto;border-radius:8px;background:#fff;padding:6px;}
.scroll{overflow-x:auto;border-radius:10px;}
table.grid{border-collapse:collapse;width:100%;font-size:12.5px;margin:.3em 0;}
table.grid th,table.grid td{border:1px solid var(--border);padding:5px 9px;text-align:left;
white-space:nowrap;}
table.grid th{background:var(--card2);color:var(--accent2);text-transform:uppercase;
letter-spacing:.04em;font-size:11px;}
table.grid tr:nth-child(even) td{background:rgba(255,255,255,.02);}
tr.added td{background:rgba(64,180,90,.14);} tr.removed td{background:rgba(210,70,70,.16);}
tr.changed td{background:rgba(230,180,60,.10);}
td.chg{background:rgba(230,180,60,.34)!important;font-weight:700;color:#ffe6a0;}
.badge{display:inline-block;padding:1px 9px;border-radius:20px;font-size:11px;font-weight:700;
letter-spacing:.03em;}
.badge.added{background:#256b39;color:#c8f5d6;} .badge.removed{background:#7a2f2f;color:#f7cccc;}
.badge.changed{background:#7a5f22;color:#ffe6a0;} .badge.identical{background:#20406b;color:#bcd6ff;}
.tabbar{display:flex;gap:6px;flex-wrap:wrap;margin:18px 0 0;border-bottom:1px solid var(--border);}
.tab{background:transparent;border:1px solid var(--border);border-bottom:none;color:var(--muted);
padding:9px 18px;border-radius:10px 10px 0 0;cursor:pointer;font:inherit;font-size:14px;font-weight:600;}
.tab:hover{color:var(--text);} .tab.active{background:var(--card);color:var(--accent);
border-color:var(--border);box-shadow:0 -2px 0 var(--accent) inset;}
.panel{display:block;} .panel.hidden{display:none;}
.toolbar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:14px 0;}
.toolbar input[type=text]{background:var(--code);border:1px solid var(--border);color:var(--text);
border-radius:8px;padding:8px 12px;font:inherit;min-width:220px;}
.toolbar button{background:var(--accent-dim);border:1px solid var(--accent2);color:#eaf6ff;
border-radius:8px;padding:8px 14px;font:inherit;font-weight:600;cursor:pointer;}
.toolbar button:hover{background:var(--accent2);color:#062033;}
.toolbar .hint{color:var(--muted);font-size:12.5px;}
table.grid th.sortable{cursor:pointer;user-select:none;} table.grid th.sortable:hover{color:var(--accent);}
table.grid th.sortable::after{content:" ⇅";opacity:.5;font-size:10px;}
table.grid td.fb{background:rgba(127,219,255,.05);min-width:120px;cursor:text;}
table.grid td.fb:focus{outline:2px solid var(--accent2);background:rgba(127,219,255,.12);}
.plotgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:16px;}
@media(max-width:520px){.plotgrid{grid-template-columns:1fr;}}
.chipbar{display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin:14px 0 4px;}
.chip{background:var(--card2);border:1px solid var(--border);color:var(--muted);border-radius:20px;
padding:5px 13px;cursor:pointer;font:inherit;font-size:12.5px;font-weight:600;}
.chip:hover{color:var(--text);border-color:var(--accent2);}
.chip.inc{background:var(--accent-dim);color:#eaf6ff;border-color:var(--accent2);}
.chip.inc::before{content:"✓ ";}
.chip.exc{background:#5c2626;color:#f7cccc;border-color:#c07;text-decoration:line-through;}
.chip.exc::before{content:"✕ ";text-decoration:none;display:inline-block;}
.chipmode,.chipclear{background:var(--code);border:1px solid var(--accent2);color:var(--accent);
border-radius:20px;padding:5px 12px;cursor:pointer;font:inherit;font-size:12px;font-weight:700;}
.chipmode:hover,.chipclear:hover{background:var(--accent2);color:#062033;}
.chiphint{color:var(--muted);font-size:11.5px;margin-right:2px;}
.sortbar{display:flex;gap:8px;align-items:center;margin:2px 0 10px;}
.sortbtn{background:var(--code);border:1px solid var(--accent2);color:var(--accent);border-radius:20px;
padding:5px 13px;cursor:pointer;font:inherit;font-size:12px;font-weight:700;}
.sortbtn:hover{background:var(--accent2);color:#062033;}
/* wafer map: a die grid coloured by pass/fail or soft bin */
.wafermap{display:grid;grid-auto-rows:13px;gap:1px;justify-content:start;margin:12px 0;
background:var(--code);padding:8px;border-radius:8px;overflow:auto;max-width:100%;}
.die{width:13px;height:13px;border-radius:2px;background:var(--card2);}
.die.pass{background:#3cb371;} .die.fail{background:#d64545;}
td .tag,.tag{display:inline-block;background:var(--card2);border:1px solid var(--border);
color:var(--accent2);border-radius:6px;padding:0 6px;margin:1px 2px;font-size:10.5px;font-weight:600;}
/* fixed-size, self-scrolling table with a frozen header row; expand toggles a full-window overlay */
.tablepanel{margin:.3em 0;}
/* Full screen stops BELOW the tab bar and pins it, so you can still switch tabs while
   maximized — a full-viewport overlay hid the one control you most want up there. */
.tablepanel.full{position:fixed;inset:var(--tabbar-h,56px) 0 0 0;z-index:1000;
background:var(--bg);padding:14px 16px;
display:flex;flex-direction:column;overflow:hidden;box-shadow:0 0 0 100vmax var(--bg);}
body.panel-full .tabbar{position:fixed;top:0;left:0;right:0;z-index:1001;margin:0;
background:var(--bg);padding:8px 16px 0;box-shadow:0 6px 18px -12px #000;}
.tablewrap{overflow:auto;max-height:60vh;border:1px solid var(--border);border-radius:10px;
-webkit-overflow-scrolling:touch;}
.tablepanel.full .tablewrap{max-height:none;flex:1;}
.tablewrap table.grid thead th{position:sticky;top:0;z-index:3;}
.expand-btn{background:var(--card2);border:1px solid var(--accent2);color:var(--accent);
border-radius:8px;padding:8px 14px;font:inherit;font-weight:600;cursor:pointer;}
.expand-btn:hover{background:var(--accent2);color:#062033;}
"""


def themed_page(
    title: str,
    body_html: str,
    *,
    subtitle: str = "",
    classification: str = "Confidential",
    meta: Sequence[tuple[str, str]] = (),
    eyebrow: str = "docusearch analytics",
) -> str:
    """Wrap analytics body HTML in the shared docusearch theme (ribbon + banner + CSS), so STDF
    plots/diffs read like the cited reports, not bare fragments."""
    esc = _html.escape
    chips = "".join(f'<span class="chip">{icon}&nbsp;{esc(v)}</span>' for icon, v in meta)
    ribbon = f'<div class="ribbon">{esc(classification)}</div>' if classification else ""
    subtitle_html = f'<p class="subtitle">{esc(subtitle)}</p>' if subtitle else ""
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{esc(title)}</title><style>{_CSS}{_ANALYTICS_CSS}</style></head><body>"
        '<div class="report">'
        f"{ribbon}"
        f'<header class="banner"><div class="eyebrow">{esc(eyebrow)}</div>'
        f"<h1>{esc(title)}</h1>{subtitle_html}"
        f'<div class="meta">{chips}</div></header>'
        f"<main>{body_html}</main>"
        '<footer class="foot">Generated by docusearch.</footer>'
        "</div></body></html>"
    )


_IMG_MEDIA = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "gif": "gif", "webp": "webp",
              "bmp": "bmp", "svg": "svg+xml"}


def reference_targets(
    db_path: str, evidence: set[tuple[int, int]], *, base_url: str = ""
) -> dict[tuple[int, int], tuple[str, str]]:
    """Map each cited ``(doc_id, chunk_id)`` to ``(href, "store — title — heading")`` so a report's
    references carry a meaningful label AND a link. The label is identical everywhere; only the href
    differs by context: a **served** report (``base_url`` given) links to the HTTP
    ``/v1/documents`` endpoint a remote client can open, while the local CLI (no ``base_url``) links
    to the original ``file://`` document. Shared by the CLI and the MCP/REST builders (R-REUSE-2)."""
    from pathlib import Path

    from .store import Store

    targets: dict[tuple[int, int], tuple[str, str]] = {}
    if db_path == ":memory:" or not evidence:
        return targets
    with Store.open(db_path) as store:
        for doc_id, chunk_id in evidence:
            info = store.citation_target(doc_id, chunk_id)
            if info is None:
                continue
            source, title, path, locator = info
            parts = [p for p in (source, title or f"document {doc_id}") if p]
            if locator and locator != title:
                parts.append(locator)
            label = " — ".join(parts)
            if base_url:
                href = f"{base_url}/v1/documents/{doc_id}?chunk={chunk_id}"
            else:
                try:
                    href = Path(path).as_uri() if path else ""
                except ValueError:  # non-absolute path -> leave as-is
                    href = path
            targets[(doc_id, chunk_id)] = (href, label)
    return targets


def evidence_images(
    db_path: str, staging_dir: str, evidence: set[tuple[int, int]]
) -> list[tuple[str, str]]:
    """For every cited image chunk, embed its retained diagram as a base64 ``data:`` URI so the
    report is self-contained — the figure renders even if the original file later disappears.
    Returns ``[(data_uri, caption)]``, deduped by image, in citation order. Shared by CLI + MCP."""
    import base64
    from pathlib import Path

    from .store import Store

    out: list[tuple[str, str]] = []
    if db_path == ":memory:" or not evidence:
        return out
    images_dir = (Path(staging_dir) / "images").resolve()
    seen: set[str] = set()
    with Store.open(db_path) as store:
        for doc_id, chunk_id in sorted(evidence):
            for img in store.images_for_chunk(doc_id, chunk_id):
                sha, ext = str(img["sha256"]), str(img["ext"] or "").lower()
                if sha in seen:
                    continue
                path = (images_dir / f"{sha}.{ext}").resolve()
                if not path.is_relative_to(images_dir) or not path.is_file():
                    continue
                seen.add(sha)
                media = _IMG_MEDIA.get(ext, "png")
                b64 = base64.b64encode(path.read_bytes()).decode("ascii")
                caption = " — ".join(
                    t for t in (str(img["alt"] or ""), str(img["caption"] or "")) if t
                ) or f"figure [D:{doc_id}#{chunk_id}]"
                out.append((f"data:image/{media};base64,{b64}", caption))
    return out


# ------------------------------------------------------------------------- slide deck


_SLIDE_CSS = """
/* Slide deck: same theme as the report, one section per slide. */
html,body{height:100%;overflow:hidden;}
.deck{height:100vh;width:100vw;position:relative;}
.slide{position:absolute;inset:0;display:none;flex-direction:column;
justify-content:center;padding:5vh 7vw 12vh;overflow-y:auto;}
.slide.on{display:flex;animation:slide-in .18s ease-out;}
@keyframes slide-in{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
.slide h1{font-size:clamp(28px,4.6vw,60px);line-height:1.12;margin:0 0 .3em;}
.slide h2{font-size:clamp(22px,3.2vw,40px);line-height:1.2;margin:0 0 .5em;
color:var(--accent);display:flex;align-items:center;gap:.4em;}
.slide .body{font-size:clamp(15px,1.55vw,22px);}
.slide .body p{margin:0 0 .7em;}
.slide.cover{justify-content:center;text-align:left;}
.slide.cover .subtitle{font-size:clamp(16px,2vw,26px);color:var(--muted);margin:0 0 1.2em;}
.deck-meta{color:var(--muted);font-size:14px;display:flex;flex-wrap:wrap;gap:.4em 1.4em;}
.refs-slide ol{font-size:clamp(13px,1.25vw,18px);}
/* chrome */
.bar{position:fixed;left:0;bottom:0;height:3px;background:var(--accent);
transition:width .18s ease;z-index:5;}
.hud{position:fixed;right:16px;bottom:12px;color:var(--muted);font-size:13px;
display:flex;align-items:center;gap:14px;z-index:5;}
.hud button{background:transparent;border:1px solid var(--border);color:var(--muted);
border-radius:8px;padding:2px 9px;font-size:14px;cursor:pointer;}
.hud button:hover{color:var(--text);border-color:var(--accent-dim);}
.hint{position:fixed;left:16px;bottom:12px;color:var(--muted);font-size:12.5px;
opacity:.75;z-index:5;}
@media print{html,body{overflow:visible;height:auto}
.slide{display:flex!important;position:relative;page-break-after:always;height:100vh;}
.bar,.hud,.hint{display:none}}
"""

# PowerPoint's own navigation keys, so muscle memory transfers.
_SLIDE_JS = """
(function(){
 var s=[].slice.call(document.querySelectorAll('.slide')),i=0;
 var bar=document.querySelector('.bar'),now=document.querySelector('.now');
 function go(n){
  i=Math.max(0,Math.min(s.length-1,n));
  s.forEach(function(el,k){el.classList.toggle('on',k===i);});
  if(bar)bar.style.width=((i+1)/s.length*100)+'%';
  if(now)now.textContent=(i+1)+' / '+s.length;
  if(history.replaceState)history.replaceState(null,'','#'+(i+1));
 }
 var NEXT=['ArrowRight','ArrowDown','PageDown','Enter',' ','n','N'];
 var PREV=['ArrowLeft','ArrowUp','PageUp','Backspace','p','P'];
 document.addEventListener('keydown',function(e){
  if(NEXT.indexOf(e.key)>=0){go(i+1);e.preventDefault();}
  else if(PREV.indexOf(e.key)>=0){go(i-1);e.preventDefault();}
  else if(e.key==='Home'){go(0);e.preventDefault();}
  else if(e.key==='End'){go(s.length-1);e.preventDefault();}
  else if(e.key==='f'||e.key==='F'||e.key==='F5'){full();e.preventDefault();}
 });
 function full(){if(document.fullscreenElement)document.exitFullscreen();
  else if(document.documentElement.requestFullscreen)document.documentElement.requestFullscreen();}
 var fb=document.querySelector('.fs');if(fb)fb.addEventListener('click',full);
 var nb=document.querySelector('.nx');if(nb)nb.addEventListener('click',function(){go(i+1);});
 var pb=document.querySelector('.pv');if(pb)pb.addEventListener('click',function(){go(i-1);});
 // advance on click, except on a link or a control
 document.addEventListener('click',function(e){
  if(e.target.closest('a,button'))return;go(i+1);});
 var start=parseInt((location.hash||'').replace('#',''),10);
 go(isNaN(start)?0:start-1);
})();
"""


def _render_slides(
    title: str,
    subtitle: str,
    header: tuple[str, str],
    meta: list[tuple[str, str]],
    secs: list[tuple[str, str, str]],
    numbering: dict[tuple[int, int], int],
    refs: list[tuple[str, str]],
    image_urls: list[str],
    embedded_images: Sequence[tuple[str, str]],
    trace: Mapping[str, object] | None,
) -> str:
    """A self-contained HTML slide deck in the report theme: one section per slide, navigated
    with PowerPoint's own keys. Same content, numbering and references as the HTML report — only
    the layout differs, so a deck and a report of the same spec cite identically (R-REUSE-2)."""
    esc = _html.escape
    classification, request = header
    slides: list[str] = []
    meta_html = "".join(f"<span><b>{esc(k)}</b> {esc(v)}</span>" for k, v in meta)
    request_html = f'<div class="request"><span>Request</span> “{esc(request)}”</div>' if request else ""
    slides.append(
        '<section class="slide cover"><p class="eyebrow">' + esc(classification) + "</p>"
        f"<h1>{esc(title)}</h1>"
        + (f'<p class="subtitle">{esc(subtitle)}</p>' if subtitle else "")
        + request_html
        + f'<div class="deck-meta">{meta_html}</div>'
        + f'<div class="ai-warning"><span class="wic">⚠️</span>{esc(_AI_WARNING)}</div>'
        + "</section>"
    )
    for heading, kind, body in secs:
        icon, cls = _KINDS.get(kind.lower(), ("📄", "kind-generic"))
        slides.append(
            f'<section class="slide {cls}">'
            + (f"<h2><span class=\"ic\">{icon}</span>{esc(heading)}</h2>" if heading else "")
            + f'<div class="body">{_blocks_html(body, numbering)}</div></section>'
        )
    if image_urls or embedded_images:
        figs = "".join(f'<img src="{esc(u)}" alt="">' for u in image_urls)
        figs += "".join(f'<img src="{esc(src)}" alt="{esc(alt)}">' for src, alt in embedded_images)
        slides.append(
            '<section class="slide"><h2><span class="ic">🖼️</span>Figures</h2>'
            f'<div class="body figures">{figs}</div></section>'
        )
    ref_items = "".join(f'<li>{esc(label)}</li>' for _href, label in refs) or "<li>(none)</li>"
    slides.append(
        '<section class="slide refs-slide"><h2><span class="ic">📎</span>References</h2>'
        f'<div class="body"><ol class="refs">{ref_items}</ol></div></section>'
    )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{esc(title)}</title><style>{_CSS}{_SLIDE_CSS}</style></head><body>"
        f'<div class="deck">{"".join(slides)}</div>'
        '<div class="bar"></div>'
        '<div class="hint">← → navigate · F full screen</div>'
        '<div class="hud"><button class="pv">‹</button><span class="now"></span>'
        '<button class="nx">›</button><button class="fs">⛶</button></div>'
        f"<script>{_SLIDE_JS}</script></body></html>"
    )
