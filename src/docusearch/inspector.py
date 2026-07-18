"""Document-shape inspector — an ingestion-pipeline aid (config tuning).

Samples a source's HTML and proposes ``content_selector`` / ``strip_selectors`` so a human
(or an AI tuning the config) can match the config to the document's shape instead of
hand-inspecting the markup. Entirely generic — it knows nothing about any specific corpus,
so it helps every source (and never special-cases one, which would risk the others).

Heuristics:
    content_selector : of a set of common body-container selectors, the one that matches
                       most sampled docs and captures most of the body text (but not the
                       whole page, so page-level chrome is excluded).
    strip_selectors  : semantic chrome tags present in the corpus (script/style/nav/…) plus
                       id/class selectors whose name looks like chrome (breadcrumb/nav/…)
                       and that recur across many docs.

Public surface:
    inspect_html(docs) -> InspectResult
    InspectResult(sampled, content_selector, content_candidates, strip_selectors)
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from selectolax.parser import HTMLParser

_CONTENT_CANDIDATES: tuple[str, ...] = (
    "main",
    "article",
    "[role=main]",
    "[role=article]",
    "div.body",
    "div.content",
    "div#content",
    "div#main",
    "div.markdown-body",
    "div.wh_topic_content",
    "section.content",
    "div.document",
)
_SAFE_TAGS: tuple[str, ...] = ("script", "style", "nav", "header", "footer", "aside")
_CHROME_KEYWORDS: tuple[str, ...] = (
    "nav",
    "breadcrumb",
    "crumb",
    "footer",
    "header",
    "sidebar",
    "toc",
    "menu",
    "banner",
    "prevnext",
    "pagination",
)


@dataclass
class InspectResult:
    sampled: int
    content_selector: str
    content_candidates: list[tuple[str, float, float]]  # (selector, match_rate, coverage)
    strip_selectors: list[str]


def _text_len(node: object) -> int:
    return len(" ".join(node.text().split())) if node is not None else 0  # type: ignore[attr-defined]


def inspect_html(docs: list[str]) -> InspectResult:
    """Propose selectors from a list of raw HTML documents."""
    bodies = []
    for html in docs:
        try:
            tree = HTMLParser(html)
        except Exception:  # noqa: BLE001 - skip an unparseable sample, don't abort
            continue
        if tree.body is not None:
            bodies.append(tree.body)
    n = len(bodies)
    if n == 0:
        return InspectResult(0, "", [], [])

    # --- content_selector: match rate + text coverage per candidate ------------------
    stats: list[tuple[str, float, float]] = []
    for sel in _CONTENT_CANDIDATES:
        matches = 0
        cov_sum = 0.0
        for body in bodies:
            try:
                el = body.css_first(sel)
            except Exception:  # noqa: BLE001 - a bad selector on a weird doc
                el = None
            if el is not None:
                matches += 1
                cov_sum += _text_len(el) / (_text_len(body) or 1)
        if matches:
            stats.append((sel, matches / n, cov_sum / matches))
    # prefer high match-rate, then high coverage but capped so a whole-page match (chrome
    # included) doesn't beat a tight body container.
    stats.sort(key=lambda s: (round(s[1], 2), round(min(s[2], 0.97), 2)), reverse=True)
    content = stats[0][0] if stats and stats[0][1] >= 0.6 and stats[0][2] >= 0.4 else ""

    # --- strip_selectors: recurring chrome ------------------------------------------
    counter: Counter[str] = Counter()
    for body in bodies:
        seen: set[str] = set()
        for tag in _SAFE_TAGS:
            if body.css_first(tag) is not None:
                seen.add(tag)
        for el in body.css("*"):
            attrs = el.attributes or {}
            nid = attrs.get("id") or ""
            cls = attrs.get("class") or ""
            blob = f"{nid} {cls}".lower()
            if any(k in blob for k in _CHROME_KEYWORDS):
                if nid:
                    seen.add(f"#{nid}")
                elif cls:
                    seen.add("." + cls.split()[0])
        counter.update(seen)
    threshold = max(1, int(0.3 * n))
    strip = [sel for sel, count in counter.items() if count >= threshold]
    strip.sort(key=lambda s: (s not in _SAFE_TAGS, s))  # safe tags first, then chrome

    return InspectResult(n, content, stats[:6], strip)
