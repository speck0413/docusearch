"""Derive a house **style guide** from indexed code symbols (Phase 9 / GATE 9).

Pure statistics over the :class:`docusearch.code_index.Symbol` set the parser already produced — no AI,
no re-parsing. For each language it reports the dominant **naming convention** per symbol group
(callables vs types), **docstring coverage**, **typing discipline** (Python/TS return annotations),
and **structure** (symbol mix, typical size, methods per type) — the conventions a human or an agent
reads before writing new code that fits the codebase.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from html import escape
from typing import TYPE_CHECKING

from . import report

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .code_index import Symbol

# Symbol kinds grouped for convention reporting: things you call vs things you instantiate/implement.
_CALLABLE = frozenset({"function", "method"})
_TYPE = frozenset({"class", "struct", "interface", "trait", "enum"})
_GROUPS: dict[str, frozenset[str]] = {"callable": _CALLABLE, "type": _TYPE}


@dataclass(frozen=True)
class Naming:
    """The naming picture for one symbol group: the dominant convention, its share, and the spread."""

    convention: str
    rate: float
    distribution: dict[str, int]


@dataclass(frozen=True)
class StyleGuide:
    """Derived conventions for one language."""

    language: str
    counts: dict[str, int] = field(default_factory=dict)          # kind -> n
    naming: dict[str, Naming] = field(default_factory=dict)       # group -> Naming
    docstring_coverage: dict[str, float] = field(default_factory=dict)  # group -> fraction
    return_annotation_rate: float | None = None                  # callables with a return type (py/ts)
    avg_lines: float = 0.0
    median_lines: float = 0.0
    avg_methods_per_type: float = 0.0
    total: int = 0


def name_convention(name: str) -> str:
    """Classify an identifier's case into a convention family (leading/trailing ``_`` ignored, so
    ``__init__`` and ``_helper`` classify by their core). A single lowercase word is snake-compatible;
    a single capitalised word is Pascal-compatible."""
    core = name.strip("_")
    if not core:
        return "other"
    if any(c.isalpha() for c in core) and core.upper() == core and core.lower() != core:
        return "SCREAMING_SNAKE"
    if "_" in core:
        return "snake_case" if core.islower() else "mixed"
    if core[0].isupper():
        return "PascalCase"
    if any(c.isupper() for c in core):
        return "camelCase"
    return "snake_case"  # single lowercase word — valid snake_case


def _has_return_annotation(signature: str, language: str) -> bool | None:
    if language == "python":
        return "->" in signature
    if language == "typescript":
        return bool(re.search(r"\)\s*:", signature))  # `): ReturnType`
    return None


def derive_style(symbols: Sequence[Symbol], language: str) -> StyleGuide:
    """Derive :class:`StyleGuide` conventions for one ``language`` from its symbols."""
    syms = [s for s in symbols if s.language == language]
    if not syms:
        return StyleGuide(language=language)

    counts = dict(sorted(Counter(s.kind for s in syms).items()))
    naming: dict[str, Naming] = {}
    coverage: dict[str, float] = {}
    for group, kinds in _GROUPS.items():
        members = [s for s in syms if s.kind in kinds]
        if not members:
            continue
        dist = Counter(name_convention(s.name) for s in members)
        top, n = dist.most_common(1)[0]
        naming[group] = Naming(convention=top, rate=n / len(members), distribution=dict(dist))
        coverage[group] = sum(1 for s in members if s.docstring) / len(members)

    callables = [s for s in syms if s.kind in _CALLABLE]
    ann = [_has_return_annotation(s.signature, language) for s in callables]
    ann_known = [a for a in ann if a is not None]
    return_rate = (sum(a for a in ann_known) / len(ann_known)) if ann_known else None

    lines = [s.end_line - s.start_line + 1 for s in callables]
    n_types = sum(1 for s in syms if s.kind in _TYPE)
    n_methods = sum(1 for s in syms if s.kind == "method")
    return StyleGuide(
        language=language, counts=counts, naming=naming, docstring_coverage=coverage,
        return_annotation_rate=return_rate,
        avg_lines=(sum(lines) / len(lines)) if lines else 0.0,
        median_lines=float(statistics.median(lines)) if lines else 0.0,
        avg_methods_per_type=(n_methods / n_types) if n_types else 0.0,
        total=len(syms),
    )


def derive_all(symbols: Sequence[Symbol]) -> list[StyleGuide]:
    """One :class:`StyleGuide` per language present, most-populated language first."""
    langs = {s.language for s in symbols}
    guides = [derive_style(symbols, lang) for lang in langs]
    return sorted(guides, key=lambda g: (-g.total, g.language))


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def _guide_section(g: StyleGuide) -> str:
    if not g.counts:
        return (f'<section class="acard"><h2>{escape(g.language)}</h2>'
                '<p class="stats">no symbols</p></section>')
    mix = " · ".join(f"{n} {escape(k)}" for k, n in g.counts.items())
    rows = []
    for group, label in (("callable", "functions &amp; methods"), ("type", "classes &amp; types")):
        nm = g.naming.get(group)
        if nm is None:
            continue
        spread = ", ".join(f"{escape(c)} {n}" for c, n in
                           sorted(nm.distribution.items(), key=lambda kv: -kv[1]))
        cov = _pct(g.docstring_coverage.get(group, 0.0))
        rows.append(
            f"<tr><td>{label}</td><td><b>{escape(nm.convention)}</b> ({_pct(nm.rate)})</td>"
            f"<td>{cov}</td><td>{escape(spread)}</td></tr>"
        )
    ann = "n/a" if g.return_annotation_rate is None else _pct(g.return_annotation_rate)
    return (
        f'<section class="acard"><h2>{escape(g.language)} — {g.total} symbols</h2>'
        f'<p class="stats">{mix} · return-typed callables {ann} · '
        f"avg {g.avg_lines:.0f} lines/callable (median {g.median_lines:.0f}) · "
        f"{g.avg_methods_per_type:.1f} methods/type</p>"
        '<table class="tbl"><thead><tr><th>group</th><th>naming (dominant)</th>'
        "<th>documented</th><th>distribution</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table></section>"
    )


def style_guide_html(guides: Sequence[StyleGuide]) -> str:
    """A themed, self-contained style-guide report — one section per language."""
    body = "".join(_guide_section(g) for g in guides) or '<p class="stats">no symbols</p>'
    langs = ", ".join(g.language for g in guides if g.counts) or "no languages"
    return report.themed_page("Style guide", body, subtitle=f"derived conventions · {langs}")
