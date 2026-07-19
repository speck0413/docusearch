"""Phase 5 — enrichment: pre-flight classification, gotchas, summaries, discrepancies (§17).

Pre-flight classification (R-ING-7): sample the corpus stratified by folder, ask Claude (temperature
0) to propose chunk rules + gotcha regexes, write them to ``preflight_rules.yaml`` for Stephen to
**approve before they run**. This module holds the deterministic machinery; the model call reuses
the temperature-0 Claude backend from ``vision.py``.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

GOTCHA_PREFIX = "[GOTCHA]"


@dataclass(frozen=True)
class GotchaPattern:
    """A regex that marks a chunk as a *gotcha* (R-ING-8), plus a short label for the flag."""

    pattern: str
    label: str


@dataclass
class PreflightRules:
    """The pre-flight proposal (R-ING-7): gotcha regexes + free-text chunking notes. ``approved``
    starts False — the rules do NOT run until Stephen sets it true in the file."""

    approved: bool
    gotcha_patterns: list[GotchaPattern] = field(default_factory=list)
    notes: str = ""
    sampled: int = 0


_PREFLIGHT_HEADER = (
    "# Pre-flight classification (R-ING-7) — proposed by Claude at temperature 0.\n"
    "# REVIEW these, then set `approved: true` to let them run at ingest. Edit freely.\n"
    "# gotcha_patterns: a matching chunk gets a [GOTCHA] prefix (BM25-visible) + a flags row (R-ING-8).\n"
)


def write_preflight_rules(rules: PreflightRules, path: Path | str) -> None:
    """Write ``preflight_rules.yaml`` — commented, human-editable, ``approved: false`` by default."""
    target = Path(path)
    if target.parent != Path():
        target.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "approved": rules.approved,
        "sampled": rules.sampled,
        "notes": rules.notes,
        "gotcha_patterns": [{"pattern": g.pattern, "label": g.label} for g in rules.gotcha_patterns],
    }
    target.write_text(
        _PREFLIGHT_HEADER + yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def load_preflight_rules(path: Path | str) -> PreflightRules:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return PreflightRules(
        approved=bool(data.get("approved", False)),
        gotcha_patterns=[
            GotchaPattern(str(g["pattern"]), str(g.get("label", "")))
            for g in (data.get("gotcha_patterns") or [])
            if isinstance(g, dict) and g.get("pattern")
        ],
        notes=str(data.get("notes", "")),
        sampled=int(data.get("sampled", 0)),
    )


def active_gotcha_patterns(path: Path | str) -> list[GotchaPattern]:
    """The gotcha patterns to apply at ingest — **empty** unless the rules file exists AND is
    approved (R-ING-7: proposed rules never run until Stephen approves the file)."""
    p = Path(path)
    if not p.is_file():
        return []
    rules = load_preflight_rules(p)
    return rules.gotcha_patterns if rules.approved else []


def match_gotcha(text: str, patterns: list[GotchaPattern]) -> str | None:
    """The label of the first gotcha pattern that matches ``text`` (case-insensitive), or None. A
    malformed regex is skipped, not fatal (R-ING-8)."""
    for g in patterns:
        try:
            if re.search(g.pattern, text, re.IGNORECASE):
                return g.label
        except re.error:
            continue
    return None


def gotcha_tag_text(text: str) -> str:
    """Prefix a chunk's text with the BM25-visible ``[GOTCHA]`` marker (R-ING-8). Idempotent."""
    return text if text.startswith(GOTCHA_PREFIX) else f"{GOTCHA_PREFIX} {text}"


def stratified_sample(paths: list[Path], n: int, *, seed: int) -> list[Path]:
    """Pick ``n`` paths **stratified by parent folder** so every folder is represented (a big folder
    can't crowd out a small one) — the sample that drives rule proposal (R-ING-7). Deterministic for
    a given ``seed``: the per-folder allocation is fixed (proportional, largest-remainder, ≥1 each),
    and the within-folder choice is a seeded shuffle. Returns all paths when ``n`` exceeds the corpus."""
    ordered = sorted(paths)
    if n >= len(ordered):
        return ordered

    groups: dict[Path, list[Path]] = {}
    for p in ordered:
        groups.setdefault(p.parent, []).append(p)
    folders = sorted(groups)
    rng = random.Random(seed)

    if n < len(folders):  # can't cover every folder — sample n folders, one doc each
        chosen = sorted(rng.sample(folders, n), key=str)
        return sorted(sorted(groups[f], key=str)[0] for f in chosen)

    # ≥1 per folder for coverage, then distribute the remainder proportionally to spare capacity
    alloc = {f: 1 for f in folders}
    remaining = n - len(folders)
    weights = {f: len(groups[f]) - 1 for f in folders}
    wtotal = sum(weights.values())
    if remaining and wtotal:
        exact = {f: remaining * weights[f] / wtotal for f in folders}
        for f in folders:
            alloc[f] += int(exact[f])
        left = remaining - sum(int(exact[f]) for f in folders)
        for f in sorted(folders, key=lambda f: (-(exact[f] - int(exact[f])), str(f)))[:left]:
            alloc[f] += 1
    for f in folders:  # never ask for more than the folder holds
        alloc[f] = min(alloc[f], len(groups[f]))

    picked: list[Path] = []
    for f in folders:
        items = sorted(groups[f])
        rng.shuffle(items)
        picked.extend(items[: alloc[f]])
    return sorted(picked)
