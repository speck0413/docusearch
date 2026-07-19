"""Phase 5 — enrichment: pre-flight classification, gotchas, summaries, discrepancies (§17).

Pre-flight classification (R-ING-7): sample the corpus stratified by folder, ask Claude (temperature
0) to propose chunk rules + gotcha regexes, write them to ``preflight_rules.yaml`` for Stephen to
**approve before they run**. This module holds the deterministic machinery; the model call reuses
the temperature-0 Claude backend from ``vision.py``.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import yaml

GOTCHA_PREFIX = "[GOTCHA]"

Runner = Callable[[list[str]], tuple[int, str, str]]  # argv -> (returncode, stdout, stderr)


class EnrichError(Exception):
    """Pre-flight / enrichment failure with a one-line, actionable message."""


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


_PROPOSE_PROMPT = """You are proposing pre-flight ingestion rules for a documentation corpus. Below
are excerpts from a stratified sample of the documents. Study them and propose:
1. GOTCHA regexes — patterns that mark a passage as a caution/pitfall/deprecation an engineer must
   not miss (e.g. warnings, "do not", deprecations, known-issues). Prefer a few precise regexes over
   many loose ones. Each needs a short lowercase `label`.
2. NOTES — brief observations about the document structure (heading depth, code blocks, tables) that
   would help chunking. Plain text, a few sentences.

Reply with ONLY a JSON object: {"gotcha_patterns": [{"pattern": "<regex>", "label": "<label>"}, ...],
"notes": "<text>"} and no other prose.

--- SAMPLED DOCUMENT EXCERPTS ---
%s
"""


def propose_rules(
    doc_texts: list[str],
    *,
    model: str = "claude-opus-4-8",
    runner: Runner | None = None,
    cli: str = "claude",
    timeout: float = 300.0,
    excerpt_chars: int = 2000,
) -> PreflightRules:
    """Ask Claude to propose gotcha regexes + chunking notes from the sampled document text
    (R-ING-7). Returns an **unapproved** ``PreflightRules`` (Stephen reviews + approves the file
    before it runs). ``runner`` is injectable for tests; by default it shells out to the ``claude``
    CLI headless (``-p … --output-format json``) — the operator's Claude Code login, no API key."""
    excerpts = "\n\n---\n\n".join(t[:excerpt_chars] for t in doc_texts)
    argv = [cli, "-p", _PROPOSE_PROMPT % excerpts, "--model", model, "--output-format", "json"]

    def _default_runner(a: list[str]) -> tuple[int, str, str]:
        import subprocess  # lazy: only when actually calling Claude

        proc = subprocess.run(a, capture_output=True, text=True, timeout=timeout)  # noqa: S603
        return proc.returncode, proc.stdout, proc.stderr

    run = runner or _default_runner
    try:
        code, out, err = run(argv)
    except Exception as exc:  # noqa: BLE001 - missing binary / timeout / OSError
        raise EnrichError(f"claude CLI invocation failed: {type(exc).__name__}: {exc}") from exc
    if code != 0:
        raise EnrichError(f"claude CLI failed (exit {code}): {(err or out).strip()[:200]}")

    import contextlib

    text = out.strip()
    env = None
    with contextlib.suppress(json.JSONDecodeError):  # `--output-format json` -> result envelope
        env = json.loads(text)
    if isinstance(env, dict) and "result" in env:
        if env.get("is_error"):
            raise EnrichError(f"claude CLI returned an error: {str(env['result'])[:200]}")
        text = str(env["result"])
    try:
        payload = json.loads(_strip_code_fence(text))
    except json.JSONDecodeError as exc:
        raise EnrichError(f"could not parse rule proposal as JSON: {text[:200]}") from exc
    patterns = [
        GotchaPattern(str(g["pattern"]), str(g.get("label", "")))
        for g in (payload.get("gotcha_patterns") or [])
        if isinstance(g, dict) and g.get("pattern")
    ]
    return PreflightRules(
        approved=False, gotcha_patterns=patterns,
        notes=str(payload.get("notes", "")), sampled=len(doc_texts),
    )


def _strip_code_fence(text: str) -> str:
    """Peel a ```json … ``` fence if the model wrapped its JSON in one."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1] if "\n" in t else t
        if t.endswith("```"):
            t = t[: t.rfind("```")]
    return t.strip()


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
