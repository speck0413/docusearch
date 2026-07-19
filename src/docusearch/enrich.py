"""Phase 5 — enrichment: pre-flight classification, gotchas, summaries, discrepancies (§17).

Pre-flight classification (R-ING-7): sample the corpus stratified by folder, ask Claude (temperature
0) to propose chunk rules + gotcha regexes, write them to ``preflight_rules.yaml`` for the operator to
**approve before they run**. This module holds the deterministic machinery; the model call reuses
the temperature-0 Claude backend from ``vision.py``.
"""

from __future__ import annotations

import json
import random
import re
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from .config import Config, SourceConfig
    from .store import Store

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
    starts False — the rules do NOT run until the operator sets it true in the file."""

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


def _as_approved(value: object) -> bool:
    """Strict truthiness for the approval gate (R-ING-7, red-team H1): only a real YAML boolean
    ``true`` or an explicit true-ish string (``true``/``yes``/``on``) opens the gate. A hand-typed
    quoted ``"false"`` / ``"no"`` / ``0`` must NOT — plain ``bool("false")`` is ``True`` in Python,
    which would silently apply unapproved rules."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "on"}
    return False


def load_preflight_rules(path: Path | str) -> PreflightRules:
    """Parse ``preflight_rules.yaml``. Raises ``EnrichError`` (not a raw parser traceback) on a
    malformed file so a typo fails safe with a one-line, actionable message (red-team M1)."""
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        first = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
        raise EnrichError(f"{path} is not valid YAML: {first[:150]} — fix or remove it") from exc
    if not isinstance(data, dict):
        raise EnrichError(
            f"{path} must be a YAML mapping (approved:/gotcha_patterns:), got {type(data).__name__}"
        )
    return PreflightRules(
        approved=_as_approved(data.get("approved", False)),
        gotcha_patterns=[
            GotchaPattern(str(g["pattern"]), str(g.get("label", "")))
            for g in (data.get("gotcha_patterns") or [])
            if isinstance(g, dict) and g.get("pattern")
        ],
        notes=str(data.get("notes", "")),
        sampled=int(data.get("sampled", 0)),
    )


def _is_risky_regex(pattern: str) -> bool:
    """Fast static ReDoS pre-filter (red-team H2): flag a **nested quantifier** — a quantified group
    immediately followed by another quantifier, e.g. ``(a+)+``, ``(a*)*``, ``([a-z]+)+``, ``(.*)+``,
    ``(a+){2,}``. Catches the obvious catastrophic shape without spawning a probe; the authoritative
    check is the empirical ``_pattern_is_safe`` (which also catches ``(a|a)*b`` and friends)."""
    return (
        re.search(r"\([^()]*[+*][^()]*\)\s*[*+]", pattern) is not None
        or re.search(r"\([^()]*[+*][^()]*\)\{\d+,", pattern) is not None
    )


_REDOS_PROBE = (
    "import re,sys\n"
    "rx=re.compile(sys.argv[1], re.IGNORECASE)\n"
    "for s in ['a'*64+'!','0'*64+'!','ab'*40+'!',' '*64+'!','aaaa'*32+'X']:\n"
    "    rx.search(s)\n"
)


def _pattern_is_safe(pattern: str, *, timeout: float = 2.0) -> bool:
    """Empirically confirm a gotcha regex can't catastrophically backtrack: run it against
    adversarial probe strings in a **separate process** with a wall-clock timeout (killed on
    expiry). Returns False on timeout or error. Reliable where a static heuristic isn't — catches
    ``(a|a)*b`` etc. — and cross-platform (no signals/threads, Windows-safe, R-ARCH-5). Run once per
    pattern at load (a handful per ingest), never on the query path."""
    import subprocess  # lazy: only when validating rules
    import sys

    try:
        proc = subprocess.run(  # noqa: S603
            [sys.executable, "-c", _REDOS_PROBE, pattern],
            capture_output=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False
    except Exception:  # noqa: BLE001 - any spawn failure -> treat as unsafe, don't apply
        return False
    return proc.returncode == 0


def active_gotcha_patterns(path: Path | str) -> list[GotchaPattern]:
    """The gotcha patterns to apply at ingest — **empty** unless the rules file exists AND is
    approved (R-ING-7: proposed rules never run until the operator approves the file). A malformed file
    fails **safe**: warn loudly and apply no rules rather than abort the whole ingest (red-team M1).
    Patterns that are invalid or ReDoS-risky are dropped with a warning, never applied (H2)."""
    p = Path(path)
    if not p.is_file():
        return []
    try:
        rules = load_preflight_rules(p)
    except EnrichError as exc:
        warnings.warn(f"{exc} — applying no gotcha rules this ingest", stacklevel=2)
        return []
    if not rules.approved:
        return []
    safe: list[GotchaPattern] = []
    for g in rules.gotcha_patterns:
        try:
            re.compile(g.pattern)
        except re.error as exc:
            warnings.warn(f"skipping invalid gotcha regex {g.pattern!r}: {exc}", stacklevel=2)
            continue
        if _is_risky_regex(g.pattern) or not _pattern_is_safe(g.pattern):
            warnings.warn(
                f"skipping ReDoS-risky gotcha pattern {g.pattern!r} (catastrophic backtracking) — "
                "rewrite it without ambiguous/nested quantifiers",
                stacklevel=2,
            )
            continue
        safe.append(g)
    return safe


def match_gotcha(text: str, patterns: list[GotchaPattern]) -> str | None:
    """The label of the first gotcha pattern that matches ``text`` (case-insensitive), or None. A
    malformed regex is skipped, and a ReDoS-risky pattern is skipped rather than run (R-ING-8,
    red-team H2), so even a direct call can't hang."""
    for g in patterns:
        if _is_risky_regex(g.pattern):
            continue
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
    (R-ING-7). Returns an **unapproved** ``PreflightRules`` (the operator reviews + approves the file
    before it runs). ``runner`` is injectable for tests; by default it shells out to the ``claude``
    CLI headless (``-p … --output-format json``) — the operator's Claude Code login, no API key."""
    excerpts = "\n\n---\n\n".join(t[:excerpt_chars] for t in doc_texts)
    text = _claude_text(
        _PROPOSE_PROMPT % excerpts, model=model, runner=runner, cli=cli, timeout=timeout
    )
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


def _claude_text(
    prompt: str, *, model: str, runner: Runner | None, cli: str, timeout: float
) -> str:
    """Run a headless ``claude -p … --output-format json`` call and return the model's result text.
    ``runner`` is injectable for tests; the default shells out to the ``claude`` CLI — the
    operator's Claude Code login, no API key. Raises ``EnrichError`` on any failure."""
    import contextlib

    argv = [cli, "-p", prompt, "--model", model, "--output-format", "json"]

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
    text = out.strip()
    env = None
    with contextlib.suppress(json.JSONDecodeError):  # `--output-format json` -> result envelope
        env = json.loads(text)
    if isinstance(env, dict) and "result" in env:
        if env.get("is_error"):
            raise EnrichError(f"claude CLI returned an error: {str(env['result'])[:200]}")
        text = str(env["result"])
    return text


_SUMMARIZE_PROMPT = """Summarize this documentation page for search and quick reference. Write 2-4
plain-text sentences, no preamble or markdown — capture what the page covers and any key specifics
(part numbers, procedures, cautions). Reply with ONLY the summary.

--- DOCUMENT ---
%s
"""


def summarize_document(
    text: str,
    *,
    model: str = "claude-opus-4-8",
    runner: Runner | None = None,
    cli: str = "claude",
    timeout: float = 300.0,
    excerpt_chars: int = 6000,
) -> str:
    """A concise, searchable AI summary of one document (§17 optional AI summaries). Called only at
    enrichment time and persisted (determinism by persistence, R-SRCH-5). Raises ``EnrichError`` on
    failure so the caller can skip that doc and continue."""
    summary = _claude_text(
        _SUMMARIZE_PROMPT % text[:excerpt_chars],
        model=model, runner=runner, cli=cli, timeout=timeout,
    ).strip()
    if not summary:
        raise EnrichError("claude returned an empty summary")
    return summary


def _strip_code_fence(text: str) -> str:
    """Peel a ```json … ``` fence if the model wrapped its JSON in one."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1] if "\n" in t else t
        if t.endswith("```"):
            t = t[: t.rfind("```")]
    return t.strip()


@dataclass(frozen=True)
class DuplicateGroup:
    """A set of ACTIVE documents that are byte-identical (same content_hash)."""

    content_hash: str
    docs: list[tuple[int, str]]  # (doc_id, path)


@dataclass(frozen=True)
class ConflictPair:
    """Two chunks in DIFFERENT documents that are highly similar but not identical — a candidate
    for conflicting/duplicated guidance a human should reconcile."""

    chunk_a: int
    chunk_b: int
    doc_a: int
    doc_b: int
    similarity: float


@dataclass
class DiscrepancyReport:
    duplicate_actives: list[DuplicateGroup] = field(default_factory=list)
    conflict_candidates: list[ConflictPair] = field(default_factory=list)


def scan_discrepancies(
    store: Store,
    *,
    vector_index: object | None = None,
    sim_lo: float = 0.90,
    sim_hi: float = 0.999,
    neighbors: int = 6,
    limit: int = 200,
) -> DiscrepancyReport:
    """Scan the index for discrepancies (§17): (1) duplicate ACTIVE documents (same content_hash);
    (2) high-similarity **conflict candidates** — chunk pairs across different documents whose
    cosine similarity falls in ``[sim_lo, sim_hi)`` (near-identical excluded — those are dupes, not
    conflicts). Conflict detection needs embeddings + a ``vector_index`` (``.query(vec, k)``); a
    BM25-only index reports duplicate actives only. Deterministic: ordered by (−sim, chunk ids)."""
    dups = [DuplicateGroup(h, docs) for h, docs in store.duplicate_active_documents()]
    pairs: list[ConflictPair] = []
    if vector_index is not None and store.count_embeddings() > 0:
        import numpy as np

        chunk_doc = store.chunk_doc_map()
        seen: set[tuple[int, int]] = set()
        for cid, blob in store.all_embeddings():
            vec = np.frombuffer(blob, dtype=np.float32)
            for nid, sim in vector_index.query(vec, neighbors + 1):  # type: ignore[attr-defined]
                if nid == cid or not (sim_lo <= sim < sim_hi):
                    continue
                da, db = chunk_doc.get(cid), chunk_doc.get(nid)
                if da is None or db is None or da == db:
                    continue  # same-doc near-dupes aren't cross-document conflicts
                key = (min(cid, nid), max(cid, nid))
                if key in seen:
                    continue
                seen.add(key)
                a, b = key
                pairs.append(ConflictPair(a, b, chunk_doc[a], chunk_doc[b], round(float(sim), 4)))
        pairs.sort(key=lambda p: (-p.similarity, p.chunk_a, p.chunk_b))
        pairs = pairs[:limit]
    return DiscrepancyReport(duplicate_actives=dups, conflict_candidates=pairs)


def persist_discrepancies(store: Store, report: DiscrepancyReport) -> int:
    """Write the scan's findings as ``flags`` rows (kind=discrepancy), replacing any prior scan's.
    Returns the number of flags written. Duplicate actives flag each doc; conflict candidates flag
    both chunks with the peer + similarity in the note."""
    store.clear_flags("discrepancy")
    written = 0
    for g in report.duplicate_actives:
        peers = ", ".join(str(d) for d, _ in g.docs)
        for doc_id, _ in g.docs:
            store.add_flag(
                doc_id=doc_id, chunk_id=None, kind="discrepancy", source="scan",
                rule_id="duplicate-active", note=f"identical content to docs [{peers}]",
            )
            written += 1
    for p in report.conflict_candidates:
        for cid, other, doc in (
            (p.chunk_a, p.chunk_b, p.doc_a),
            (p.chunk_b, p.chunk_a, p.doc_b),
        ):
            store.add_flag(
                doc_id=doc, chunk_id=cid, kind="discrepancy", source="scan",
                rule_id="near-duplicate", note=f"~{p.similarity:.3f} cosine to chunk {other}",
            )
            written += 1
    return written


def run_preflight(
    config: Config,
    *,
    out_path: Path | str,
    model: str = "claude-opus-4-8",
    runner: Runner | None = None,
    cli: str = "claude",
    seed: int = 0,
) -> PreflightRules:
    """Pre-flight classification end to end (R-ING-7): sample the configured sources stratified by
    folder (``enrich.preflight_sample`` docs), extract their text, ask Claude to propose rules, and
    write an **unapproved** ``preflight_rules.yaml`` — nothing takes effect until the operator reviews the
    file and sets ``approved: true``."""
    from .ingest import extract_document, iter_files  # lazy: keeps enrich import light

    by_path: dict[Path, SourceConfig] = {}
    for source in config.sources:
        for p in iter_files(source.location, source.include, source.exclude):
            by_path.setdefault(p, source)
    sample = stratified_sample(sorted(by_path), config.enrich.preflight_sample, seed=seed)

    texts: list[str] = []
    for p in sample:
        source = by_path[p]
        try:
            doc = extract_document(
                p, p.suffix.lstrip(".").lower(),
                content_selector=source.content_selector,
                strip_selectors=source.strip_selectors,
            )
            texts.append("\n".join(s.text for s in doc.segments))
        except Exception:  # noqa: BLE001 - one bad file must not abort the whole proposal
            continue
    rules = propose_rules(texts, model=model, runner=runner, cli=cli)
    write_preflight_rules(rules, out_path)
    return rules


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
