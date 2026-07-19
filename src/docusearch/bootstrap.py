"""`docusearch bootstrap` (task #35) — scan a mixed repo and emit a starter ``docusearch.yaml``.

Onboarding aid: point it at a folder (or repo), and it categorises the files (docs / code / data),
recommends a ``store_type``, and writes a valid, commented config with the right ``include`` globs and
inline **hints** — run ``docusearch inspect`` to tune HTML selectors, the PDF font profile (task #34)
for how headings will be inferred, the code languages found, and a note for any secondary content that
would suit a separate federated store. Entirely generic — no corpus-specific knowledge.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from . import code_index

# Extensions that mean "prose document" vs "structured data" (code is detected via code_index).
_DOC_EXT = frozenset({".html", ".htm", ".pdf", ".docx", ".md", ".markdown", ".pptx", ".xlsx"})
_DATA_EXT = frozenset({".csv", ".tsv", ".psv", ".tab", ".stdf", ".std"})
# Directories never worth scanning (VCS / build / vendor / caches).
_SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", "dist", "build", ".tox", ".idea", ".vscode", "target",
})


@dataclass
class RepoScan:
    """What a repo contains, by category / language / extension."""

    root: Path
    counts: dict[str, int] = field(default_factory=dict)          # doc | code | data | other
    code_languages: dict[str, int] = field(default_factory=dict)  # language -> file count
    extensions: dict[str, int] = field(default_factory=dict)      # ext (no dot) -> count
    total: int = 0


def _category(path: Path) -> tuple[str, str | None]:
    """(category, language) for a file: category ∈ doc|code|data|other; language for code files."""
    lang = code_index.detect_language(path.name)
    if lang is not None:
        return "code", lang
    ext = path.suffix.lower()
    if ext in _DATA_EXT:
        return "data", None
    if ext in _DOC_EXT:
        return "doc", None
    return "other", None


def scan_repo(root: Path, *, max_files: int = 50000) -> RepoScan:
    """Walk ``root`` (skipping VCS/build/vendor dirs) and categorise its files."""
    counts: Counter[str] = Counter()
    langs: Counter[str] = Counter()
    exts: Counter[str] = Counter()
    seen = 0
    for path in root.rglob("*"):
        if seen >= max_files:
            break
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        if not path.is_file():
            continue
        seen += 1
        cat, lang = _category(path)
        counts[cat] += 1
        if path.suffix:
            exts[path.suffix.lstrip(".").lower()] += 1
        if lang:
            langs[lang] += 1
    return RepoScan(root=root, counts=dict(counts), code_languages=dict(langs),
                    extensions=dict(exts), total=seen)


def recommend_store_type(scan: RepoScan) -> str:
    """Pick the store_type for the dominant content: code | data | document (ties → document)."""
    code = scan.counts.get("code", 0)
    data = scan.counts.get("data", 0)
    doc = scan.counts.get("doc", 0)
    if code and code >= data and code > doc:
        return "code"
    if data and data > doc:
        return "data"
    return "document"


# The include globs each store_type indexes, from the extensions actually present.
_STORE_EXTS: dict[str, frozenset[str]] = {
    "code": frozenset(_EXT.lstrip(".") for _EXT in code_index._EXT),  # noqa: SLF001
    "data": frozenset(e.lstrip(".") for e in _DATA_EXT),
    "document": frozenset(e.lstrip(".") for e in _DOC_EXT),
}


def _globs_for(store_type: str, scan: RepoScan) -> list[str]:
    wanted = _STORE_EXTS[store_type]
    present = [e for e in scan.extensions if e in wanted]
    return [f"*.{e}" for e in sorted(present)] or ["*"]


def bootstrap_config(root: Path, *, name: str | None = None) -> str:
    """A valid, commented starter ``docusearch.yaml`` for the repo at ``root``."""
    scan = scan_repo(root)
    store_type = recommend_store_type(scan)
    src_name = name or (root.resolve().name or "source")
    globs = _globs_for(store_type, scan)
    inc = ", ".join(f'"{g}"' for g in globs)

    mix = ", ".join(f"{n} {cat}" for cat, n in sorted(scan.counts.items(), key=lambda kv: -kv[1]) if n)
    hints: list[str] = []
    if store_type == "document" and any(e in scan.extensions for e in ("html", "htm")):
        hints.append(f"# HTML present — run `docusearch inspect {src_name}` to fill "
                     "content_selector / strip_selectors.")
    if store_type == "document" and "pdf" in scan.extensions:
        prof = _pdf_hint(root)
        if prof:
            hints.append(f"# {prof}")
    if store_type == "code" and scan.code_languages:
        langs = ", ".join(sorted(scan.code_languages))
        hints.append(f"# code languages detected: {langs}. A git URL as `location` is cloned for you.")
    primary_cat = {"document": "doc", "code": "code", "data": "data"}[store_type]
    secondary = [f"{n} {cat}" for cat, n in sorted(scan.counts.items())
                 if cat in ("doc", "code", "data") and cat != primary_cat and n]
    if secondary:
        hints.append(f"# also found {', '.join(secondary)} file(s) — for a different store_type, "
                     "add a separate store and combine them under a `federation:` (see the RUNBOOK).")
    hint_block = ("\n" + "\n".join(f"    {h}" for h in hints)) if hints else ""

    return (
        f"# Generated by `docusearch bootstrap` from {root} — review before use.\n"
        f"# Detected: {mix or 'no indexable files'} ({scan.total} files scanned).\n"
        f'store_type: "{store_type}"\n'
        "paths:\n"
        '  staging_dir: "./staging"\n'
        '  db_path: "./docusearch.db"\n'
        '  tmp_dir: "./tmp"\n'
        "sources:\n"
        f"  - name: {src_name}\n"
        f'    location: "{root.as_posix()}"\n'
        f"    include: [{inc}]\n"
        "    min_content_chars: 1"
        f"{hint_block}\n"
        "embed:\n"
        '  model: "auto"\n'
    )


def _pdf_hint(root: Path) -> str:
    """A one-line PDF font-profile hint (task #34) from a small sample of the repo's PDFs."""
    from . import ingest

    pdfs = [p for p in root.rglob("*.pdf")
            if not any(part in _SKIP_DIRS for part in p.parts)][:5]
    if not pdfs:
        return ""
    try:
        prof = ingest.pdf_font_profile([p.read_bytes() for p in pdfs])
    except Exception:  # noqa: BLE001 - a hint is best-effort, never fatal to bootstrap
        return ""
    if not prof.detected:
        return f"PDFs: uniform ~{prof.body_size:g}pt font — headings fall back to 'page N'."
    mapping = ", ".join(f"{size:g}pt→H{lvl}" for size, lvl in sorted(prof.levels.items(),
                                                                     key=lambda kv: kv[1]))
    return f"PDF headings inferred from font size: {mapping} (body ~{prof.body_size:g}pt)."
