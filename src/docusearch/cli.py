"""Command-line interface — a thin front-end over the library (§4).

Every command resolves the config, does one job, and logs one event.

    docusearch init [--config PATH] [--force]
    docusearch ingest [--dry-run] [--force] [--reembed] [--config PATH]
    docusearch audit [--config PATH]
    docusearch remove <source> [--yes] [--config PATH]
    docusearch models
    docusearch inspect [<source>] [--sample N] [--config PATH]
    docusearch search <query> [--top-k N] [--prefix] [--json] [--batch-file F --out O] [--config PATH]
    docusearch report --spec SPEC.yaml [--format md|html] [--out FILE] [--config PATH]
    docusearch show <doc_id> [--config PATH]
    docusearch serve [--host H] [--port P] [--config PATH]
    docusearch gate <n> [--name NAME] [--config PATH]

The console entry point is ``main`` (see pyproject ``[project.scripts]``).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, TextIO

import yaml

from . import config, embed, enrich, ingest, report, runlog
from ._version import __version__
from .catalog import Catalog, open_federation
from .config import Config, ConfigError
from .mcp_client import MCPClient, MCPError
from .search import SearchHit, roles_from_env
from .store import Store, StoreError
from .vision import VisionError


def _configure_logging(cfg: Config) -> None:
    runlog.configure(
        Path(cfg.paths.tmp_dir) / "logs",
        level=cfg.logging.level,
        enabled=cfg.logging.jsonl,
    )


class _ProgressBar:
    """Render ``(phase, done, total)`` callbacks to stderr.

    On a TTY it redraws one line in place (``\\r``); when output is piped/redirected it
    prints a line each time it crosses a 10% boundary, so logs stay readable instead of
    scrolling thousands of lines. Long-running work (embedding on GPU) finally has a
    heartbeat so it's obvious the process is alive, not hung."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream if stream is not None else sys.stderr
        self._isatty = self._stream.isatty()
        self._last_decile: dict[str, int] = {}

    def __call__(self, phase: str, done: int, total: int) -> None:
        if total <= 0:
            return
        pct = done * 100 // total
        if self._isatty:
            end = "\n" if done >= total else ""
            self._stream.write(f"\r  {phase}: {done}/{total} ({pct}%)      {end}")
            self._stream.flush()
            return
        decile = pct // 10
        if decile != self._last_decile.get(phase) or done >= total:
            self._last_decile[phase] = decile
            self._stream.write(f"  {phase}: {done}/{total} ({pct}%)\n")
            self._stream.flush()


def _cmd_init(args: argparse.Namespace) -> int:
    path = Path(args.config)
    written = config.write_template(path, force=args.force)
    if written:
        print(f"Wrote config template to {path}")
    else:
        print(f"Config already exists at {path} (use --force to overwrite)")
    cfg = config.load(path)
    _configure_logging(cfg)
    runlog.log("cli.init", config=str(path), created=written)
    runlog.flush()
    return 0


def _cmd_ingest(args: argparse.Namespace) -> int:
    cfg = config.load(Path(args.config))
    _configure_logging(cfg)
    if args.dry_run:
        print(f"Ingest plan (dry run) — mode: {cfg.mode}")
        if not cfg.sources:
            print("  (no sources configured)")
        for src in cfg.sources:
            print(f"  source {src.name!r} [{src.type}] @ {src.location}")
            print(f"    include={src.include} exclude={src.exclude}")
            print(
                f"    content_selector={src.content_selector!r} "
                f"strip_selectors={src.strip_selectors} "
                f"min_content_chars={src.min_content_chars}"
            )
            print(f"    audience={src.audience}")
        runlog.log("cli.ingest.dryrun", sources=[s.name for s in cfg.sources])
        runlog.flush()
        return 0

    log_path = runlog.active_log_path()
    if log_path is not None:
        print(f"Logging to {log_path} (tail -f to watch)", file=sys.stderr)
    if args.force:
        print("Full rebuild (--force): re-parsing files and re-embedding.", file=sys.stderr)
    elif args.reembed:
        print("Re-embedding: dropping existing vectors first.", file=sys.stderr)
    result = Catalog(cfg).ingest(force=args.force, reembed=args.reembed, progress=_ProgressBar())
    reports = Path(cfg.paths.tmp_dir) / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    report_path = reports / f"ingest-audit-{runlog.RUN_ID}.md"
    report_path.write_text(
        ingest.render_ingest_audit(result, run_id=runlog.RUN_ID), encoding="utf-8"
    )
    unresolved = (
        f", {result.images_unresolved} img refs unresolved" if result.images_unresolved else ""
    )
    print(
        f"Ingested {result.documents} docs, {result.chunks} chunks, {result.images} images "
        f"({result.skipped_unchanged} unchanged, {result.stripped_empty} too short, "
        f"{result.excluded_glob} excluded{unresolved}); embedded {result.embedded} chunks."
    )
    print(f"Audit report: {report_path}")
    runlog.log("cli.ingest", documents=result.documents, chunks=result.chunks)
    runlog.flush()
    return 0


def _cmd_vision(args: argparse.Namespace) -> int:
    """Enrich retained images with cloud OCR + description (enrich.vision_images)."""
    cfg = config.load(Path(args.config))
    _configure_logging(cfg)
    if not cfg.enrich.vision_images:
        raise config.ConfigError(
            "enrich.vision_images is off — set `enrich.vision_images: true` in your config "
            "to run image vision (it calls a paid cloud API)."
        )
    with Store.open(cfg.paths.db_path) as store:
        pending = len(store.images_needing_vision(limit=args.limit))
    if pending == 0:
        print("No images need vision enrichment (all retained images already enriched).")
        return 0
    print(
        f"{pending} images will be sent to {cfg.enrich.vision_model!r} — a paid cloud API. "
        "Auth: ANTHROPIC_API_KEY or an `ant auth login` profile.",
        file=sys.stderr,
    )
    if not args.yes and sys.stdin.isatty():
        confirm = input(f"Enrich {pending} images? [y/N] ").strip().lower()
        if confirm not in {"y", "yes"}:
            print("Aborted.")
            return 1
    log_path = runlog.active_log_path()
    if log_path is not None:
        print(f"Logging to {log_path} (tail -f to watch)", file=sys.stderr)
    result = Catalog(cfg).enrich_vision(
        limit=args.limit, by_size=args.largest, progress=_ProgressBar()
    )
    print(
        f"Enriched {result.enriched} images "
        f"({result.skipped} unsupported type, {result.failed} failed)."
    )
    for sha, msg in result.errors[:10]:
        print(f"  ! {sha[:12]}…: {msg}", file=sys.stderr)
    runlog.log("cli.vision", enriched=result.enriched, failed=result.failed)
    runlog.flush()
    # total failure (e.g. the `claude` binary is missing → every image failed) exits non-zero
    # so automation notices; a partial failure still exits 0 with the count reported above.
    return 1 if result.enriched == 0 and result.failed > 0 else 0


def _cmd_summarize(args: argparse.Namespace) -> int:
    """Generate a searchable AI summary per document (enrich.ai_summaries; off by default)."""
    cfg = config.load(Path(args.config))
    _configure_logging(cfg)
    if not cfg.enrich.ai_summaries:
        raise config.ConfigError(
            "enrich.ai_summaries is off — set `enrich.ai_summaries: true` in your config to "
            "generate AI summaries (each doc is sent once to the `claude` CLI)."
        )
    with Store.open(cfg.paths.db_path) as store:
        pending = len(store.documents_needing_summary(args.limit or 0))
    if pending == 0:
        print("No documents need summaries (all active docs already summarized).")
        return 0
    print(
        f"{pending} documents will be summarized by {args.model!r} via the `claude` CLI "
        "(your Claude Code login, no API key).",
        file=sys.stderr,
    )
    result = Catalog(cfg).enrich_summaries(
        model=args.model, limit=args.limit, progress=_ProgressBar()
    )
    print(
        f"Summarized {result.summarized} documents "
        f"({result.skipped} empty, {result.failed} failed)."
    )
    for doc_id, msg in result.errors[:10]:
        print(f"  ! doc {doc_id}: {msg}", file=sys.stderr)
    runlog.log("cli.summarize", summarized=result.summarized, failed=result.failed)
    runlog.flush()
    return 1 if result.summarized == 0 and result.failed > 0 else 0


def _cmd_preflight(args: argparse.Namespace) -> int:
    """Pre-flight classification (R-ING-7): sample the corpus, ask Claude (temp 0) to propose
    gotcha rules, write an UNAPPROVED preflight_rules.yaml for review."""
    cfg = config.load(Path(args.config))
    _configure_logging(cfg)
    out_path = Path(args.out) if args.out else Path(cfg.enrich.preflight_rules)

    if out_path.is_file():  # never silently blow away rules you've already approved
        existing = enrich.load_preflight_rules(out_path)
        if existing.approved and not args.yes:
            print(
                f"{out_path} already exists and is APPROVED ({len(existing.gotcha_patterns)} "
                "rules). Re-running would replace it with a fresh, unapproved proposal.\n"
                "Re-run with --yes to overwrite, or pass --out to write elsewhere.",
                file=sys.stderr,
            )
            return 1

    print(
        f"Sampling up to {cfg.enrich.preflight_sample} docs (stratified by folder) and asking "
        f"{args.model!r} to propose gotcha rules — uses your Claude Code login, no API key.",
        file=sys.stderr,
    )
    try:
        rules = enrich.run_preflight(
            cfg, out_path=out_path, model=args.model, seed=args.seed
        )
    except enrich.EnrichError as exc:
        print(f"preflight failed: {exc}", file=sys.stderr)
        return 1

    print(
        f"Proposed {len(rules.gotcha_patterns)} gotcha rule(s) from {rules.sampled} sampled docs "
        f"→ {out_path}"
    )
    for g in rules.gotcha_patterns:
        print(f"  · [{g.label}] /{g.pattern}/")
    print(
        f"\nReview {out_path}, edit as needed, then set `approved: true` to apply the rules at "
        "the next `docusearch ingest`. Nothing runs until you approve.",
        file=sys.stderr,
    )
    runlog.log("cli.preflight", sampled=rules.sampled, rules=len(rules.gotcha_patterns))
    runlog.flush()
    return 0


def _cmd_audit(args: argparse.Namespace) -> int:
    cfg = config.load(Path(args.config))
    _configure_logging(cfg)
    text = Catalog(cfg).audit()
    reports = Path(cfg.paths.tmp_dir) / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    out = reports / f"audit-{runlog.RUN_ID}.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    runlog.log("cli.audit", report=str(out))
    runlog.flush()
    return 0


def _cmd_remove(args: argparse.Namespace) -> int:
    """Purge everything ingested under a source label (delete_me_next -> gone)."""
    cfg = config.load(Path(args.config))
    _configure_logging(cfg)
    name = args.source
    with Store.open(cfg.paths.db_path) as store:
        count = len(store.document_ids_for_source(name))
        known = store.source_names()
    if count == 0:
        print(f"No documents found for source {name!r}.")
        if known:
            listing = ", ".join(f"{n or '(blank)'} ({c})" for n, c in known)
            print(f"Known sources: {listing}")
        return 0
    if not args.yes and sys.stdin.isatty():
        confirm = input(f"Remove {count} documents for source {name!r}? [y/N] ").strip().lower()
        if confirm not in {"y", "yes"}:
            print("Aborted.")
            return 1
    removed = Catalog(cfg).remove_source(name)
    print(f"Removed {removed} documents (chunks, embeddings, relations, images) for {name!r}.")
    runlog.log("cli.remove", source=name, removed=removed)
    runlog.flush()
    return 0


def _cmd_prune(args: argparse.Namespace) -> int:
    """Remove documents whose source file no longer exists (e.g. after a folder rename)."""
    cfg = config.load(Path(args.config))
    _configure_logging(cfg)
    cat = Catalog(cfg)
    n = cat.prune_missing(apply=False)
    if n == 0:
        print("No documents with missing source files — nothing to prune.")
        return 0
    print(
        f"{n} documents reference source files that no longer exist "
        "(the source folder was likely moved or renamed, orphaning the originals)."
    )
    if not args.yes and sys.stdin.isatty():
        confirm = input(f"Remove these {n} orphaned documents? [y/N] ").strip().lower()
        if confirm not in {"y", "yes"}:
            print("Aborted.")
            return 1
    removed = cat.prune_missing(apply=True)
    print(f"Pruned {removed} orphaned documents (chunks, vectors, relations, images).")
    runlog.log("cli.prune", removed=removed)
    runlog.flush()
    return 0


def _hf_cache_dir() -> Path:
    """Where the Hugging Face hub caches downloaded embedding models (cross-platform)."""
    hub = os.environ.get("HF_HUB_CACHE")
    if hub:
        return Path(hub)
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _human(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num_bytes < 1024 or unit == "TB":
            return f"{num_bytes:.0f} {unit}" if unit == "B" else f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def _list_cached_models(cache: Path) -> list[tuple[str, str]]:
    """(repo_id, human size) for every model in the HF cache — via huggingface_hub when
    available (accurate, dedups blobs), else a plain directory scan."""
    try:  # huggingface_hub ships with sentence-transformers ([embeddings] extra)
        from huggingface_hub import scan_cache_dir

        info = scan_cache_dir(cache)  # honors the resolved cache dir
        rows = [
            (r.repo_id, r.size_on_disk_str) for r in sorted(info.repos, key=lambda r: r.repo_id)
        ]
        if rows:
            rows.append(("TOTAL", _human(info.size_on_disk)))
        return rows
    except Exception:  # noqa: BLE001 -- CacheNotFound / missing lib -> manual scan
        entries: list[tuple[str, int]] = []
        if cache.is_dir():
            for child in sorted(cache.iterdir()):
                if child.is_dir() and child.name.startswith("models--"):
                    size = sum(
                        f.stat().st_size
                        for f in child.rglob("*")
                        if f.is_file() and not f.is_symlink()  # skip snapshot symlinks
                    )
                    entries.append((child.name[len("models--") :].replace("--", "/"), size))
        rows = [(repo_id, _human(size)) for repo_id, size in entries]
        if rows:
            rows.append(("TOTAL", _human(sum(s for _, s in entries))))
        return rows


def _cmd_models(args: argparse.Namespace) -> int:
    """List downloaded embedding models and where to delete them (disk hygiene)."""
    cache = _hf_cache_dir()
    print(f"Model cache: {cache}")
    rows = _list_cached_models(cache)
    if rows:
        for repo_id, size in rows:
            print(f"  {repo_id:50s} {size:>10s}")
    else:
        print("  (empty — no models downloaded yet)")
    print("\nDelete a model you no longer use:")
    print("  huggingface-cli delete-cache     # interactive picker (needs huggingface_hub[cli])")
    print(f"  # …or delete its folder under {cache}")
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    """Sample a source and propose content_selector / strip_selectors for its shape."""
    import random

    from . import inspector

    cfg = config.load(Path(args.config))
    _configure_logging(cfg)
    src = next((s for s in cfg.sources if s.name == args.source), None)
    if src is None and args.source is None:
        src = cfg.sources[0] if cfg.sources else None
    if src is None:
        print(f"No source named {args.source!r}. Sources: {[s.name for s in cfg.sources]}")
        return 1
    files = list(ingest.iter_files(src.location, src.include, src.exclude))
    if not files:
        print(f"No files found for source {src.name!r} at {src.location}")
        return 1
    sample_n = args.sample if args.sample is not None else cfg.enrich.preflight_sample
    random.seed(0)  # deterministic sample
    picked = random.sample(files, min(sample_n, len(files)))
    docs = []
    for f in picked:
        with contextlib.suppress(OSError):
            docs.append(f.read_text("utf-8", errors="replace"))
    result = inspector.inspect_html(docs)

    print(f"Inspected {result.sampled} of {len(files)} files in source {src.name!r}\n")
    print("Body-container candidates  (selector: matched% / text-coverage%):")
    for sel, mr, cov in result.content_candidates:
        marker = "  <- suggested" if sel == result.content_selector else ""
        print(f"  {sel:26s} {mr * 100:4.0f}% / {cov * 100:4.0f}%{marker}")
    if not result.content_candidates:
        print("  (no common container matched; keep content_selector empty)")
    print("\nSuggested config for this source (paste into docusearch.yaml):")
    print(f'    content_selector: "{result.content_selector}"')
    if result.strip_selectors:
        print("    strip_selectors:")
        for s in result.strip_selectors:
            print(f'      - "{s}"')
    else:
        print("    strip_selectors: []")
    runlog.log("cli.inspect", source=src.name, sampled=result.sampled)
    runlog.flush()
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    cfg = config.load(Path(args.config))
    _configure_logging(cfg)
    stores = [s.strip() for s in (args.stores or "").split(",") if s.strip()]
    if cfg.federation:
        return _run_federated_search(cfg, args, stores)
    if stores:
        print("error: --stores needs a config with a 'federation:' section (single-store config).")
        return 2
    catalog = Catalog(cfg)
    if args.batch_file:
        return _run_batch(catalog, cfg, args)
    if not args.query:
        print("Provide a query, or --batch-file <goldens.yaml>.")
        return 2
    hits = catalog.search(args.query, top_k=args.top_k, prefix=args.prefix)
    banner = (
        f"({hits[0].search_mode} search; embed_model={hits[0].embed_model_used})" if hits else ""
    )
    _print_hits(args.query, hits, args.json, banner)
    runlog.log("cli.search", query=args.query, results=len(hits))
    runlog.flush()
    return 0


def _print_hits(query: str, hits: list[SearchHit], as_json: bool, banner: str) -> None:
    """Render search hits for the CLI — one path for single-store and federated results."""
    if as_json:
        print(json.dumps(_json_result(query, hits), ensure_ascii=False))
        return
    if not hits:
        print("No results.")
        return
    print(banner)
    for i, hit in enumerate(hits, 1):
        print(f"{i}. [{hit.citation}] {hit.title}  ({hit.locator})")
        print(f"   {hit.snippet}")
        print(f"   score={hit.score}  kind={hit.kind}  path={hit.path}")


def _run_federated_search(cfg: Config, args: argparse.Namespace, stores: list[str]) -> int:
    """Fan the query across the config's federation members (R-TEST-3), optionally scoped to the
    named subset in ``--stores`` (e.g. only internal)."""
    if args.batch_file:
        print("error: --batch-file is not supported with a federation; query one at a time.")
        return 2
    if not args.query:
        print("Provide a query to search the federation.")
        return 2
    k = args.top_k if args.top_k is not None else cfg.search.top_k_default
    try:
        with open_federation(cfg) as fed:
            available = fed.store_names()
            unknown = [s for s in stores if s not in available]
            if unknown:
                print(f"error: unknown store(s) {unknown}; available: {sorted(available)}")
                return 2
            hits = fed.search(
                args.query, top_k=k, prefix=args.prefix, roles=roles_from_env(),
                stores=stores or None,
            )
    except (ConfigError, StoreError) as err:
        print(f"error: {err}")
        return 2
    scope = ",".join(stores) if stores else f"all {len(cfg.federation)}"
    _print_hits(args.query, hits, args.json, f"(federated search; stores: {scope})")
    runlog.log("cli.search", query=args.query, results=len(hits), stores=stores or "all")
    runlog.flush()
    return 0


# Report-assembly helpers live in report.py so the CLI and the MCP/REST report builders share
# one implementation (R-REUSE-2).
_reference_targets = report.reference_targets
_evidence_images = report.evidence_images


def _cmd_report(args: argparse.Namespace) -> int:
    """Render a cited answer spec (YAML) to an md/html report, verifying every citation
    against the evidence the agent actually retrieved (refuses hallucinated references)."""
    from . import citations, report, report_export

    cfg = config.load(Path(args.config))
    _configure_logging(cfg)
    spec = yaml.safe_load(Path(args.spec).read_text(encoding="utf-8")) or {}
    evidence = {(int(d), int(c)) for d, c in spec.get("evidence", [])}
    out_str = str(args.out or "")
    _ext = out_str.rsplit(".", 1)[-1].lower() if "." in out_str else ""
    fmt = args.format or (_ext if _ext in ("html", "pdf", "docx", "pptx", "xlsx") else "md")
    base_url = f"http://localhost:{cfg.serve.port}"

    # Binary export formats (PDF/DOCX/PPTX/XLSX) — same citation guard, written as bytes (R-CIT-1).
    if fmt in report_export.EXPORT_FORMATS:
        if not args.out:
            print("error: --out is required for binary formats (pdf/docx/pptx/xlsx).", file=sys.stderr)
            return 1
        ref_targets = _reference_targets(cfg.paths.db_path, evidence)
        try:
            data = report_export.export_report(
                title=str(spec.get("title", "Report")), subtitle=str(spec.get("subtitle", "")),
                body=str(spec.get("body", "")), sections=spec.get("sections"), evidence=evidence,
                fmt=fmt, request=args.request or str(spec.get("request", "")),
                requested_by=args.requested_by or str(spec.get("requested_by", "")),
                model=args.model or str(spec.get("model", "")),
                classification=(args.classification if args.classification is not None
                                else str(spec.get("classification", "Confidential"))),
                ref_targets=ref_targets,
            )
        except citations.CitationError as err:
            print(f"error: {err}", file=sys.stderr)
            return 1
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        print(f"Wrote {fmt} report to {out}")
        runlog.log("cli.report", spec=str(args.spec), out=str(args.out), fmt=fmt)
        runlog.flush()
        return 0
    # header provenance: CLI flags win over spec fields (agents can pass either)
    sources = list(spec.get("sources", [])) or [s.name for s in cfg.sources]
    # References link to the ORIGINAL vendor document (file://), labelled "store — title —
    # heading", so the reader can open the parsed source, not an opaque chunk URL.
    ref_targets = _reference_targets(cfg.paths.db_path, evidence)
    # Embed any cited diagram directly in the report (self-contained; survives file moves).
    embedded_images = _evidence_images(cfg.paths.db_path, cfg.paths.staging_dir, evidence)
    try:
        rendered = report.render_report(
            title=str(spec.get("title", "Report")),
            subtitle=str(spec.get("subtitle", "")),
            body=str(spec.get("body", "")),
            sections=spec.get("sections"),
            evidence=evidence,
            base_url=base_url,
            fmt=fmt,
            run_id=runlog.RUN_ID,
            audience=list(spec.get("audience", [])),
            embed_model=cfg.embed.model,
            sources=sources,
            images=list(spec.get("images", [])),
            embedded_images=embedded_images,
            request=args.request or str(spec.get("request", "")),
            requested_by=args.requested_by or str(spec.get("requested_by", "")),
            model=args.model or str(spec.get("model", "")),
            classification=(
                args.classification
                if args.classification is not None
                else str(spec.get("classification", "Confidential"))
            ),
            ref_targets=ref_targets,
            trace=spec.get("trace"),
        )
    except citations.CitationError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rendered, encoding="utf-8")
        print(f"Wrote {fmt} report to {out}")
    else:
        print(rendered)
    runlog.log("cli.report", spec=str(args.spec), out=str(args.out or "-"), fmt=fmt)
    runlog.flush()
    return 0


def _json_result(query: str, hits: list[SearchHit]) -> dict[str, Any]:
    """One query's hits as a plain dict — the agent-facing search shape."""
    return {
        "query": query,
        "mode": hits[0].search_mode if hits else "bm25",
        "embed_model": hits[0].embed_model_used if hits else "none",
        "hits": [
            {
                "doc_id": h.doc_id,
                "chunk_id": h.chunk_id,
                "citation": h.citation,
                "title": h.title,
                "path": h.path,
                "locator": h.locator,
                "kind": h.kind,
                "score": h.score,
                "snippet": h.snippet,
            }
            for h in hits
        ],
    }


def _graded_pass(entry: dict[str, Any], hits: list[SearchHit]) -> bool | None:
    """PASS if any expected doc appears in the results; None when the golden is ungraded."""
    expect = entry.get("expect_docs") or []
    if not expect:
        return None
    paths = [h.path for h in hits]
    return any(any(str(exp) in p for p in paths) for exp in expect)


def _render_golden(entries: list[dict[str, Any]], results: list[list[SearchHit]]) -> str:
    graded = [(_graded_pass(e, hits), e, hits) for e, hits in zip(entries, results, strict=False)]
    scored = [g for g in graded if g[0] is not None]
    passed = sum(1 for g in scored if g[0])
    mode = results[0][0].search_mode if results and results[0] else "bm25"
    lines = [
        "# Golden query run",
        "",
        f"queries: **{len(entries)}**  ·  graded: **{len(scored)}**  ·  "
        f"PASS: **{passed}/{len(scored)}**  ·  mode: {mode}",
        "",
    ]
    for verdict, entry, hits in graded:
        tag = "—" if verdict is None else ("PASS" if verdict else "FAIL")
        lines += [
            f"## [{tag}] {entry.get('id', '?')}: `{entry.get('query', '')}`",
            f"expect_docs: {entry.get('expect_docs') or '(ungraded)'}",
            "",
        ]
        for i, hit in enumerate(hits[:10], 1):
            lines.append(f"{i}. [{hit.citation}] {hit.title} — {hit.path}")
        if entry.get("notes"):
            lines.append(f"_notes: {entry['notes']}_")
        lines.append("")
    return "\n".join(lines)


def _run_batch(catalog: Catalog, cfg: Config, args: argparse.Namespace) -> int:
    entries = yaml.safe_load(Path(args.batch_file).read_text(encoding="utf-8")) or []
    queries = [str(e.get("query", "")) for e in entries]
    top_k = args.top_k if args.top_k is not None else cfg.search.top_k_default
    # One process, one model load, all queries embedded together — this is the throughput
    # win over N separate `search` calls (each of which would reload the model).
    results = catalog.search(queries, top_k=top_k)
    if args.json:  # agents: structured results for every query in the batch
        print(
            json.dumps(
                [_json_result(q, hits) for q, hits in zip(queries, results, strict=False)],
                ensure_ascii=False,
            )
        )
        runlog.log("cli.search.batch", queries=len(entries), json=True)
        runlog.flush()
        return 0
    report = _render_golden(entries, results)
    out = (
        Path(args.out)
        if args.out
        else Path(cfg.paths.tmp_dir) / "reports" / f"golden-run-{runlog.RUN_ID}.md"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    graded = [_graded_pass(e, hits) for e, hits in zip(entries, results, strict=False)]
    passed = sum(1 for g in graded if g)
    scored = sum(1 for g in graded if g is not None)
    print(f"Graded {scored}/{len(entries)} golden queries: {passed} PASS -> {out}")
    runlog.log("cli.search.batch", queries=len(entries), passed=passed)
    runlog.flush()
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    cfg = config.load(Path(args.config))
    _configure_logging(cfg)
    with Store.open(cfg.paths.db_path) as store:
        doc = store.get_document(args.doc_id)
        if doc is None:
            print(f"No document with id {args.doc_id}")
            return 1
        print(f"# doc {doc['id']}: {doc['title']}")
        print(f"path: {doc['path']}")
        print(f"fmt: {doc['fmt']}  audience: {doc['audience']}  status: {doc['status']}")
        cap = int(getattr(args, "max_chars", 0) or 0)
        for chunk in store.chunks_for_document(args.doc_id):
            print(f"\n-- chunk {chunk['id']} [{chunk['kind']}] {chunk['locator']}")
            text = str(chunk["text"])
            # Full text by default (0 = no cap): `show` is for verbatim inspection, so it must
            # not silently truncate. `--max-chars N` opts into a display cap.
            print(text if cap <= 0 or len(text) <= cap else text[:cap] + " …")
    runlog.log("cli.show", doc_id=args.doc_id)
    runlog.flush()
    return 0


def _cmd_related(args: argparse.Namespace) -> int:
    """Documents cross-referenced from / to a doc over the relations graph (N-hop, §17)."""
    cfg = config.load(Path(args.config))
    _configure_logging(cfg)
    with Store.open(cfg.paths.db_path) as store:
        if store.get_document(args.doc_id) is None:
            print(f"No document with id {args.doc_id}")
            return 1
        rows = store.related_documents(args.doc_id, args.direction, depth=args.depth)
    if not rows:
        print(
            f"No related documents for doc {args.doc_id} "
            f"(direction={args.direction}, depth={args.depth})."
        )
        return 0
    for r in rows:
        lt = f" [{r['link_type']}]" if r["link_type"] else ""
        print(f"{r['hops']}·{str(r['direction']):<4} doc {r['doc_id']}: {r['title']}{lt}  ({r['path']})")
    runlog.log("cli.related", doc_id=args.doc_id, direction=args.direction, depth=args.depth)
    runlog.flush()
    return 0


def _cmd_discrepancies(args: argparse.Namespace) -> int:
    """Scan for duplicate active documents + high-similarity conflict candidates (§17)."""
    cfg = config.load(Path(args.config))
    _configure_logging(cfg)
    report = Catalog(cfg).check_discrepancies(persist=args.persist)
    dups, conflicts = report.duplicate_actives, report.conflict_candidates
    print(f"# Discrepancy scan\n\nDuplicate active documents: **{len(dups)}** group(s)")
    for g in dups:
        ids = ", ".join(f"{d} ({p})" for d, p in g.docs)
        print(f"  · {g.content_hash[:12]}…: docs {ids}")
    print(f"\nConflict candidates (near-duplicate across docs): **{len(conflicts)}**")
    for p in conflicts[: args.limit]:
        print(
            f"  · ~{p.similarity:.3f}  chunk {p.chunk_a} (doc {p.doc_a}) ↔ "
            f"chunk {p.chunk_b} (doc {p.doc_b})"
        )
    if args.persist:
        print("\nRecorded findings as `discrepancy` flags (filterable in the index).")
    if not report.conflict_candidates and cfg.embed.model == "none":
        print(
            "\nNote: this index is BM25-only — conflict detection needs embeddings "
            "(set `embed.model` and re-ingest).",
            file=sys.stderr,
        )
    runlog.log("cli.discrepancies", dups=len(dups), conflicts=len(conflicts), persist=args.persist)
    runlog.flush()
    return 0


def _self_heal_loop(cat: Catalog, minutes: int) -> None:  # pragma: no cover - lifetime loop
    """Periodically prune orphaned documents for the life of a long-running server."""
    while True:
        time.sleep(minutes * 60)
        try:
            pruned = cat.prune_missing(apply=True)
            if pruned:
                runlog.log("serve.selfheal.periodic", pruned=pruned)
        except Exception:  # noqa: BLE001 - a healer failure must never take the server down
            pass


def _start_self_heal(cat: Catalog, minutes: int) -> threading.Thread | None:
    """Start the periodic self-heal as a daemon thread; ``None`` if disabled (minutes<=0)."""
    if minutes <= 0:
        return None
    thread = threading.Thread(
        target=_self_heal_loop, args=(cat, minutes), name="docusearch-selfheal", daemon=True
    )
    thread.start()
    return thread


def _cmd_serve(args: argparse.Namespace) -> int:  # pragma: no cover - blocks on uvicorn
    cfg = config.load(Path(args.config))
    _configure_logging(cfg)
    # Self-healing (R-ING): documents are keyed by absolute path, so a moved/renamed source
    # folder leaves orphans. The server is long-running and rarely restarted, so prune on
    # startup AND periodically (serve.self_heal_minutes) — never serve dead docs / broken refs.
    cat = Catalog(cfg)
    healed = cat.prune_missing(apply=True)
    if healed:
        print(
            f"Self-heal: pruned {healed} orphaned documents (source files gone).", file=sys.stderr
        )
        runlog.log("serve.selfheal", pruned=healed)
    _start_self_heal(cat, cfg.serve.self_heal_minutes)
    from .server import serve

    serve(cfg, host=args.host, port=args.port)  # blocks until interrupted
    return 0


def _cmd_gate(args: argparse.Namespace) -> int:
    cfg = config.load(Path(args.config))
    _configure_logging(cfg)
    name = args.name or f"phase-{args.n}"
    gates_dir = Path(cfg.paths.tmp_dir) / "gates"
    gates_dir.mkdir(parents=True, exist_ok=True)
    path = gates_dir / f"GATE-{args.n}-{name}.md"
    path.write_text(_render_gate(args.n, name), encoding="utf-8")
    print(f"Wrote gate checklist to {path}")
    runlog.log("cli.gate", n=args.n, path=str(path))
    runlog.flush()
    return 0


def _render_gate(n: str, name: str) -> str:
    """A two-part sign-off checklist skeleton (§15.4, R-PROC-4).

    Part A: the operator audits the agent's public-corpus evidence (recomputing samples).
    Part B: the operator's independent investigation on his private dataset via the runbook.
    """
    return f"""# GATE {n} — {name}

> The signed copy of this file, committed to the repo, is the record that the gate
> passed. No gate, no progress (R-TEST-4).

## Part A — Audit the agent's evidence

- [ ] Self-verification results reviewed — `tmp/gates/evidence-{name}/`
- [ ] Audit counts vs the red team's independent recount agree
- [ ] Needle / obtuse suite tables reviewed (thresholds met)
- [ ] Performance table vs §14 budgets reviewed
- [ ] Red-team report reviewed — `redteam/REDTEAM-{name}.md`
- [ ] WORKLOG excerpt for this phase read
- [ ] Recompute spot-check: re-ran a sample and the numbers reproduce

## Part B — Independent investigation (private dataset)

- [ ] Ran the steps in `RUNBOOK-private-dataset.md` on my own data
- [ ] Spot-checked results against my expectations
- [ ] Any FAIL has a triage note (missing doc? chunking? threshold?)

## Verdict

- [ ] PASS
- [ ] FAIL

Signed: ________________________    Date: ____________

Notes:
"""


# ---- STDF tools over the LIVE MCP server (docusearch stdf ...) -----------------------------------
# These four commands are MCP clients: they drive the same `docusearch serve` an agent connects to,
# so every use is a wire-level CLIunicode-MCP parity check. Nothing here touches the store directly.


def _mcp_url(cfg: Config, override: str | None) -> str:
    """The MCP endpoint to call: an explicit --url, else derived from the config's serve section.
    A wildcard bind host (0.0.0.0/::) is dialled back to loopback for the client."""
    if override:
        return override
    host = cfg.serve.host
    if host in ("0.0.0.0", "::", ""):  # noqa: S104 - bind address, not a client dial address
        host = "127.0.0.1"
    path = cfg.serve.mcp_path if cfg.serve.mcp_path.startswith("/") else "/" + cfg.serve.mcp_path
    return f"http://{host}:{cfg.serve.port}{path}"


def _stdf_client(args: argparse.Namespace) -> tuple[Config, MCPClient]:
    cfg = config.load(Path(args.config))
    _configure_logging(cfg)
    return cfg, MCPClient(_mcp_url(cfg, args.url))


def _bundle_globs(patterns: list[str]) -> tuple[bytes, list[str]]:
    """Collect the local files matching ``patterns`` (recursive globs) and pack them into an
    in-memory ``.tar.gz`` with flat, de-duplicated basenames — the bytes ``stdf upload`` sends."""
    import glob as globmod
    import io
    import tarfile

    matched: list[Path] = []
    seen: set[str] = set()
    for pat in patterns:
        # user patterns can be absolute or carry a base dir (`ate/**/lot*.stdf`) — glob.glob, not
        # Path.glob which can't take a rooted pattern.
        for m in sorted(globmod.glob(pat, recursive=True)):  # noqa: PTH207
            p = Path(m)
            key = str(p.resolve())
            if p.is_file() and key not in seen:
                seen.add(key)
                matched.append(p)
    buf = io.BytesIO()
    used: set[str] = set()
    names: list[str] = []
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for p in matched:
            arc, i = p.name, 1
            while arc in used:  # two dirs, same basename → disambiguate so neither is lost
                arc = f"{p.stem}-{i}{p.suffix}"
                i += 1
            used.add(arc)
            names.append(arc)
            tar.add(str(p), arcname=arc)
    return buf.getvalue(), names


def _save_report(cfg: Config, out: str | None, title: str, html: str) -> Path:
    """Write a generated report under the config's tmp_dir (or an explicit --out), returning path."""
    path = Path(out) if out else Path(cfg.paths.tmp_dir) / "reports" / f"{title}.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


def _tool_error(res: Any) -> str | None:
    return str(res["message"]) if isinstance(res, dict) and res.get("error") else None


def _cmd_stdf_upload(args: argparse.Namespace) -> int:
    import base64

    cfg, client = _stdf_client(args)
    data, names = _bundle_globs(args.glob)
    if not names:
        print(f"error: no files matched {args.glob}", file=sys.stderr)
        return 2
    print(
        f"Uploading {len(names)} file(s) → SKU {args.sku!r} ({len(data) / 1e6:.1f} MB) via {client.url}…",
        file=sys.stderr,
    )
    res = client.call(
        "upload_archive", filename="upload.tar.gz",
        data_b64=base64.b64encode(data).decode("ascii"),
        sku=args.sku, insertion=args.insertion, user=args.user or "", store=args.store,
    )
    err = _tool_error(res)
    if err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    print(
        f"Uploaded to SKU {args.sku!r}: {res.get('documents')} document(s), "
        f"{res.get('chunks')} chunk(s)."
    )
    runlog.log("cli.stdf_upload", sku=args.sku, files=len(names), docs=res.get("documents"))
    runlog.flush()
    return 0


def _stdf_list(client: MCPClient, args: argparse.Namespace, glob: str) -> Any:
    return client.call(
        "list_stdf", glob=glob, sku=getattr(args, "sku", "") or "",
        store=args.store, user=args.user or None,
    )


def _cmd_stdf_ls(args: argparse.Namespace) -> int:
    _cfg, client = _stdf_client(args)
    res = _stdf_list(client, args, args.glob or "")
    err = _tool_error(res)
    if err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    docs = res.get("documents", [])
    skus = res.get("skus", [])
    if not docs:
        hint = f"  Known SKUs: {', '.join(skus)}" if skus else "  (no STDF files uploaded yet)"
        print("No STDF files matched.\n" + hint)
        return 0
    print(f"{'id':>5}  {'SKU':<16} {'lot':<10} {'insertions':<14} {'tests':>6} {'parts':>6}  file")
    for d in docs:
        print(
            f"{d['doc_id']:>5}  {d['sku'][:16]:<16} {d['lot'][:10]:<10} "
            f"{d['insertions'][:14]:<14} {d['tests']:>6} {d['parts']:>6}  {Path(d['path']).name}"
        )
    print(f"\n{len(docs)} file(s) · SKUs present: {', '.join(skus) or '—'}")
    return 0


def _cmd_stdf_report(args: argparse.Namespace) -> int:
    cfg, client = _stdf_client(args)
    res = _stdf_list(client, args, args.glob)
    err = _tool_error(res)
    if err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    docs = res.get("documents", [])
    if not docs:
        print(f"error: no STDF files matched {args.glob!r}", file=sys.stderr)
        return 2
    ids = [d["doc_id"] for d in docs]
    if args.test is None:
        print(f"{len(docs)} file(s) matched {args.glob!r}:")
        for d in docs:
            print(f"  doc {d['doc_id']}  SKU {d['sku']}  lot {d['lot']}  "
                  f"{d['tests']} tests  {d['parts']} parts  {Path(d['path']).name}")
        print("\nPass --test N to render a plot (1 file → distribution, many → trend across runs).")
        return 0
    if len(ids) == 1:
        res = client.call("stdf_plot", doc_id=ids[0], test_num=args.test, kind=args.kind,
                          backend=args.backend or "", store=args.store)
        title = f"stdf_plot_doc{ids[0]}_test{args.test}_{args.kind}"
    else:
        res = client.call("stdf_trend", doc_ids=ids, test_num=args.test, stat=args.stat,
                          backend=args.backend or "", store=args.store)
        title = f"stdf_trend_test{args.test}_{args.stat}"
    err = _tool_error(res)
    if err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    out = _save_report(cfg, args.out, title, res["html"])
    print(f"Wrote {out}  (test {args.test} across {len(ids)} file(s))")
    runlog.log("cli.stdf_report", test=args.test, files=len(ids), out=str(out))
    runlog.flush()
    return 0


def _cmd_stdf_audit(args: argparse.Namespace) -> int:
    cfg, client = _stdf_client(args)
    if args.glob_b:
        a = _stdf_list(client, args, args.glob_a)
        b = _stdf_list(client, args, args.glob_b)
        for res in (a, b):
            err = _tool_error(res)
            if err:
                print(f"error: {err}", file=sys.stderr)
                return 1
        da, db = a.get("documents", []), b.get("documents", [])
        if len(da) != 1 or len(db) != 1:
            print(f"error: each glob must match exactly one STDF file "
                  f"({args.glob_a!r}→{len(da)}, {args.glob_b!r}→{len(db)})", file=sys.stderr)
            return 2
        ids = [da[0]["doc_id"], db[0]["doc_id"]]
    else:
        res = _stdf_list(client, args, args.glob_a)
        err = _tool_error(res)
        if err:
            print(f"error: {err}", file=sys.stderr)
            return 1
        docs = res.get("documents", [])
        if len(docs) != 2:
            print(f"error: audit needs exactly two STDF files; {args.glob_a!r} matched {len(docs)}. "
                  "Give two globs, or tighten the pattern.", file=sys.stderr)
            return 2
        ids = sorted(d["doc_id"] for d in docs)
    res = client.call("stdf_audit", doc_a=ids[0], doc_b=ids[1],
                      backend=args.backend or "", store=args.store)
    err = _tool_error(res)
    if err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    out = _save_report(cfg, args.out, f"stdf_audit_doc{ids[0]}_vs_doc{ids[1]}", res["html"])
    print(f"Wrote audit dashboard {out}  (doc {ids[0]} vs doc {ids[1]})")
    runlog.log("cli.stdf_audit", doc_a=ids[0], doc_b=ids[1], out=str(out))
    runlog.flush()
    return 0


def _resolve_one(client: MCPClient, args: argparse.Namespace) -> dict[str, Any] | None:
    """Resolve an STDF glob to exactly one document (for the single-file wafer commands)."""
    res = _stdf_list(client, args, args.glob)
    err = _tool_error(res)
    if err:
        print(f"error: {err}", file=sys.stderr)
        return None
    docs = res.get("documents", [])
    if len(docs) != 1:
        print(f"error: {args.glob!r} matched {len(docs)} STDF files; need exactly one.",
              file=sys.stderr)
        return None
    return docs[0]  # type: ignore[no-any-return]


def _cmd_stdf_wafermap(args: argparse.Namespace) -> int:
    cfg, client = _stdf_client(args)
    doc = _resolve_one(client, args)
    if doc is None:
        return 2
    res = client.call("wafer_map", doc_id=doc["doc_id"], wafer_id=args.wafer,
                      color_by=args.color_by, store=args.store, user=args.user or None)
    err = _tool_error(res)
    if err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    out = _save_report(cfg, args.out, f"wafermap_doc{doc['doc_id']}_{args.wafer or 'first'}", res["html"])
    print(f"Wrote wafer map {out}")
    return 0


def _cmd_stdf_motherlot(args: argparse.Namespace) -> int:
    cfg, client = _stdf_client(args)
    doc = _resolve_one(client, args)
    if doc is None:
        return 2
    res = client.call("mother_lot", doc_id=doc["doc_id"], backend=args.backend or "",
                      store=args.store, user=args.user or None)
    err = _tool_error(res)
    if err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    out = _save_report(cfg, args.out, f"motherlot_doc{doc['doc_id']}", res["html"])
    print(f"Wrote mother-lot view {out}")
    return 0


def _cmd_stdf_yieldtrend(args: argparse.Namespace) -> int:
    cfg, client = _stdf_client(args)
    res = _stdf_list(client, args, args.glob)
    err = _tool_error(res)
    if err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    ids = sorted(d["doc_id"] for d in res.get("documents", []))
    if not ids:
        print(f"error: no STDF files matched {args.glob!r}", file=sys.stderr)
        return 2
    res = client.call("production_trend", doc_ids=ids, backend=args.backend or "",
                      store=args.store, user=args.user or None)
    err = _tool_error(res)
    if err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    out = _save_report(cfg, args.out, f"yieldtrend_{len(ids)}lots", res["html"])
    print(f"Wrote yield trend {out}  ({len(ids)} lots)")
    return 0


def _add_stdf_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default="docusearch.yaml", help="config path")
    p.add_argument("--url", default=None, help="MCP endpoint (default: from config's serve section)")
    p.add_argument("--store", default=None, help="federation member to target (default: single store)")
    p.add_argument("--user", default=None, help="requester (for a private store / attribution)")


# ---- Generic data-store tools over the live MCP server (docusearch data ...) ---------------------
# Any CSV/table's columns, not just STDF — mirrors `docusearch stdf`, same live-MCP-client transport.


def _match_col(col: dict[str, Any], pattern: str) -> bool:
    """Case-insensitive glob against a column's ``dataset.name`` or bare ``name``."""
    import fnmatch
    pat = pattern.lower()
    name, dataset = str(col.get("name", "")).lower(), str(col.get("dataset", "")).lower()
    return fnmatch.fnmatch(name, pat) or fnmatch.fnmatch(f"{dataset}.{name}", pat)


def _cmd_data_ls(args: argparse.Namespace) -> int:
    _cfg, client = _stdf_client(args)
    res = client.call("list_data", store=args.store, user=args.user or None)
    err = _tool_error(res)
    if err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    cols = [c for c in res.get("columns", []) if not args.glob or _match_col(c, args.glob)]
    if not cols:
        print("No data columns matched.")
        return 0
    print(f"{'id':>5}  {'dataset':<16} {'column':<20} {'kind':<11} {'n':>6}  limits")
    for c in cols:
        lim = "" if c["lo"] is None and c["hi"] is None else f"{c['lo']}..{c['hi']} {c['units']}"
        print(f"{c['id']:>5}  {str(c['dataset'])[:16]:<16} {str(c['name'])[:20]:<20} "
              f"{str(c['kind'])[:11]:<11} {c['n']:>6}  {lim}")
    print(f"\n{len(cols)} column(s)")
    return 0


def _cmd_data_plot(args: argparse.Namespace) -> int:
    cfg, client = _stdf_client(args)
    res = client.call("list_data", store=args.store, user=args.user or None)
    err = _tool_error(res)
    if err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    cols = [c for c in res.get("columns", []) if _match_col(c, args.glob)]
    if len(cols) != 1:
        print(f"error: {args.glob!r} matched {len(cols)} columns; need exactly one (see `data ls`).",
              file=sys.stderr)
        return 2
    col = cols[0]
    res = client.call("data_plot", column_id=col["id"], kind=args.kind, backend=args.backend or "",
                      by_group=args.by_group, store=args.store, user=args.user or None)
    err = _tool_error(res)
    if err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    out = _save_report(cfg, args.out, f"data_{col['dataset']}_{col['name']}_{args.kind}", res["html"])
    s = res.get("stats", {})
    print(f"Wrote {out}  ({col['dataset']}.{col['name']} · n={res.get('n')}"
          + (f" · mean={s['mean']:.4g}" if s.get("n") else "") + ")")
    runlog.log("cli.data_plot", column=col["name"], out=str(out))
    runlog.flush()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="docusearch",
        description="Enterprise documentation catalog — local search with citations.",
    )
    parser.add_argument("--version", action="version", version=f"docusearch {__version__}")
    parser.set_defaults(func=None)
    sub = parser.add_subparsers(dest="command")

    p_init = sub.add_parser("init", help="write a fully-commented docusearch.yaml")
    p_init.add_argument("--config", default="docusearch.yaml", help="config path to write")
    p_init.add_argument("--force", action="store_true", help="overwrite an existing config")
    p_init.set_defaults(func=_cmd_init)

    p_ingest = sub.add_parser("ingest", help="ingest sources into the index")
    p_ingest.add_argument("--config", default="docusearch.yaml", help="config path")
    p_ingest.add_argument("--dry-run", action="store_true", help="preview the plan, touch nothing")
    p_ingest.add_argument(
        "--force",
        action="store_true",
        help="full rebuild: re-parse every file (ignore hash cache) AND re-embed all vectors",
    )
    p_ingest.add_argument(
        "--reembed",
        action="store_true",
        help="drop existing vectors first, then re-embed (switch models / heal a mixed index)",
    )
    p_ingest.set_defaults(func=_cmd_ingest)

    p_audit = sub.add_parser("audit", help="print the current index audit (counts + anomalies)")
    p_audit.add_argument("--config", default="docusearch.yaml", help="config path")
    p_audit.set_defaults(func=_cmd_audit)

    p_remove = sub.add_parser("remove", help="purge everything ingested under a source label")
    p_remove.add_argument("source", help="the source name to purge (e.g. delete_me_next)")
    p_remove.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    p_remove.add_argument("--config", default="docusearch.yaml", help="config path")
    p_remove.set_defaults(func=_cmd_remove)

    p_models = sub.add_parser("models", help="list downloaded embedding models + how to delete")
    p_models.set_defaults(func=_cmd_models)

    p_prune = sub.add_parser("prune", help="remove documents whose source file no longer exists")
    p_prune.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    p_prune.add_argument("--config", default="docusearch.yaml", help="config path")
    p_prune.set_defaults(func=_cmd_prune)

    p_vision = sub.add_parser(
        "vision", help="enrich retained images with cloud OCR + description (enrich.vision_images)"
    )
    p_vision.add_argument(
        "--limit", type=int, default=None, help="only enrich the first N pending images (cost cap)"
    )
    p_vision.add_argument(
        "--largest",
        action="store_true",
        help="enrich the largest images first (real diagrams before tiny icons)",
    )
    p_vision.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    p_vision.add_argument("--config", default="docusearch.yaml", help="config path")
    p_vision.set_defaults(func=_cmd_vision)

    p_summarize = sub.add_parser(
        "summarize",
        help="generate a searchable AI summary per document (enrich.ai_summaries; off by default)",
    )
    p_summarize.add_argument(
        "--model", default="claude-opus-4-8", help="Claude model for summaries"
    )
    p_summarize.add_argument(
        "--limit", type=int, default=None, help="only summarize the first N pending docs"
    )
    p_summarize.add_argument("--config", default="docusearch.yaml", help="config path")
    p_summarize.set_defaults(func=_cmd_summarize)

    p_preflight = sub.add_parser(
        "preflight",
        help="sample the corpus → Claude proposes gotcha rules → preflight_rules.yaml (you approve)",
    )
    p_preflight.add_argument(
        "--model", default="claude-opus-4-8", help="Claude model for the proposal (temp 0)"
    )
    p_preflight.add_argument(
        "--out", default=None, help="where to write the rules (default: enrich.preflight_rules)"
    )
    p_preflight.add_argument("--seed", type=int, default=7, help="sampling seed (deterministic)")
    p_preflight.add_argument(
        "--yes", action="store_true", help="overwrite an already-approved rules file"
    )
    p_preflight.add_argument("--config", default="docusearch.yaml", help="config path")
    p_preflight.set_defaults(func=_cmd_preflight)

    p_inspect = sub.add_parser(
        "inspect", help="sample a source and propose content_selector / strip_selectors"
    )
    p_inspect.add_argument("source", nargs="?", default=None, help="source name (default: first)")
    p_inspect.add_argument("--sample", type=int, default=None, help="how many files to sample")
    p_inspect.add_argument("--config", default="docusearch.yaml", help="config path")
    p_inspect.set_defaults(func=_cmd_inspect)

    p_search = sub.add_parser("search", help="search the index (hybrid if embeddings exist)")
    p_search.add_argument("query", nargs="?", help="search text (omit when using --batch-file)")
    p_search.add_argument("--top-k", type=int, default=None, help="number of results")
    p_search.add_argument("--prefix", action="store_true", help="prefix matching (partial terms)")
    p_search.add_argument("--batch-file", help="YAML goldens (id, query, expect_docs) to grade")
    p_search.add_argument("--out", help="write the graded golden report here")
    p_search.add_argument("--json", action="store_true", help="emit hits as JSON (for agents)")
    p_search.add_argument(
        "--stores",
        default="",
        help="federation only: comma-separated member names to search (e.g. internal). "
        "Omit to search all members.",
    )
    p_search.add_argument("--config", default="docusearch.yaml", help="config path")
    p_search.set_defaults(func=_cmd_search)

    p_report = sub.add_parser("report", help="render a cited answer spec (YAML) to md/html")
    p_report.add_argument(
        "--spec", required=True, help="YAML: title, body (with [D:] cites), evidence"
    )
    p_report.add_argument(
        "--format", choices=("md", "html", "pdf", "docx", "pptx", "xlsx"), default=None,
        help="output format (pdf/docx/pptx/xlsx need --out)",
    )
    p_report.add_argument("--out", help="write here (default: stdout); format inferred from .html")
    p_report.add_argument("--request", default="", help="the exact request this report answers")
    p_report.add_argument("--requested-by", default="", help="user the report is for")
    p_report.add_argument("--model", default="", help="model that generated the report")
    p_report.add_argument(
        "--classification",
        default=None,
        help="confidentiality banner (default: Confidential)",
    )
    p_report.add_argument("--config", default="docusearch.yaml", help="config path")
    p_report.set_defaults(func=_cmd_report)

    p_show = sub.add_parser("show", help="print a document's chunks by id (full text)")
    p_show.add_argument("doc_id", type=int, help="document id")
    p_show.add_argument(
        "--max-chars", type=int, default=0, help="cap each chunk's printed text (0 = full text)"
    )
    p_show.add_argument("--config", default="docusearch.yaml", help="config path")
    p_show.set_defaults(func=_cmd_show)

    p_related = sub.add_parser(
        "related", help="documents cross-referenced from/to a doc over the relations graph (N-hop)"
    )
    p_related.add_argument("doc_id", type=int, help="document id")
    p_related.add_argument(
        "--direction", choices=("out", "in", "both"), default="both",
        help="out=this doc links to · in=links to this doc · both",
    )
    p_related.add_argument("--depth", type=int, default=1, help="walk N hops (default 1)")
    p_related.add_argument("--config", default="docusearch.yaml", help="config path")
    p_related.set_defaults(func=_cmd_related)

    p_disc = sub.add_parser(
        "discrepancies",
        help="scan for duplicate active docs + high-similarity conflict candidates (§17)",
    )
    p_disc.add_argument(
        "--persist", action="store_true", help="record findings as filterable `discrepancy` flags"
    )
    p_disc.add_argument("--limit", type=int, default=50, help="max conflict pairs to print")
    p_disc.add_argument("--config", default="docusearch.yaml", help="config path")
    p_disc.set_defaults(func=_cmd_discrepancies)

    p_serve = sub.add_parser("serve", help="run the REST + MCP server (Phase 3)")
    p_serve.add_argument("--config", default="docusearch.yaml", help="config path")
    p_serve.add_argument("--host", default=None, help="bind host (default from config)")
    p_serve.add_argument("--port", type=int, default=None, help="port (default from config)")
    p_serve.set_defaults(func=_cmd_serve)

    p_gate = sub.add_parser("gate", help="write a Part A/B sign-off checklist")
    p_gate.add_argument("n", help="gate id, e.g. 1 or 4a")
    p_gate.add_argument("--name", default="", help="gate name slug (default phase-<n>)")
    p_gate.add_argument("--config", default="docusearch.yaml", help="config path")
    p_gate.set_defaults(func=_cmd_gate)

    p_stdf = sub.add_parser("stdf", help="STDF data tools over the running MCP server (serve)")
    p_stdf.set_defaults(func=None)
    stdf_sub = p_stdf.add_subparsers(dest="stdf_command")

    p_up = stdf_sub.add_parser("upload", help="bundle local STDF files and upload them under a SKU")
    p_up.add_argument("sku", help="part SKU/name to file these under (the upload bucket — required)")
    p_up.add_argument("glob", nargs="+", help="local file glob(s), e.g. 'ate/*.stdf' '**/lot*.stdf'")
    p_up.add_argument("--insertion", required=True,
                      help="operator insertion label for these files (WS1 | WS1-RT | FT | …) — "
                           "required so yield separates first-pass from retest correctly")
    _add_stdf_common(p_up)
    p_up.set_defaults(func=_cmd_stdf_upload)

    p_ls = stdf_sub.add_parser("ls", help="list uploaded STDF files (SKU + glob filtering)")
    p_ls.add_argument("glob", nargs="?", default="", help="filter by file name/path glob")
    p_ls.add_argument("--sku", default="", help="filter to one part SKU/name")
    _add_stdf_common(p_ls)
    p_ls.set_defaults(func=_cmd_stdf_ls)

    p_rep = stdf_sub.add_parser("report", help="analyze/plot uploaded STDF(s) matched by glob")
    p_rep.add_argument("glob", help="file name/path glob selecting the STDF(s) to analyze")
    p_rep.add_argument("--sku", default="", help="also constrain to one part SKU/name")
    p_rep.add_argument("--test", type=int, default=None, help="test number to plot (else list matches)")
    p_rep.add_argument("--kind", default="histogram",
                       help="plot kind for a single file: histogram|whisker|quantile|qq|xy|linear")
    p_rep.add_argument("--stat", default="mean", help="trend stat across many files: mean|median|std|min|max")
    p_rep.add_argument("--backend", default="", help="matplotlib|plotly (default from config)")
    p_rep.add_argument("--out", default=None, help="output HTML path (default under tmp_dir/reports)")
    _add_stdf_common(p_rep)
    p_rep.set_defaults(func=_cmd_stdf_report)

    p_aud = stdf_sub.add_parser("audit", help="Beyond-Compare audit of two uploaded STDFs")
    p_aud.add_argument("glob_a", help="glob for the first file (or a glob matching exactly two)")
    p_aud.add_argument("glob_b", nargs="?", default="", help="glob for the second file (optional)")
    p_aud.add_argument("--sku", default="", help="constrain matches to one part SKU/name")
    p_aud.add_argument("--backend", default="", help="matplotlib|plotly (default from config)")
    p_aud.add_argument("--out", default=None, help="output HTML path (default under tmp_dir/reports)")
    _add_stdf_common(p_aud)
    p_aud.set_defaults(func=_cmd_stdf_audit)

    p_wm = stdf_sub.add_parser("wafermap", help="wafer map (die grid) for one STDF file")
    p_wm.add_argument("glob", help="glob selecting exactly one STDF file")
    p_wm.add_argument("--wafer", default="", help="wafer id (default: the first wafer present)")
    p_wm.add_argument("--color-by", default="pass", choices=("pass", "softbin"), help="die colouring")
    p_wm.add_argument("--out", default=None, help="output HTML path (default under tmp_dir/reports)")
    _add_stdf_common(p_wm)
    p_wm.set_defaults(func=_cmd_stdf_wafermap)

    p_ml = stdf_sub.add_parser("motherlot", help="per-wafer yield across a lot (one STDF file)")
    p_ml.add_argument("glob", help="glob selecting exactly one STDF file")
    p_ml.add_argument("--backend", default="", help="matplotlib|plotly (default from config)")
    p_ml.add_argument("--out", default=None, help="output HTML path (default under tmp_dir/reports)")
    _add_stdf_common(p_ml)
    p_ml.set_defaults(func=_cmd_stdf_motherlot)

    p_yt = stdf_sub.add_parser("yieldtrend", help="long-term yield trend across matched STDF files")
    p_yt.add_argument("glob", help="glob selecting the STDF files (each a lot/date point)")
    p_yt.add_argument("--backend", default="", help="matplotlib|plotly (default from config)")
    p_yt.add_argument("--out", default=None, help="output HTML path (default under tmp_dir/reports)")
    _add_stdf_common(p_yt)
    p_yt.set_defaults(func=_cmd_stdf_yieldtrend)

    p_data = sub.add_parser("data", help="generic data-store tools (any CSV/table) over the MCP server")
    p_data.set_defaults(func=None)
    data_sub = p_data.add_subparsers(dest="data_command")

    p_dls = data_sub.add_parser("ls", help="list numeric columns in the data store")
    p_dls.add_argument("glob", nargs="?", default="", help="filter columns by name / dataset.name glob")
    _add_stdf_common(p_dls)
    p_dls.set_defaults(func=_cmd_data_ls)

    p_dpl = data_sub.add_parser("plot", help="plot one data column matched by name")
    p_dpl.add_argument("glob", help="column selector: name or dataset.name glob (must match one)")
    p_dpl.add_argument("--kind", default="histogram",
                       help="histogram|whisker|quantile|qq|xy|linear")
    p_dpl.add_argument("--by-group", action="store_true", help="one series per group (e.g. site)")
    p_dpl.add_argument("--backend", default="", help="matplotlib|plotly (default from config)")
    p_dpl.add_argument("--out", default=None, help="output HTML path (default under tmp_dir/reports)")
    _add_stdf_common(p_dpl)
    p_dpl.set_defaults(func=_cmd_data_plot)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.func is None:
        parser.print_help(sys.stderr)
        return 2
    try:
        result: int = args.func(args)
    except (embed.EmbedError, config.ConfigError, StoreError, VisionError, enrich.EnrichError,
            MCPError) as err:
        # Known, user-actionable failures (model mismatch, bad config, unusable DB, malformed
        # preflight rules, an unreachable/erroring MCP server): print just the guidance, not a
        # Python traceback the user can't act on.
        print(f"error: {err}", file=sys.stderr)
        return 1
    return result


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
