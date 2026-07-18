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

from . import config, embed, ingest, runlog
from ._version import __version__
from .catalog import Catalog
from .config import Config
from .search import SearchHit
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
    print(
        f"Ingested {result.documents} docs, {result.chunks} chunks, {result.images} images "
        f"({result.skipped_unchanged} unchanged, {result.stripped_empty} too short, "
        f"{result.excluded_glob} excluded); embedded {result.embedded} chunks."
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
    result = Catalog(cfg).enrich_vision(limit=args.limit, progress=_ProgressBar())
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
    catalog = Catalog(cfg)
    if args.batch_file:
        return _run_batch(catalog, cfg, args)
    if not args.query:
        print("Provide a query, or --batch-file <goldens.yaml>.")
        return 2
    hits = catalog.search(args.query, top_k=args.top_k, prefix=args.prefix)
    if args.json:  # machine-readable for agents: everything needed to cite a hit
        print(json.dumps(_json_result(args.query, hits), ensure_ascii=False))
        runlog.log("cli.search", query=args.query, results=len(hits))
        runlog.flush()
        return 0
    if not hits:
        print("No results.")
    else:
        print(f"({hits[0].search_mode} search; embed_model={hits[0].embed_model_used})")
    for i, hit in enumerate(hits, 1):
        print(f"{i}. [{hit.citation}] {hit.title}  ({hit.locator})")
        print(f"   {hit.snippet}")
        print(f"   score={hit.score}  kind={hit.kind}  path={hit.path}")
    runlog.log("cli.search", query=args.query, results=len(hits))
    runlog.flush()
    return 0


def _reference_targets(
    db_path: str, evidence: set[tuple[int, int]]
) -> dict[tuple[int, int], tuple[str, str]]:
    """Map each cited ``(doc_id, chunk_id)`` to ``(file:// href, "store — title — heading")``
    so report references open the original vendor document."""
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
            try:
                href = Path(path).as_uri() if path else ""
            except ValueError:  # non-absolute path -> leave as-is
                href = path
            targets[(doc_id, chunk_id)] = (href, label)
    return targets


def _cmd_report(args: argparse.Namespace) -> int:
    """Render a cited answer spec (YAML) to an md/html report, verifying every citation
    against the evidence the agent actually retrieved (refuses hallucinated references)."""
    from . import citations, report

    cfg = config.load(Path(args.config))
    _configure_logging(cfg)
    spec = yaml.safe_load(Path(args.spec).read_text(encoding="utf-8")) or {}
    evidence = {(int(d), int(c)) for d, c in spec.get("evidence", [])}
    fmt = args.format or ("html" if str(args.out or "").endswith(".html") else "md")
    base_url = f"http://localhost:{cfg.serve.port}"
    # header provenance: CLI flags win over spec fields (agents can pass either)
    sources = list(spec.get("sources", [])) or [s.name for s in cfg.sources]
    # References link to the ORIGINAL vendor document (file://), labelled "store — title —
    # heading", so the reader can open the parsed source, not an opaque chunk URL.
    ref_targets = _reference_targets(cfg.paths.db_path, evidence)
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
            request=args.request or str(spec.get("request", "")),
            requested_by=args.requested_by or str(spec.get("requested_by", "")),
            model=args.model or str(spec.get("model", "")),
            classification=(
                args.classification
                if args.classification is not None
                else str(spec.get("classification", "Confidential — Acme"))
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
        for chunk in store.chunks_for_document(args.doc_id):
            print(f"\n-- chunk {chunk['id']} [{chunk['kind']}] {chunk['locator']}")
            text = str(chunk["text"])
            print(text if len(text) <= 800 else text[:800] + " …")
    runlog.log("cli.show", doc_id=args.doc_id)
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

    Part A: Stephen audits the agent's public-corpus evidence (recomputing samples).
    Part B: Stephen's independent investigation on his private dataset via the runbook.
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
    p_vision.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    p_vision.add_argument("--config", default="docusearch.yaml", help="config path")
    p_vision.set_defaults(func=_cmd_vision)

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
    p_search.add_argument("--config", default="docusearch.yaml", help="config path")
    p_search.set_defaults(func=_cmd_search)

    p_report = sub.add_parser("report", help="render a cited answer spec (YAML) to md/html")
    p_report.add_argument(
        "--spec", required=True, help="YAML: title, body (with [D:] cites), evidence"
    )
    p_report.add_argument("--format", choices=("md", "html"), default=None, help="output format")
    p_report.add_argument("--out", help="write here (default: stdout); format inferred from .html")
    p_report.add_argument("--request", default="", help="the exact request this report answers")
    p_report.add_argument("--requested-by", default="", help="user the report is for")
    p_report.add_argument("--model", default="", help="model that generated the report")
    p_report.add_argument(
        "--classification",
        default=None,
        help="confidentiality banner (default: Confidential — Acme)",
    )
    p_report.add_argument("--config", default="docusearch.yaml", help="config path")
    p_report.set_defaults(func=_cmd_report)

    p_show = sub.add_parser("show", help="print a document's chunks by id")
    p_show.add_argument("doc_id", type=int, help="document id")
    p_show.add_argument("--config", default="docusearch.yaml", help="config path")
    p_show.set_defaults(func=_cmd_show)

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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.func is None:
        parser.print_help(sys.stderr)
        return 2
    try:
        result: int = args.func(args)
    except (embed.EmbedError, config.ConfigError, StoreError, VisionError) as err:
        # Known, user-actionable failures (model mismatch, bad config, unusable DB):
        # print just the guidance, not a Python traceback the user can't act on.
        print(f"error: {err}", file=sys.stderr)
        return 1
    return result


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
