"""Command-line interface — a thin front-end over the library (§4).

``serve`` arrives in Phase 3. Every command resolves the config, does one job, and
logs one event.

    docusearch init [--config PATH] [--force]
    docusearch ingest [--dry-run] [--force] [--config PATH]
    docusearch audit [--config PATH]
    docusearch search <query> [--top-k N] [--prefix] [--config PATH]
    docusearch show <doc_id> [--config PATH]
    docusearch gate <n> [--name NAME] [--config PATH]

The console entry point is ``main`` (see pyproject ``[project.scripts]``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import config, ingest, runlog
from ._version import __version__
from .catalog import Catalog
from .config import Config
from .store import Store


def _configure_logging(cfg: Config) -> None:
    runlog.configure(
        Path(cfg.paths.tmp_dir) / "logs",
        level=cfg.logging.level,
        enabled=cfg.logging.jsonl,
    )


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

    result = Catalog(cfg).ingest(force=args.force)
    reports = Path(cfg.paths.tmp_dir) / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    report_path = reports / f"ingest-audit-{runlog.RUN_ID}.md"
    report_path.write_text(
        ingest.render_ingest_audit(result, run_id=runlog.RUN_ID), encoding="utf-8"
    )
    print(
        f"Ingested {result.documents} docs, {result.chunks} chunks, {result.images} images "
        f"({result.skipped_unchanged} unchanged, {result.stripped_empty} too short, "
        f"{result.excluded_glob} excluded)."
    )
    print(f"Audit report: {report_path}")
    runlog.log("cli.ingest", documents=result.documents, chunks=result.chunks)
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


def _cmd_search(args: argparse.Namespace) -> int:
    cfg = config.load(Path(args.config))
    _configure_logging(cfg)
    hits = Catalog(cfg).search(args.query, top_k=args.top_k, prefix=args.prefix)
    if not hits:
        print("No results.")
    for i, hit in enumerate(hits, 1):
        print(f"{i}. [{hit.citation}] {hit.title}  ({hit.locator})")
        print(f"   {hit.snippet}")
        print(f"   score={hit.score}  kind={hit.kind}  path={hit.path}")
    runlog.log("cli.search", query=args.query, results=len(hits))
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
    p_ingest.add_argument("--force", action="store_true", help="re-ingest all (ignore hash cache)")
    p_ingest.set_defaults(func=_cmd_ingest)

    p_audit = sub.add_parser("audit", help="print the current index audit (counts + anomalies)")
    p_audit.add_argument("--config", default="docusearch.yaml", help="config path")
    p_audit.set_defaults(func=_cmd_audit)

    p_search = sub.add_parser("search", help="BM25 search the index")
    p_search.add_argument("query", help="search text")
    p_search.add_argument("--top-k", type=int, default=None, help="number of results")
    p_search.add_argument("--prefix", action="store_true", help="prefix matching (partial terms)")
    p_search.add_argument("--config", default="docusearch.yaml", help="config path")
    p_search.set_defaults(func=_cmd_search)

    p_show = sub.add_parser("show", help="print a document's chunks by id")
    p_show.add_argument("doc_id", type=int, help="document id")
    p_show.add_argument("--config", default="docusearch.yaml", help="config path")
    p_show.set_defaults(func=_cmd_show)

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
    result: int = args.func(args)
    return result


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
