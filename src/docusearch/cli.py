"""Command-line interface — a thin front-end over the library (§4).

Phase 0 ships three commands; the rest (``audit``, ``search``, ``serve``) arrive with
their phases. Every command resolves the config, does one job, and logs one event.

    docusearch init [--config PATH] [--force]
    docusearch ingest --dry-run [--config PATH]
    docusearch gate <n> [--name NAME] [--config PATH]

The console entry point is ``main`` (see pyproject ``[project.scripts]``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import config, runlog
from ._version import __version__
from .config import Config


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
    if not args.dry_run:
        print(
            "Ingestion runs in Phase 1. Re-run with --dry-run to preview the plan.",
        )
        return 2
    cfg = config.load(Path(args.config))
    _configure_logging(cfg)
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

    p_ingest = sub.add_parser("ingest", help="ingest sources (Phase 0: --dry-run preview only)")
    p_ingest.add_argument("--config", default="docusearch.yaml", help="config path")
    p_ingest.add_argument("--dry-run", action="store_true", help="preview the plan, touch nothing")
    p_ingest.set_defaults(func=_cmd_ingest)

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
