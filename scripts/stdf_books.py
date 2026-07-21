#!/usr/bin/env python3
"""Ingest a pair of STDF logs and produce every audit book from them — no model involved.

    python scripts/stdf_books.py --config stdf.yaml --a WS1.stdf --b WS2.stdf --out ./books

Everything here is computed: the findings, their wording, the plot chosen for each, the
statistics table and the ordering all come from the analytics layer, so the same two logs always
produce the same books. Six are written by default — pptx, pdf and html, each in both orderings:

    severity : worst capability first, the order a reviewer wants
    stdf     : the order the log runs the tests, for reading beside the program

Add --problems for the short pre-production check instead of the full book, and --dashboard for
the interactive six-tab HTML.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

from docusearch import config as cfg
from docusearch import ingest
from docusearch.server import Service
from docusearch.store import Store

FORMATS = ("pptx", "pdf", "html")
ORDERS = ("severity", "stdf")


def _ingest(config_path: Path, sources: list[Path]) -> tuple[Service, list[int]]:
    """Copy the logs into the configured source folder and ingest them."""
    conf = cfg.load(config_path)
    target = Path(conf.sources[0].location)
    target.mkdir(parents=True, exist_ok=True)
    for src in sources:
        if src.resolve().parent != target.resolve():
            shutil.copy2(src, target / src.name)
    with Store.open(conf.paths.db_path) as store:
        result = ingest.run_ingest(conf, store)
        docs = [
            int(r[0]) for r in store._conn.execute(
                "SELECT id FROM documents WHERE fmt='stdf' ORDER BY id"
            )
        ]
    print(f"ingested {result.documents} document(s), {result.stdf_tests:,} test records")
    return Service(conf), docs


def _docs_in(conf: object) -> list[int]:
    with Store.open(conf.paths.db_path) as store:  # type: ignore[attr-defined]
        return [
            int(r[0]) for r in store._conn.execute(
                "SELECT id FROM documents WHERE fmt='stdf' ORDER BY id"
            )
        ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="a data-store docusearch.yaml")
    parser.add_argument("--a", type=Path, help="the baseline STDF log")
    parser.add_argument("--b", type=Path, help="the log to audit against it")
    parser.add_argument("--out", type=Path, default=Path("./books"), help="where to write them")
    parser.add_argument("--label-a", default="", help="how to name the baseline")
    parser.add_argument("--label-b", default="", help="how to name the new run")
    parser.add_argument("--formats", default=",".join(FORMATS), help="comma-separated")
    parser.add_argument("--orders", default=",".join(ORDERS), help="severity,stdf")
    parser.add_argument("--problems", action="store_true",
                        help="the short pre-production check instead of the full book")
    parser.add_argument("--max-tests", type=int, default=14,
                        help="findings shown in --problems mode")
    parser.add_argument("--dashboard", action="store_true",
                        help="also write the interactive six-tab HTML")
    parser.add_argument("--skip-ingest", action="store_true", help="the store is already built")
    args = parser.parse_args(argv)

    if args.skip_ingest:
        conf = cfg.load(Path(args.config))
        service, docs = Service(conf), _docs_in(conf)
    else:
        if not (args.a and args.b):
            print("error: --a and --b are required unless --skip-ingest", file=sys.stderr)
            return 1
        service, docs = _ingest(Path(args.config), [args.a, args.b])
    if len(docs) < 2:
        print(f"error: need two STDF documents in the store, found {len(docs)}", file=sys.stderr)
        return 1
    doc_a, doc_b = docs[0], docs[1]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mode = "problems" if args.problems else "full"
    written = 0
    for order in [o.strip() for o in args.orders.split(",") if o.strip()]:
        for fmt in [f.strip() for f in args.formats.split(",") if f.strip()]:
            started = time.time()
            try:
                result = service.stdf_audit_report(
                    doc_a, doc_b, fmt=fmt, base_url="", mode=mode, sort=order,
                    label_a=args.label_a or f"document {doc_a}",
                    label_b=args.label_b or f"document {doc_b}",
                    max_tests=args.max_tests, plot_cap=1_000_000,
                )
            except Exception as err:  # noqa: BLE001 - one format failing must not lose the rest
                print(f"  {order:8} {fmt:5} FAILED — {type(err).__name__}: {err}")
                continue
            suffix = Path(result["filename"]).suffix
            dest = out / f"stdf-{mode}-{order}{suffix}"
            shutil.copy(result["path"], dest)
            written += 1
            print(f"  {order:8} {fmt:5} {dest.stat().st_size / 1e6:6.1f} MB  "
                  f"{time.time() - started:5.0f}s  {dest}")

    if args.dashboard:
        started = time.time()
        dest = out / "stdf-dashboard.html"
        dest.write_text(service.stdf_audit(doc_a, doc_b).get("html", ""), encoding="utf-8")
        print(f"  dashboard      {dest.stat().st_size / 1e6:6.1f} MB  "
              f"{time.time() - started:5.0f}s  {dest}")

    print(f"\n{written} book(s) in {out}")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
