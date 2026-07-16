#!/usr/bin/env bash
# Minimal end-to-end smoke: init -> dry-run plan -> gate, checking artifacts.
# POSIX counterpart of scripts/smoke.ps1. Run after `pip install -e ".[dev]"`.
set -euo pipefail

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
cd "$work"

python -m docusearch init
python -m docusearch ingest --dry-run
python -m docusearch gate 1

test -f docusearch.yaml || { echo "FAIL: config not written"; exit 1; }
test -f tmp/gates/GATE-1-phase-1.md || { echo "FAIL: gate file not written"; exit 1; }
ls tmp/logs/*.jsonl >/dev/null 2>&1 || { echo "FAIL: log not written"; exit 1; }

echo "smoke OK"
