# docusearch

An enterprise documentation catalog: ingest technical documentation and code, index
it for fast **local** search (BM25 + optional vector hybrid), and expose it to AI
agents (Claude Code, Claude Desktop, Copilot) over an HTTP MCP server and to humans
over a REST API — with full citations, cross-document relations, and image retention.

Target users are electrical engineers who write code, not software engineers. Every
interface biases toward simple, discoverable, low-ceremony use. Runs on Windows 11
laptops and on servers; heavy ingestion belongs on the server, clients stay light.

> **Status:** Phase 0 (Foundation). See `docusearch-architecture.md` for the full
> contract, `WORKLOG.md` for current state, and `RUNBOOK-private-dataset.md` for the
> copy-paste operator guide.

## Quick start (dev)

```bash
python -m venv .venv
. .venv/bin/activate            # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
docusearch init                 # writes a fully-commented docusearch.yaml
docusearch ingest --dry-run     # shows the plan without touching the index
pytest                          # run the test suite
```

Run modes are selected in `docusearch.yaml`: `standalone`, `server`, or `client`.
