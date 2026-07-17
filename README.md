# docusearch

An enterprise documentation catalog: ingest technical documentation and code, index
it for fast **local** search (BM25 + optional vector hybrid), and expose it to AI
agents (Claude Code, Claude Desktop, Copilot) over an HTTP MCP server and to humans
over a REST API — with full citations, cross-document relations, and image retention.

Target users are electrical engineers who write code, not software engineers. Every
interface biases toward simple, discoverable, low-ceremony use. Runs on Windows 11
laptops and on servers; heavy ingestion belongs on the server, clients stay light.

> **Status:** Phases 0–2 complete — HTML ingest, BM25, and local **hybrid** (vector +
> RRF) search, all with citations/relations/image retention. `serve` (REST + MCP) and
> other formats (PDF/DOCX/MD) are upcoming phases. See `docusearch-architecture.md` for
> the contract, `WORKLOG.md` for current state, and `RUNBOOK-private-dataset.md` for the
> full operator guide (PowerShell **and** bash).

## Run it on your documents (macOS)

The commands live in this project folder. Point it at your docs and search — indexing
is fully local; the embedding model downloads once, then everything runs offline.

**1. Set up (once):**
```bash
cd /Users/speck/Downloads/docusearch-kickoff
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[embeddings]"        # vector search; use ".[dev]" if you also run tests
```

**2. Point it at your folder** — write a config in one shot (edit the two ALL-CAPS bits):
```bash
cat > docusearch.yaml <<'YAML'
paths:
  staging_dir: "./staging"
  db_path: "./catalog.db"
  tmp_dir: "./tmp"
sources:
  - name: my-docs
    location: "/path/to/docs"          # <-- your folder
    include: ["**/*.html"]             #     HTML only for now (PDF/DOCX/MD are a later phase)
    content_selector: ""               #     empty = whole page (tighten later if noisy)
    min_content_chars: 200
    audience: ["engineering"]
embed:
  model: "BAAI/bge-large-en-v1.5"      # best quality; fine on 128 GB. Downloads ~1.3 GB once.
  device: "cpu"                        #     "none" for keyword-only (BM25), no download
  batch_size: 128
YAML
```

**3. Ingest and search:**
```bash
docusearch ingest        # downloads the model once, then indexes (incremental on re-runs)
docusearch audit         # sanity-check counts + a loud list of anything skipped
docusearch search "how do I configure the watchdog timer"
docusearch show 42       # print one document's chunks to eyeball extraction
```

Each result shows its citation `[D:<doc>#<chunk>]`, heading-path locator, snippet, and
score, plus whether the search ran in `hybrid` or `bm25` mode. If snippets carry nav/menu
junk, set `content_selector` (open a file in a browser → Inspect the article container,
e.g. `"main.article"`) and `strip_selectors: ["header","footer","nav"]`, then re-ingest
with `--force`.

## Change the config at runtime

External YAML is the only configuration mechanism (`docusearch.yaml`); the CLI never
needs code changes to retarget. Ways to shift it per run:

- **Different config file:** every command takes `--config PATH`, so you can keep several
  configs side by side and pick one at run time (a missing path is auto-created from the
  template):
  ```bash
  docusearch ingest --config bm25.yaml            # e.g. embed.model: none
  docusearch search "spi timing" --config hybrid.yaml
  ```
- **Different folder / selectors:** edit `location`, `include`, `content_selector`,
  `strip_selectors`, or `min_content_chars`, then re-ingest. Re-ingest is incremental
  (only changed files); add `--force` after changing extraction settings so existing
  docs are re-parsed:
  ```bash
  docusearch ingest --force
  ```
- **Switch the embedding model:** vectors are model-specific, so point at a **fresh**
  `db_path` (a new index) or re-ingest the current one with `--force`. Changing
  `embed.model` without re-embedding is refused — the index keeps its original model and
  a query with a mismatched model **falls back to BM25 with a warning** (it never mixes
  embedding spaces). `embed.model: none` disables vectors entirely (BM25-only).
- **Roles (the one runtime-only knob):** cooperative audience filtering comes from an
  environment variable, not the file, so you can scope results per invocation:
  ```bash
  DOCUSEARCH_ROLES=engineering,test-eng docusearch search "calibration"
  ```
- **Per-run flags:** `--top-k N`, `--prefix` (partial/prefix terms), `ingest --dry-run`
  (preview, touch nothing), `search --batch-file goldens.yaml --out report.md` (grade a
  set of queries).

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
