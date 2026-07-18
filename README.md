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
cd /path/to/docusearch/clone
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[embeddings]"        # vector search; use ".[dev]" if you also run tests
```

**2. Point it at your folder.** Run `docusearch init` to drop a fully-commented
`docusearch.yaml` in the current directory (every command reads `./docusearch.yaml`
unless you pass `--config PATH`), then edit these fields:

```yaml
paths:
  staging_dir: "./staging"
  db_path: "./catalog.db"
  tmp_dir: "./tmp"
sources:
  - name: my-docs
    version: "2024.3"                  # optional: which release of the docs this is
    location: "/path/to/docs"          # <-- your folder
    include:                           #     HTML only for now (PDF/DOCX/MD are a later phase)
      - "**/*.html"
    content_selector: ""               #     empty = whole page (tighten later if noisy)
    min_content_chars: 200
    audience:
      - "engineering"
embed:
  model: "BAAI/bge-large-en-v1.5"      # best quality; fine on 128 GB. Downloads ~1.3 GB once.
  device: "auto"                       #     auto = Mac GPU (Metal/mps), else CUDA, else cpu
  batch_size: 128
```

The file is plain YAML — open it in any editor and change the values. `name` is the label
you'll use to purge a source later (`docusearch remove <name>`); `version` is a free-text
provenance tag recorded on every document from that source.

> On Apple Silicon, `device: auto` runs the model on the Metal GPU (`mps`) — a big
> speed-up over `cpu`. Force a device with `cpu`, `cuda` (NVIDIA), or `mps` if you want
> to; for keyword-only search with no model download, set `embed.model: none`.

**3. Ingest and search:**
```bash
docusearch ingest        # downloads the model once, then indexes (incremental on re-runs)
docusearch audit         # sanity-check counts + a loud list of anything skipped
docusearch search "how do I configure the watchdog timer"
docusearch show 42       # print one document's chunks to eyeball extraction
```

`ingest` shows a live progress bar — `ingest: N/total` while parsing files, then
`embed: N/total` while embedding (the slow, GPU-bound part) — so a long run is visibly
alive, not hung. It also prints the JSONL log file it's writing (`tmp/logs/<date>.jsonl`);
`tail -f` it in another terminal to watch every event.

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
- **Switch the embedding model:** vectors are model-specific, so an index holds vectors
  from exactly one model. To change `embed.model` on an existing index, re-embed:
  ```bash
  docusearch ingest --reembed      # drops the old vectors, re-embeds with the new model
  ```
  `--reembed` keeps your parsed documents and only rebuilds vectors (cheaper than
  `--force`, which re-parses every file too). Switching the model **without** `--reembed`
  is refused with a message telling you exactly this — docusearch never mixes embedding
  spaces. A *query* against a mismatched model **falls back to BM25 with a warning**.
  `embed.model: none` disables vectors entirely (BM25-only).
- **Roles (the one runtime-only knob):** cooperative audience filtering comes from an
  environment variable, not the file, so you can scope results per invocation:
  ```bash
  DOCUSEARCH_ROLES=engineering,test-eng docusearch search "calibration"
  ```
- **Per-run flags:** `--top-k N`, `--prefix` (partial/prefix terms), `ingest --dry-run`
  (preview, touch nothing), `search --batch-file goldens.yaml --out report.md` (grade a
  set of queries).

## Housekeeping — purge a source, manage models

- **Remove everything from one source label** (documents, chunks, vectors, relations,
  images — and the ANN sidecar is rebuilt):
  ```bash
  docusearch remove delete_me_next          # prompts for confirmation
  docusearch remove delete_me_next --yes    # no prompt (scripts)
  ```
  Purging by `name` is why each source has a label. An unknown name is a no-op that lists
  the labels that *do* exist.
- **See / clean up downloaded models.** Embedding models are cached by Hugging Face under
  `~/.cache/huggingface/hub` (override with `HF_HOME`). List what's on disk and how much
  space each takes:
  ```bash
  docusearch models
  ```
  To delete an old model, run `huggingface-cli delete-cache` (interactive) or remove its
  `models--<org>--<name>` folder under that cache directory.
- **Heal a half-finished embed run.** If an `ingest` was interrupted mid-embed and a later
  run reports a model/dimension mismatch, `docusearch ingest --reembed` rebuilds the
  vectors cleanly.

## Choosing an embedding model (benchmark)

Benchmarked on a controlled corpus with human-authored ground truth (18 signal topics + 400
distractor docs). Two different jobs are reported separately, because averaging them hides
the tradeoff. Reproduce with `python -m harness.embed_benchmark`.

**1. Is turning on a model worth it? (product mode: hybrid BM25+vector, vs BM25-only)**

| Search | Semantic simple R@5 | Semantic complex R@5 | Exact-token top-1 | Est. embed time (342k ch) |
|--------|--------------------|----------------------|-------------------|---------------------------|
| BM25 only (no model) | 72% | 83% | **100%** | none |
| BM25 + all-MiniLM-L6-v2 (hybrid) | 56% | 89% | 44% | 1:09 |
| BM25 + bge-small-en-v1.5 (hybrid) | 67% | 94% | 28% | 1:26 |
| **BM25 + bge-large-en-v1.5 (hybrid)** | **83%** | **94%** | 22% | 10:38 |

**2. Model-only semantic quality (pure vector — ranks the models against each other)**

| Model | Dim | Simple R@5 | Complex R@5 |
|-------|-----|-----------|-------------|
| all-MiniLM-L6-v2 | 384 | 50% | 56% |
| bge-small-en-v1.5 | 384 | 67% | 83% |
| bge-large-en-v1.5 | 1024 | 72% | 78% |

**How to read this:**
- **Embeddings do help semantic/paraphrase queries.** `bge-large` hybrid beats BM25 on both
  semantic classes (83% vs 72% simple, 94% vs 83% complex) — questions phrased in different
  words than the docs. `bge-small` is close and embeds ~7× faster (1:26 vs 10:38 for a
  342k-chunk catalog); MiniLM is clearly the weakest.
- **Exact identifiers (register names, error codes, part numbers): keyword search wins,** and
  hybrid currently *regresses* it — top-1 falls 100% → 44/28/22% as the vector model gets
  stronger, because RRF fusion dilutes the exact BM25 hit with the vector half's neighbours.
  This is a **known bug to fix** (a BM25-exact-match should never be diluted). Today's
  workaround: `search.bm25_only: true`, or search the literal token.
- **Net:** with the exact-match bug fixed, hybrid would be ≥ BM25 on every axis. Until then,
  choose `bge-small`/`bge-large` when your users ask conceptual questions; keep BM25 for
  exact-identifier lookups. On large real corpora the semantic gains grow — this compact,
  keyword-friendly corpus is a conservative lower bound.

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
