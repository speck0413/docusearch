# docusearch

An enterprise documentation catalog: ingest technical documentation and code, index
it for fast **local** search (BM25 + optional vector hybrid), and expose it to AI
agents (Claude Code, Claude Desktop, Copilot) over an HTTP MCP server and to humans
over a REST API — with full citations, cross-document relations, and image retention.

Target users are electrical engineers who write code, not software engineers. Every
interface biases toward simple, discoverable, low-ceremony use. Runs on Windows 11
laptops and on servers; heavy ingestion belongs on the server, clients stay light.

> Ingests HTML, PDF, DOCX, and Markdown; BM25 with optional local **hybrid** (vector +
> RRF) search, all with citations/relations/image retention; `serve` exposes REST + MCP.
> See [`GETTING_STARTED.md`](GETTING_STARTED.md) for the full operator and user guide
> (PowerShell **and** bash).

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

## Performance & scale

Measured on a **14,938-document / 126,469-chunk** real corpus (mixed HTML/PDF/DOCX/Markdown),
`bge-small-en-v1.5`, Apple-Silicon **CPU, single thread** (no GPU). Query latency:

| Operation | p50 | p95 | Throughput (1 core) | Where this sits vs the industry |
|---|---|---|---|---|
| **BM25** (keyword) | 7 ms | **13 ms** | 148 q/s | Interactive search wants < 100 ms p95 (Nielsen's 0.1 s "feels instant"). Enterprise keyword engines (Elasticsearch/Solr) typically land 10–100 ms — this is at the **fast end**. |
| **Hybrid** (BM25 + vector, RRF) | 17 ms | **24 ms** | 59 q/s | < 50 ms p95 is considered **great** for hybrid. The cost here is the CPU query-embedding (~10 ms); a GPU/MPS cuts that to ~1–2 ms. |
| **Vector ANN** alone (HNSW) | 0.1 ms | **0.1 ms** | 15,900 q/s | HNSW is best-in-class; sub-millisecond at 10⁵–10⁶ vectors is the expected, and this hits it. |
| **Federation**, fan-out over 3 stores | 15 ms | 17 ms | — | Fan-out is ~linear in the number of member stores searched. |
| **Federation**, scoped to 1 store (`--stores`) | 5 ms | 6 ms | — | Scoping a query to the relevant store(s) cuts latency **~3×** — skip what you don't need. |

Footprint for that same 15k-doc corpus:

| Resource | Value | Context |
|---|---|---|
| On-disk index | **560 MB** (347 MB SQLite + 213 MB HNSW) ≈ 4.4 KB/chunk | Compact and linear; a laptop-sized index. |
| Peak RAM, warm hybrid | **~985 MB** (bge-small ~130 MB + HNSW 213 MB resident + PyTorch) | A single modest box. **BM25-only** drops the model + ANN entirely → a few hundred MB, no PyTorch. |
| Ingest (BM25) | ~60 docs/s (601 docs → 9.8 s, Gate 4d) | Embedding adds model-dependent time (see the model benchmark above; `bge-small` ≈ thousands of chunks/s). Ingest is incremental — unchanged files are skipped by content hash. |

**Why this is a lightweight central server.** At **p95 24 ms** hybrid / **~60 q/s per core**, an
ordinary 8-core box sustains ~**480 hybrid queries/second** — far above the handful of queries per
second a few-hundred-person organization generates at peak, and **no GPU is required**. The whole
index for a mid-size corpus fits in ~1 GB of RAM; a BM25-only deployment is lighter still. Vertical
scale (more cores) and horizontal scale (federation, below) both apply. Everything is deterministic
and offline-capable once the embedding model is cached.

> Reproduce: query latency + memory with the perf harness on any ingested store; the model-quality
> table above with `python -m harness.embed_benchmark`; federation parity with `python -m
> harness.federated --corpus <dir> --shards 3`.

## Central MCP server (one server for the whole company)

`docusearch serve` exposes the catalog over **REST + MCP** on one lightweight process (§10). The
MCP is designed to be a *central* company endpoint an AI connects to:

- **Minimal registration.** Every MCP tool carries a one-line description, so connecting the server
  costs almost no context. An agent that actually needs docusearch calls **`help()` first** — it
  returns the full research + cited-report workflow on demand (identical to the local skill), so the
  detailed instructions only load when they're used.
- **Search tools** expose the search-relevant knobs: `search_docs(queries, top_k, prefix, stores,
  bm25_only, roles)`, plus `list_stores()`, `get_document()`, `related_documents()`,
  `catalog_stats()`, and `build_report()` (which verifies every citation and refuses hallucinated
  ones). Reports built through the MCP are byte-identical to the CLI's, except references link to
  the server's `/v1/documents` URLs instead of local `file://` paths.
- **Federated, with per-query scoping.** Point one config at several member stores
  (`federation:` — e.g. `python`, `rust`, `internal`, `acme`), and a query fans out across all of
  them and merges as if it were one store. Tell it `--stores acme` (CLI) or `stores=["acme"]` (MCP)
  to search **only** that store — the "which document stores" control, which also cuts latency.

```yaml
# federation.yaml — a central server over several independent stores
federation:
  - { name: python,   config: /srv/docusearch/python.yaml }
  - { name: rust,     config: /srv/docusearch/rust.yaml }
  - { name: internal, config: /srv/docusearch/internal.yaml }
  - { name: acme,     config: /srv/docusearch/acme.yaml }
```
```bash
docusearch serve --config federation.yaml            # REST + MCP for the whole company
docusearch search --config federation.yaml --stores acme "match loop single bit"   # scope to ACME
```

### Private stores & access control

A store is **public** (anyone on the server) or **private** (a whitelist). Set it in the store's
config — omit it and the store is public:

```yaml
access:
  visibility: private
  allowed_users:  [alice, bob]
  allowed_groups: [engineering]
```

The requester's identity comes from the **`X-Docusearch-User`** (and `X-Docusearch-Groups`) HTTP
header on REST, or the `user`/`groups` arguments on the MCP `search_docs` tool. A private single
store returns **403** to anyone not whitelisted; in a federation a private member simply **drops out**
of the results and looks "unknown" if explicitly scoped — its existence isn't leaked. Heavier
isolation is just a **dedicated store** (a federation member) with its own whitelist. The local CLI
(the operator) is unrestricted.

### Adding documents (write API)

```bash
# a server-side folder or a .zip/.tar.gz archive (uncompressed server-side), labelled + attributed
curl -X POST localhost:8321/v1/ingest -H 'X-Docusearch-User: alice' \
     -H 'content-type: application/json' \
     -d '{"path": "/inbound/vendor-2025q3", "store": "vendor", "label": "vendor-2025q3"}'

# multipart upload of an archive
curl -X POST localhost:8321/v1/ingest/upload -H 'X-Docusearch-User: alice' \
     -F file=@docs.zip -F store=internal -F label=q3-notes
```

`POST /v1/ingest` (folder or archive path) and `POST /v1/ingest/upload` (multipart `.zip`/`.tar.gz`)
add documents to a chosen store (`store` = a federation member such as `vendor`/`internal`/`user`),
tagged with a `label` and attributed to the submitting username (**required** — 401 without it).
Archives are uncompressed with zip-slip / tar-traversal rejected. `POST /v1/feedback` records user
feedback. The MCP exposes the same as the `ingest_docs` and `submit_feedback` tools.

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
