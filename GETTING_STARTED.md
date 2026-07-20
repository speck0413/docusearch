# Getting started with docusearch

docusearch turns a folder of documentation into a local, cited search service that both
**people** (CLI / REST) and **AI agents** (MCP) can query. This guide has two halves:

- **[Part 1 — Administrators](#part-1--administrators)**: stand up a catalog and an MCP server, connect clients, manage models, set up per-user permissions.
- **[Part 2 — Users](#part-2--users)**: search, restrict results to your categories, generate reports, and drive the research agent.

> Everything is local. The embedding model downloads once, then the whole pipeline runs
> offline. Nothing about your documents leaves the machine.

## Table of contents

- [Part 1 — Administrators](#part-1--administrators)
  - [1.1 Install](#11-install)
  - [1.2 Point at your documents and ingest](#12-point-at-your-documents-and-ingest)
  - [1.3 Run the MCP + REST server](#13-run-the-mcp--rest-server)
  - [1.4 Enable one or more MCP servers](#14-enable-one-or-more-mcp-servers)
  - [1.5 Connect the clients (Claude Code, Claude Desktop, VS Code, Copilot)](#15-connect-the-clients)
  - [1.6 Permissions: who can see which categories](#16-permissions-who-can-see-which-categories)
  - [1.7 Manage embedding models on disk](#17-manage-embedding-models-on-disk)
  - [1.8 Keeping the catalog fresh](#18-keeping-the-catalog-fresh)
- [Part 2 — Users](#part-2--users)
  - [2.1 Search from the CLI](#21-search-from-the-cli)
  - [2.2 Restrict searching to specific categories / tags](#22-restrict-searching-to-specific-categories--tags)
  - [2.3 Generate reports in various styles](#23-generate-reports-in-various-styles)
  - [2.4 The research agent + exhaustiveness levels](#24-the-research-agent--exhaustiveness-levels)
- [FAQ](#faq)

---

# Part 1 — Administrators

## 1.1 Install

**Quickest — the wizard.** From the project folder, run:

```bash
./install.sh
```

It creates the `.venv`, installs docusearch, writes a starter `docusearch.yaml`, asks
whether your MCP server is **local or remote**, writes the matching client configs
(`.mcp.json`, `.vscode/mcp.json`), and prints how to connect each client. On Windows, use
the manual steps below in PowerShell.

**Manual.**

```bash
python3 -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[embeddings,server]"     # vector search + REST/MCP server
docusearch --version
```

Extras: `embeddings` (vector search: torch + sentence-transformers + hnswlib),
`server` (FastAPI + uvicorn + MCP), `client` (httpx). `.[dev]` installs everything.

## 1.2 Point at your documents and ingest

```bash
docusearch init                            # writes a fully-commented docusearch.yaml
```

Edit the `sources:` block and the `embed.model`. Each source has a **`name`** (its
category label) and an **`audience`** list (who may see it — this is what permissions key
off, see §1.6):

```yaml
sources:
  - name: rf-docs                          # a category label (used for permissions + purge)
    version: "2024.3"                       # optional provenance tag
    location: "/path/to/rf-docs"
    include: ["**/*.html"]
    content_selector: ""                    # tighten later if snippets carry nav/menu junk
    audience: ["engineering", "rf-team"]    # who may see this category
  - name: test-procedures
    location: "/path/to/test-docs"
    audience: ["engineering", "test-eng"]
embed:
  model: "BAAI/bge-small-en-v1.5"           # 384-dim, good quality/size balance
  device: "auto"                            # Mac GPU (mps) / CUDA / cpu
```

Then ingest (a live progress bar shows files, then embedding; the JSONL log path prints so
you can `tail -f` it):

```bash
docusearch ingest
docusearch audit                            # counts + a loud list of anything skipped
```

## 1.3 Run the MCP + REST server

The MCP server speaks **HTTP**, so it must be running **before** you open Claude Code /
Claude Desktop / VS Code — a client connects to its URL, it does not launch it. Start it
with the helper (starts in the background, offers to restart if already running):

```bash
./start-server.sh                           # binds 0.0.0.0:8321 by default; start/restart
```

Or run it in the foreground directly:

```bash
docusearch serve                            # REST + MCP over HTTP on port 8321
docusearch serve --host 127.0.0.1 --port 9000   # override bind host/port
```

- **REST** (OpenAPI docs at `http://localhost:8321/docs`): `POST /v1/search`,
  `GET /v1/documents/{id}`, `GET /v1/images/{sha}`, `GET /v1/relations/{id}`,
  `POST /v1/reports`, `GET /v1/health`.
- **MCP** (streamable HTTP) at `http://localhost:8321/mcp`. Tools: `search_docs`,
  `get_document`, `related_documents`, `catalog_stats`.

## 1.4 Enable one or more MCP servers

Each `docusearch serve` process serves **one catalog**. To expose several document sets
(e.g. RF docs and firmware docs as separate MCP servers), run one process per catalog on
its own port with its own config, then register each as a separate MCP server in the
client:

```bash
docusearch serve --config rf.yaml       --port 8321   # catalog A -> /mcp on 8321
docusearch serve --config firmware.yaml --port 8322   # catalog B -> /mcp on 8322
```

```jsonc
{ "mcpServers": {
    "rf-docs":       { "type": "http", "url": "http://localhost:8321/mcp" },
    "firmware-docs": { "type": "http", "url": "http://localhost:8322/mcp" }
} }
```

(Run each on a server/VM and use its hostname instead of `localhost` for shared access.)

## 1.5 Connect the clients

The connection is always the same HTTP MCP URL (`http://<host>:<port>/mcp`); each client
just registers it differently. Two ready-to-copy configs ship in the repo: `.mcp.json`
(Claude Code / Claude Desktop) and `.vscode/mcp.json` (VS Code / Copilot).

- **Claude Code (CLI):**
  ```bash
  claude mcp add --transport http docusearch http://localhost:8321/mcp
  claude mcp list                          # verify it connected
  ```
- **Claude Desktop:** Settings → Developer → Edit Config, add the `mcpServers` entry from
  §1.4 (or copy `.mcp.json`), then restart the app.
- **VS Code (Claude / Copilot MCP):** the repo's `.vscode/mcp.json` registers it
  automatically; or add the same `{ "type": "http", "url": ".../mcp" }` shape to the
  extension's MCP settings, then reload the window.
- **GitHub Copilot:** reads `CLAUDE.md` as always-on instructions automatically (same as
  Claude Code — no separate Copilot file needed) and connects via `.vscode/mcp.json`.

A single `CLAUDE.md` at the repo root is the instruction file for **both** Claude Code and
VS Code / Copilot. It carries the **mandatory citation grammar** — every catalog fact ends
in `[D:<doc>#<chunk>]`, everything else in `[GK]` — and the report builder refuses citations
outside the retrieved evidence, so agents can't fabricate refs. The bundled skills under
`.claude/skills/` are likewise read by both tools; the only tool-specific files are the two
MCP configs (`.mcp.json` for Claude, `.vscode/mcp.json` for VS Code / Copilot).

## 1.6 Permissions: who can see which categories

Permissions are **cooperative audience filtering** (honest scoping for trusted internal
users — not a cryptographic boundary; anyone with direct DB/file access can still read it).

1. **Tag each category** with the audiences allowed to see it (the `audience:` list per
   source, §1.2). Think of audiences as roles/tags: `engineering`, `rf-team`, `test-eng`,
   `finance`, …
2. **Give each user their roles** via the `DOCUSEARCH_ROLES` environment variable on the
   process that queries (CLI, agent, or client). A search only returns documents whose
   `audience` intersects the caller's roles. No roles set ⇒ no audience filtering applied.

Example — user X may see categories tagged `rf-team` and `test-eng`:

```bash
# bash
export DOCUSEARCH_ROLES=rf-team,test-eng
# PowerShell
$env:DOCUSEARCH_ROLES = "rf-team,test-eng"
```

Any `docusearch search` (or MCP/REST query) from that shell is now scoped to those
categories. For agents, set `DOCUSEARCH_ROLES` in the environment of the MCP server or the
client process that owns the user's session.

## 1.7 Manage embedding models on disk

```bash
docusearch models                          # list downloaded models + sizes + cache path
```

Models cache under `~/.cache/huggingface/hub` (override with `HF_HOME`). Delete an unused
one with `huggingface-cli delete-cache` (interactive) or by removing its
`models--<org>--<name>` folder.

**Switching the catalog's model:** an index holds vectors from exactly one model. After
changing `embed.model`, run `docusearch ingest --reembed` to rebuild the vectors (cheaper
than `--force`, which also re-parses every file). Switching without `--reembed` is refused
with guidance rather than mixing embedding spaces.

## 1.8 Keeping the catalog fresh

```bash
docusearch ingest                          # incremental: only changed files (by SHA-256)
docusearch ingest --force                  # full rebuild: re-parse + re-embed everything
docusearch remove <source-name>            # purge one category entirely (docs+vectors+images)
```

Large first ingests parse across all CPU cores automatically; cap it with
`DOCUSEARCH_INGEST_WORKERS=N` (`1` = serial).

---

# Part 2 — Users

## 2.1 Search from the CLI

```bash
docusearch search "how do I configure the watchdog timer"
docusearch search "spi timing" --top-k 20
docusearch search "ZQX7734" --prefix                 # partial / prefix terms
docusearch search "clock control" --json             # structured output (for scripts/agents)
```

Each hit shows a citation `[D:<doc>#<chunk>]`, the heading-path locator, a snippet, a
score, and whether it ran in `hybrid` or `bm25` mode.

## 2.2 Restrict searching to specific categories / tags

Two independent axes:

- **By audience/tag** — set `DOCUSEARCH_ROLES` to the categories you're entitled to (see
  §1.6). Results are limited to documents whose `audience` intersects your roles:
  ```bash
  DOCUSEARCH_ROLES=rf-team docusearch search "power amplifier bias"
  ```
- **By catalog/source** — keep a config per document set and pick one with `--config`, so a
  query only ever touches that catalog:
  ```bash
  docusearch search "calibration" --config rf.yaml
  ```

## 2.3 Generate reports in various styles

Reports are produced **by code** (never free-typed), so every claim is a verified citation.
Provide a small YAML spec — a title, a body whose catalog claims end in `[D:doc#chunk]`
(general knowledge in `[GK]`), and the `evidence` (the `(doc, chunk)` pairs you retrieved):

```yaml
# answer.yaml
title: "Watchdog timer configuration"
audience: ["engineering"]
sources: ["rf-docs"]
body: |
  ## Enabling
  The watchdog is enabled by setting WDT_EN [D:812#2]. It resets the core after the
  timeout [D:812#3]. General background on watchdog timers [GK].
evidence:
  - [812, 2]
  - [812, 3]
```

```bash
docusearch report --spec answer.yaml --format html --out watchdog.html   # styled HTML
docusearch report --spec answer.yaml --format md   --out watchdog.md      # Markdown
```

The renderer **refuses** any `[D:...]` not in `evidence`, so a report can't cite something
that wasn't actually retrieved. (Reports are also available over REST at `POST /v1/reports`.)

## 2.4 Research questions + exhaustiveness levels

For a question you want answered end-to-end — batched searches, a synthesized cited
answer, and a rendered report — just ask any connected agent (Claude Code or Copilot). The
generic prompt in `CLAUDE.md` already tells it to batch its searches and cite every claim;
both tools also pick up the bundled **`docusearch` skill** (`.claude/skills/docusearch`),
which drives this with an explicit level-of-effort knob (1–10, **default 5**):

| Level | Use it for | Behavior |
|-------|-----------|----------|
| **1** | a quick lookup | one batched search, concise answer |
| **5** *(default)* | most questions | 6–8 angles batched + one gap-fill pass |
| **10** | complex, cross-domain | 12–20 angles over several rounds; follows every lead, only dismisses one after verifying it's a dead end |

It **always batches** its queries (all phrasings for a round go in one
`search --batch-file ... --json` call — 10 queries batched is far faster than 10 separate
calls, which each reload the model).

---

# FAQ

**Do I need a GPU?** No. BM25 keyword search needs no model at all (`embed.model: none`).
Hybrid (vector) search downloads a model once; `device: auto` uses your GPU (Apple `mps`
or CUDA) if present, else CPU.

**Hybrid vs BM25 — which am I getting?** Every search prints its mode. You get `hybrid`
when the catalog has vectors and the query model matches the index model; otherwise `bm25`.

**A search says `bm25` even though I embedded.** The query-time model must match the index
model. If they differ, docusearch falls back to BM25 with a warning rather than mixing
embedding spaces. Re-ingest with `--reembed` after a model change.

**Ingest was interrupted / a model switch errors.** Run `docusearch ingest --reembed` to
rebuild vectors cleanly, or point `paths.db_path` at a fresh database.

**How do I see what got skipped?** `docusearch audit` prints a loud list (too-short pages,
`content_selector` misses, zero-chunk docs, parse errors, unresolved links).

**Snippets carry nav/menu junk.** Set `content_selector` to the article container (open a
file in a browser → Inspect, e.g. `main.article`) and `strip_selectors: ["header","nav"]`,
then re-ingest with `--force`.

**Are permissions secure?** They're cooperative (audience/role scoping) for trusted
internal users, not an access-control boundary. Anyone with direct file/DB access can read
everything; use OS-level permissions on the catalog for hard boundaries.

**Where does generated output go?** Under the config's `tmp_dir` (reports, logs, gate
files). Nothing is written outside your project.
