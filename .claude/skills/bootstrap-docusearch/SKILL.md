---
name: bootstrap-docusearch
description: Stand up a docusearch config for a repo or folder the user points at — pick the right store_type, includes, and content hints, then verify it ingests. Use when a user has a pile of docs/code/data and wants docusearch pointed at it, or asks "how do I index this repo/folder?".
---

# Bootstrap a docusearch config for a repo or folder

Goal: from a folder (or git repo) the user names, produce a **working `docusearch.yaml`** and confirm
it ingests — with the **least manual config**. Do it in this order; stop as soon as it works.

## 1. Let the tool do the scan

```bash
docusearch bootstrap /path/to/repo --out docusearch.yaml
# a git URL works too — clone it first, or set it as the source `location` (it's cloned for you)
```

`bootstrap` walks the tree (skipping `.git`/`node_modules`/build dirs), categorises files into
**doc / code / data**, and writes a valid config with:
- `store_type` for the **dominant** content (`code` | `data` | `document`),
- `include` globs for the extensions actually present,
- inline **hints** as comments (see below).

Read the generated file with the user. It is a starting point, not the final answer.

## 2. Act on the hints the scan leaves

- **HTML** → `# run docusearch inspect <name>` : run it, then paste the suggested
  `content_selector` / `strip_selectors` into the source so page chrome is stripped.
  ```bash
  docusearch inspect <source-name>
  ```
- **PDF** → the comment shows how headings map from font sizes (e.g. `22pt→H1, 15pt→H2`). If that's
  wrong for these PDFs, say so — extraction is automatic, but the hint tells you what to expect.
- **code** → the languages found are listed; a git URL as `location` is cloned for you (auth is your
  own git, no token in the config).
- **"also found N …"** → the repo is **mixed** (e.g. code *and* docs). One config has one
  `store_type`. To index the secondary content too, make a **second** config for it and combine them
  under a `federation:` block. Don't force everything
  into one store.

## 3. Verify — always

```bash
docusearch ingest --config docusearch.yaml
docusearch audit --config docusearch.yaml     # counts + anomalies
docusearch search "something you know is in there"
```
If `audit` shows `zero_chunk_docs`, `content_selector_misses`, or an unexpected file count, fix the
`include`/selector and re-ingest. Only call it done when a real query returns the right result.

## Rules

- **Don't hand-write the config from scratch** when `bootstrap` + `inspect` can derive it — reuse the
  tools (they're generic and corpus-agnostic).
- **Never invent a `content_selector`** — get it from `docusearch inspect` against the real HTML.
- Keep the user's dominant intent: if they clearly want *code* analysis, keep `store_type: code` even
  if a few docs are present (index the docs as a federated member instead).
- Everything the tools generate is a **draft for the user to review** before ingest.
