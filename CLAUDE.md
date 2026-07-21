# docusearch — instructions for AI coding agents

> Using docusearch to answer questions? This file is for you — read on.
> **Working ON the docusearch codebase?** Read [`DEVELOPERS.md`](DEVELOPERS.md) first; it
> covers the extension seams, the invariants, and the traps. This file is the end-user
> agent contract and ships to every install.

You have a **docusearch** MCP server: a local catalog of the team's technical
documentation and code. Use it to ground your answers in real sources. It connects over
**HTTP MCP** (streamable) at `http://localhost:8321/mcp`; the human runs `docusearch serve`
first. Claude Code reads the connection from `.mcp.json`; VS Code / Copilot from
`.vscode/mcp.json`.

## Tools (names are stable — depend on them)

- `search_docs(queries: list[str], top_k=10)` — **always pass a list**; batch related
  queries in one call instead of many round-trips. The reply is a **table**, not a list of
  objects: `results[i]` holds query i's rows, ranked best-first, with columns named by
  `hit_fields` (`cite, locator, kind, snippet`). `cite` is the citation to quote verbatim.
  For a row's title/path, take the doc part of `cite` (`D:12#5` → `12`) and read
  `documents["12"]`, whose columns are named by `doc_fields`. Batches over 4 queries clamp
  `top_k` to 5. Each fact is stated once — that is what keeps a batched reply small.
- `get_document(doc_id, chunk=None)` — full metadata + chunks for one document.
- `related_documents(doc_id, direction="both")` — linked / linking documents.
- `catalog_stats()` — counts + which embedding model the index uses.

## Citations are mandatory

Every factual sentence you write from the catalog **must end with a citation tag**:

- `[D:<doc_id>#<chunk_id>]` — the fact came from that catalog chunk (copy the `citation`
  field from a search hit).
- `[GK]` — general knowledge you did **not** find in the catalog. Any sentence you can't
  back with a `[D:...]` chunk must be marked `[GK]`. Do not invent `[D:...]` tags — a
  report will be **refused** if it cites a chunk that wasn't in your search results.

**A citation must always resolve to the original source document — never a bare id.**
`D:<doc_id>#<chunk_id>` is an internal key, **not** a source, and the reader must never be
shown one. In the report body, render each citation as a numbered footnote link to the
References entry (e.g. `<sup><a href="#ref3">[3]</a></sup>`) — not as `[D:...]` text.
Always end with a **References** section where every entry names the real document:
*title — locator — path* (link the `url` when useful). A report that shows raw `D:` ids,
or whose references are ids instead of documents, is **not acceptable**.

## Report output format

<!-- docusearch:output-format:start -->
**Reports are files, and the server writes them — you deliver the link.**

1. Call `report_format()` **before you draft.** It returns the operator's configured
   default (`reports.default_format` in `docusearch.yaml`) and how to author for that
   target — a deck needs short bullets, a spreadsheet one fact per row, a document prose.
   The renderer cannot invent structure you did not write.
2. **If the requester names a format, that wins** ("make me a PowerPoint" → `pptx`); the
   configured default applies only when they did not say.
3. Call `build_report(spec, fmt=...)`. It verifies every citation, saves the file under
   `tmp_dir/reports/`, and returns `{fmt, filename, url, bytes}` — **give the user the
   `url`.** Do not re-write the file yourself and never hand-format a report.

One report, one file. All formats (`md`, `html`, `html-slide`, `pdf`, `docx`, `pptx`,
`xlsx`) come back the same way, so this step never changes with the format.
<!-- docusearch:output-format:end -->

(`install.sh` rewrites the block above to whatever output format the operator chose.)

## Workflow

1. Turn the question into a few focused queries; call `search_docs` **once** with the list.
2. Read the top snippets; open promising ones with `get_document`; follow
   `related_documents` when a source points elsewhere.
3. Answer in short factual sentences, each ending in `[D:...]` or `[GK]`.
4. For a written deliverable, don't hand-format it — send your cited body to the
   `/v1/reports` endpoint (or the `create_report` tool), which renders a consistent banner
   + numbered references and verifies every citation.

Prefer the catalog over guessing. If `search_docs` returns nothing relevant, say so and
mark the answer `[GK]` — never fabricate a citation.
