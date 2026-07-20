# docusearch — instructions for AI coding agents

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
  field from a search hit). The tag resolves to a clickable source URL.
- `[GK]` — general knowledge you did **not** find in the catalog. Any sentence you can't
  back with a `[D:...]` chunk must be marked `[GK]`. Do not invent `[D:...]` tags — a
  report will be **refused** if it cites a chunk that wasn't in your search results.

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
