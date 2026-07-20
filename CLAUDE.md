# docusearch — instructions for AI coding agents

You have a **docusearch** MCP server: a local catalog of the team's technical
documentation and code. Use it to ground your answers in real sources. It connects over
**HTTP MCP** (streamable) at `http://localhost:8321/mcp`; the human runs `docusearch serve`
first. Claude Code reads the connection from `.mcp.json`; VS Code / Copilot from
`.vscode/mcp.json`.

## Tools (names are stable — depend on them)

- `search_docs(queries: list[str], top_k=10)` — **always pass a list**; batch related
  queries in one call instead of many round-trips. Returns `{documents, results}`: each
  ranked hit carries a `citation` (`D:<doc>#<chunk>`), a `snippet`, and a `locator`, while
  the title, path and clickable `url` live once per document under `documents[doc_id]` —
  join the two on `doc_id`. Batches over 4 queries clamp `top_k` to 5, and a `truncated`
  key means the reply hit its size budget: search again with fewer queries.
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
