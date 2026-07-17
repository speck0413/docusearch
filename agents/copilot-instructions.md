# docusearch — instructions for GitHub Copilot

A **docusearch** MCP server exposes the team's technical docs + code for grounded
answers. It connects over **HTTP MCP** at `http://localhost:8321/mcp` (see
`agents/mcp.json`); the human starts it with `docusearch serve`.

## Tools (stable names)

- `search_docs(queries, top_k=10)` — pass a **list** of queries (batch, one call).
  Each hit has a `citation` (`D:<doc>#<chunk>`), a clickable `url`, and a `snippet`.
- `get_document(doc_id, chunk=None)` — metadata + chunks.
- `related_documents(doc_id, direction="both")` — linked / linking docs.
- `catalog_stats()` — counts + embedding model.

## Citations (required)

End every catalog-sourced sentence with `[D:<doc_id>#<chunk_id>]` (from a hit's
`citation`). Mark anything not found in the catalog `[GK]`. Never fabricate a `[D:...]`
tag — reports refuse citations that weren't in the evidence set.

## Workflow

1. Batch a few focused queries into one `search_docs` call.
2. Open the best hits with `get_document`; follow `related_documents`.
3. Write short factual sentences, each ending in `[D:...]` or `[GK]`.
4. Generate deliverables through the report endpoint/tool, not by hand.

If nothing relevant comes back, say so and answer `[GK]`; do not guess a citation.
