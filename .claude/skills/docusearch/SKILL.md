---
name: docusearch
description: Answer a question from the docusearch catalog and produce a cited, themed report (md/html/html-slide/pdf/docx/pptx/xlsx) saved on the server. Use when the user asks something that should be answered from the ingested documentation, or asks for a docusearch report. Exposes a level-of-effort knob (1–10, default 5).
---

# docusearch research + report

Answer the user's question **only** from the docusearch catalog and render a themed, cited
report. Search goes through the **docusearch MCP server**; you never rely on prior knowledge
of the domain — discover the domain's terminology from the search results themselves.

## Ground rules

- **Cite everything.** Every claim taken from the catalog ends with `[D:<doc_id>#<chunk_id>]`;
  anything that is general knowledge (not in the catalog) ends with `[GK]`. Cite the exact
  `(doc_id, chunk_id)` the fact came from.
- **Don't assume what acronyms mean.** e.g. "PA" might be *Protocol Aware*, not power
  amplifier — let the retrieved documents define the terms. If the catalog doesn't cover
  something, say so plainly instead of inventing it.
- **Batch your searches.** `search_docs` takes a *list* of queries — send all the phrasings
  for a round in one call, never one at a time.

## Effort

The user picks an effort level (default **medium**):

| Effort | Behavior |
|--------|----------|
| **low** | One `search_docs` call (3–4 phrasings); a short, direct, cited answer. |
| **medium** *(default)* | 6–8 phrasings in one `search_docs` call; read the hits; one follow-up batch to fill gaps; a structured multi-card report. |
| **high** | Many phrasings over several batched rounds; `get_document` for full chunk text and `related_documents` to follow leads; keep going until new searches surface nothing new. **Only dismiss a lead after a search confirms it's a dead end.** |

## Tools (docusearch MCP)

- `search_docs(queries: list[str], top_k=10)` → a **table**: `results[i]` is query i's rows,
  ranked best-first, columns named by `hit_fields` (`cite, locator, kind, snippet`). `cite` is
  the citation, quoted verbatim. Join a row to its document on the doc part of `cite`
  (`D:12#5` → `documents["12"]`, columns named by `doc_fields`). **Always pass a list.** Over
  4 queries per call, `top_k` clamps to 5.
- `get_document(doc_id, chunk=None)` → full text of a document or one chunk (use it to fill
  a card with real code / a full procedure, not just a snippet).
- `related_documents(doc_id, direction="both")` → cross-referenced docs (follow leads).
- `catalog_stats()` → sanity-check the catalog is populated.

## Workflow

1. **Discover + retrieve.** Plan phrasings for the effort level (synonyms, subtopics,
   how-to framings) and `search_docs` them in one batched call. Repeat per the level's
   rounds. For depth, `get_document` the strongest hits for full text; `related_documents`
   to follow references.
2. **Select evidence.** Collect the `(doc_id, chunk_id)` pairs whose text actually supports
   your answer — these become the report's `evidence`.
3. **Write the report as cards.** Group the answer into sections that match what the lookup
   surfaced. Pick the `kind` per card so it renders with the right icon/accent:
   `overview`, `procedure`, `code`, `hardware`, `config`, `test-program`, `warning`,
   `reference`. Put every catalog claim's `[D:doc#chunk]` inline in the prose; the renderer
   turns them into little superscript links to the References card.
4. **Render with `build_report`.** This is the terminal step and it is not optional — you are
   talking to a REMOTE docusearch server over MCP, so there is no local `docusearch` CLI to
   shell out to and no local file you can write the report into. The server renders it, saves
   it, and hands you back a URL.

   ```
   build_report(spec, fmt="<the configured format>")
   ```

   It **verifies every citation against your evidence and refuses hallucinated ones**, in every
   format. It returns:

   ```json
   {"fmt": "html", "filename": "pa-overview-<run>.html",
    "url": "http://<server>/v1/reports/pa-overview-<run>.html", "bytes": 48213}
   ```

   **Give the user the `url`** — it is a direct, clickable link to the file. For `md` and
   `html` the text also comes back as `report` if you want to quote from it; the binary
   formats deliberately do not inline their payload.

   `fmt` is one of `md` · `html` · `html-slide` · `pdf` · `docx` · `pptx` · `xlsx`.
   `html-slide` is a keyboard-navigated deck (PowerPoint's own keys). Use the format the
   operator configured; do not substitute another because one seems easier.

   The `spec` (JSON object):
   ```yaml
   title: "Protocol Aware (PA) — Overview"
   subtitle: "How PA drives serialized bus protocols in the test program"
   request: "Give me a comprehensive overview of controlling PA"   # the user's verbatim ask
   requested_by: "<the requesting user>"                            # who the report is for
   model: "<your model id, e.g. claude-sonnet-5>"                   # what generated it
   audience: ["engineering"]
   sections:
     - heading: "Overview"
       kind: overview
       body: |
         Protocol Aware (PA) lets a pattern drive serialized bus protocols [D:101#2].
     - heading: "Example — register write"
       kind: code
       body: |
         ```
         pa.frame("WRITE", addr=0x40, data=0x1F)
         ```
         The captured bytes are compared to the expected pattern [D:318#2].
   evidence:
     - [101, 2]
     - [318, 2]
   ```

   Set `request`, `requested_by`, and `model` — they populate the report's provenance header.
   `sources` defaults to the config's document stores. You do **not** set references: the
   builder links each one to the original document automatically (store — title — heading),
   so leave that to the tool.

5. **Always include a `trace`** so the reader can see how the report was produced (it renders
   as a collapsed "Generation log" and is NOT citation-verified — it's a log, not claims):

   ```yaml
   trace:
     prompt: "<the question you were given + effort level>"
     queries:            # every search phrasing you ran, in order
       - "controlling the power amplifier"
       - "PA bias register"
     retrieved:          # the notable hits you considered (cited or not)
       - "[D:1420#3] PA Bias — the PABIAS register sets quiescent current…"
     reasoning: "Why you structured the report as you did; leads you followed or dropped."
   ```

6. If `build_report` returns `{"error": "HALLUCINATED_CITATION", …}`, you cited a pair that
   isn't in `evidence` — add it (if you really retrieved it) or drop the claim, then re-render.
   Never invent an evidence entry. An `{"error": "EXPORT", …}` means the server is missing that
   format's writer; report the message verbatim rather than silently falling back to another
   format.

## Invoking as a subagent

To run this in a clean context, spawn a subagent that has the **docusearch MCP** connected
and this skill available, and hand it the user's question verbatim plus the effort level.
Give it no other domain instructions — the skill and the catalog are the only sources of
truth. Have it return the report **URL** + metadata (searches run, chunks cited, mode), not the
document contents.
