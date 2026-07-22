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
| **xhigh** | As high, plus: open every figure the hits carry (see *Figures*), and reconcile contradictions between documents rather than picking one. |
| **max** | As xhigh, with no stopping rule but exhaustion — pursue every lead to a confirmed dead end and state what you could not resolve. |

Levels use the **same words as the model harness's own effort setting** (`low`, `medium`,
`high`, `xhigh`, `max`) so one word means one thing everywhere. Put the level you ran at in
the report spec as `effort`, and the harness level (if you know it) as `model_effort` —
they render as separate chips in the banner.

## Figures — look before you cite, place with intent

Search hits carry an `img` list of figure shas.

1. `get_image(sha256)` → returns the figure itself, plus any cached description.
2. If it comes back with `cached: false`, you are the first to see it: read the image, then
   `describe_image(sha256, description)` with the CONTENT — states, values, channel/bit
   assignments, axes, labels — not "a diagram of X". The description is cached for everyone
   after you, so this cost is paid once per figure.
3. Use what you saw in your answer, and cite the image chunk normally.

Much of this catalog's real detail lives in figures, not prose — if a question looks
unanswerable from text, open the figures before concluding the documentation does not cover it.

**Showing a figure in the report is a deliberate choice, not automatic.** The report does NOT
dump every image you cited. To display one, add its `sha` to the `images` of the *section that
discusses it* — it renders inline, right after that section's text, renumbered in report order.
Include a figure only where it earns its place (a diagram the reader needs to follow the
point); a wall of figures at the end adds no value, and every source you cite is already
reachable from the References.

## Tools (docusearch MCP)

- **Figures ride along with search results.** When a hit's section contains an image, its row
  gains an `img` column of shas; `images[sha]` gives the caption and `GET img_base + sha`
  returns the file over plain HTTP. This works whether or not vision enrichment was ever run,
  because the figure is tied to the chunk by position, not by a generated description. Fetch one
  when a diagram would explain something better than a paragraph, and pass the shas as the
  **section's own `images` list** so the figure renders inside that section, beside the text it explains — not collected at the end. **A deck or document with no visuals
  is usually a missed opportunity** — an image is worth a thousand words of prose.
- `search_docs(queries: list[str], top_k=10)` → a **table**: `results[i]` is query i's rows,
  ranked best-first, columns named by `hit_fields` (`cite, locator, kind, snippet`). `cite` is
  the citation, quoted verbatim. Join a row to its document on the doc part of `cite`
  (`D:12#5` → `documents["12"]`, columns named by `doc_fields`). **Always pass a list.** Over
  4 queries per call, `top_k` clamps to 5.
- `get_document(doc_id, chunk=None)` → full text of a document or one chunk (use it to fill
  a card with real code / a full procedure, not just a snippet).
- `related_documents(doc_id, direction="both")` → cross-referenced docs (follow leads).
- `catalog_stats()` → sanity-check the catalog is populated.
- `get_image(sha256)` → **look at** a figure from a hit's `img` list (+ cached description).
- `describe_image(sha256, description)` → cache what you saw, once, for everyone after you.
- `report_format(fmt="")` → the target format's authoring rules + the configured default.

## Workflow

0. **Learn the target format first.** Call `report_format()`. It returns the operator's
   configured default and how to author for it.

   **If the user named a format, that wins.** "Create a PowerPoint…" → `pptx`, even when the
   configured default is something else; the default only applies when they did not say. Map
   plain words: PowerPoint/deck/slides → `pptx`, Word/document → `docx`, spreadsheet/Excel →
   `xlsx`, PDF → `pdf`, web page → `html`, browsable deck → `html-slide`.

   **Do this before you draft**, not after: the
   renderer lays out what you give it and cannot invent structure you did not write, so a
   section written as dense paragraphs becomes a wall of text on a slide. Shape the content to
   the destination —

   | Target | Write it as |
   |--------|-------------|
   | `md` · `html` | Prose. Full paragraphs, fenced code, markdown lists. |
   | `docx` · `pdf` | A document. Short paragraphs, markdown lists for steps. |
   | `pptx` · `html-slide` | A deck. One idea per section, 4–6 bullets of <15 words, written as `- ` list items. Split a long procedure into several sections rather than one dense one. |
   | `xlsx` | A grid. Every list item one self-contained fact; nested `  - ` items for detail. Paragraphs become unreadable single cells. |

1. **Discover + retrieve.** Plan phrasings for the effort level (synonyms, subtopics,
   how-to framings) and `search_docs` them in one batched call. Repeat per the level's
   rounds. For depth, `get_document` the strongest hits for full text; `related_documents`
   to follow references.
2. **Select evidence.** Collect the `(doc_id, chunk_id)` pairs whose text actually supports
   your answer — these become the report's `evidence`.
3. **Write the report — your way.** Two things are fixed because a reader needs them to trust
   the document: the **banner** (classification, request, provenance) and the **References**
   list. The builder adds both from your `evidence`; never write them yourself. Every catalog
   claim carries `[D:doc#chunk]` inline and general knowledge carries `[GK]`.

   **Everything else is your call** — how many sections, what to call them, how long they run,
   what order they take, whether to open with a summary or build to one, when a table beats
   prose, when a figure beats both. There is no required outline and no required length. A
   three-card answer to a small question is better than a padded ten. Pick each section's
   `kind` for how it should read (`overview`, `procedure`, `code`, `hardware`, `config`,
   `test-program`, `warning`, `reference`) — it sets the icon and accent.

   **A deck is for an audience, not a reader.** Code belongs on a slide only when it is a few
   lines someone could read from the back of a room; a long listing is kept off the slide
   automatically and preserved in the speaker notes. When the listing IS the deliverable, put it
   in the document formats and let the deck point at them — tailoring to the format means
   deciding what does *not* belong, not just reformatting the same content.

   **For a deck, the builder picks the slide** — a section with `images` is automatically laid
   out with the figure beside its points, which is usually the right answer. A section *may*
   override with `layout: statement` (one large centred claim) or `layout: compare` (two lists
   side by side), but these are **rare**: at most one or two in a whole deck, and only when the
   content genuinely is a single claim or a real this-versus-that. Their effect comes from
   contrast with ordinary slides, so if in doubt leave `layout` unset.

   Aim for something a person is glad to be handed, not a filled-in template.

   **Deliverables go IN the report, never to a local file.** Asked to produce a script, a
   config, or a test program, put the complete thing in a `code` section — the report is the
   deliverable and there is no local filesystem to write to. Do not call Write, do not ask for
   permission to save a file, and do not truncate the code to an excerpt: if the reader is meant
   to run it, ship all of it.
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

   Optional `theme` picks the look (`midnight` · `paper` · `slate` · `contrast`) — set it only
   if the requester asks; otherwise the operator's configured theme applies.

   Set `request`, `requested_by`, and `model` — they populate the report's provenance header.
   `sources` defaults to the config's document stores.

   **Never write a References section yourself.** Do not add a section titled References,
   Sources, or Further Reading, and do not add a "sources are listed below" note. The builder
   appends the real reference list (store — title — heading, linked) from your `evidence`. A
   hand-written one is duplicated in the output.

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
