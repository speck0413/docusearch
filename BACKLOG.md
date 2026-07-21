# Backlog

Living document. Each item says *why* it exists — usually a defect we actually hit, because a
reason that traces to an observed failure survives longer than a reason that does not.

---

## 1. IG-XL correctness checker (in progress)

**Why.** Reports refuse a hallucinated *citation*, but a generated engineering artifact was
never checked at all — so channel maps rendered happily with SPI pins straddling two PA
engines, every site on identical channels, and no scan type declared. Anything checkable
should be checked.

**Done**
- `igxl_rules.validate_channel_map` — scan type declared, channel used once, sites disjoint,
  site count, PA pins inside one mod-8 engine, even/odd (a **three-bit-only** rule).

**Next (agent-researched, see §5 for the method)**
- Full rule sweep for **HexVS, UltraPin1600, UltraPin2200, UltraVS, UVI80**.
- **Pin-map identity must become slot-qualified** (`16.ch0`), not a bare channel int. A
  channel is a hardware resource: one cell in the entire pin map. The current int model
  cannot express slot ownership or detect a cross-slot collision.
- **Groups and aliases**: an alias is a *name*, not a second claim on a resource. Modelling
  them as allocations would produce false "channel used twice" violations.
- **Valid-channel tables read from the corpus**, not hard-coded. The help set already carries
  `SCAN Channel Assignment Rules for <type>` with enumerated channel lists; reading them keeps
  the checker from drifting from the documentation it enforces.

**Then — widen beyond channel maps** (same method):
- **Timeset basics**, **AC / DC / global specs**, **pin map**, **channel map**.
- These rules are a mix of prose *and figures*, so the sweep doubles as **enrichment**: every
  figure a researcher must open to extract a rule gets described and cached (§3), which makes
  it searchable for everyone afterwards. Build the checker and enrich the catalog in one pass.

---

## 2. Retrieval scope must be *used*, not just stored

**Why.** An UltraPin2200-only statement was read as a universal claim about SCAN and shaped a
UP1600 recommendation. Tagging alone does not prevent that; filtering does.

- **Done**: `instrument` is a filter on `Service.search`, `search_docs`, `/v1/search`. A page
  naming no instrument is general and always kept.
- **Open**: nothing *tells an agent to pass it*. Until the skill does, the filter sits unused.
- **Open**: `applies_from` is extracted (182 docs) but is not yet a filter. Version scope is
  the other half of the same error class.

---

## 3. Figures are first-class content

**Why.** 5,523 images, and the state/channel-allocation detail lives in them. Search returned
only an image SHA, no tool returned image content, and reports auto-embed cited figures — so
models were placing diagrams they had never seen.

- **Done**: `get_image` returns real image content; `describe_image` caches what the agent saw;
  descriptions survive a re-ingest (`image_vision`) **and** a database deletion (sidecar beside
  the bytes in `staging/`).
- **Open**: only ~25 of 5,523 described. Lazy enrichment fills this as questions arrive; a bulk
  pass over rule-bearing figures would front-load the ones that matter.

---

## 4. Known gaps

- **`search_docs` payload**: the compact `hit_fields` + shared `documents` shape landed. Watch
  for regression — a flat self-describing hit shape cost ~0.65 KB each and overran the client
  tool-output cap at 8 queries × top_k 10.
- **Inline `[D:…]` ids still leak** into report bodies. References resolve to real documents,
  but the body should render footnote anchors. Fix belongs in the renderer, not in a prompt:
  prompt-level pressure has failed three times.
- **`vision_model` is empty** on agent-written descriptions — the tool does not record which
  model looked at the figure. Provenance should not be optional.
- **`install.sh` / `start-server.sh` have never had a clean-room run.** Syntax and guard paths
  are tested; a fresh-machine install is not.
- **No PowerShell equivalents** (`install.ps1`, `start-server.ps1`) despite Windows being a
  first-class target.

---

## 5. Method: researcher → writer, via a temp file

Large rule sweeps run as two agents. A **researcher** mines the catalog and writes structured
findings (statement, scope, verbatim source, `CHECKABLE`, `CHECK_LOGIC`) to a temp file. A
**writer** turns that file into code and deletes it. The orchestrating context never holds the
raw research, which is what makes an exhaustive sweep affordable.

Two standing rules for any rule captured this way:
- **Never flatten scope.** An instrument- or version-specific statement recorded as universal
  is a wrong answer stated confidently — the failure this whole effort exists to prevent.
- **A figure you cannot read is `UNCERTAIN`, not a guess.** Record the caption and doc id.

---

## 6. Feedback policy

AI-initiated feedback is **not** accepted. Entries come from a user, or from a user correcting
an answer — then scope returns to the original. A model already flattened a scoped statement
once; letting models write their own grounding would make such an error permanent.
`annotations.origin` carries the distinction.

## 7. build_report evidence-array friction

**Why.** Three separate headless runs lost time discovering that `build_report` refuses
inline `[D:...]` tags and requires an explicit `evidence: [[doc_id, chunk_id], ...]` array.
The requirement is real (it is the hallucinated-citation guard) but undocumented in the tool.

Fix: accept the inline `[D:...]` tags the renderer already parses as the evidence set, or
state the `evidence` requirement in the `build_report` docstring. Small, and it recurs.
