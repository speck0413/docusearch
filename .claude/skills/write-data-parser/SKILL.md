---
name: write-data-parser
description: Get a user's data (any format) into the docusearch data store so it can be queried and plotted. Try the built-in delimited/fixed-width reader first; convert odd formats to CSV; only write a real parser behind the ingest seam for a genuinely non-standard format. Use when a user wants to analyze/plot a data file docusearch can't already read.
---

# Get any data into the docusearch data store

The data store holds **numeric columns** you can query (`data_values`) and plot (`data_plot`). STDF
is just one input; a data store is `store_type: "data"`. Your job: get the user's data in with the
**least code**, in this order.

## 1. First try: no parser at all (delimited or fixed-width)

`dataset.read_table` already reads **any delimited or fixed-width table** — CSV, TSV, pipe/semicolon,
or fixed columns. Point a `store_type: data` source at the files and set the per-source `csv:` block;
`docusearch ingest` stores every numeric column. No code.

```yaml
store_type: "data"
sources:
  - name: measurements
    location: "/path/to/files"
    include: ["**/*.csv", "**/*.tsv", "**/*.dat"]
    csv:
      delimiter: "pipe"     # or ",", ";", "|", or a name (tab|pipe|semicolon); omit → by extension
      widths: [8, 10, 12]   # OR read a fixed-width file by these column widths (no delimiter)
      label: "test"         # long/tidy: a metric-name column + a value column
      value: "reading"
      group: "site"         # optional group dimension
      lo: "lo_limit"        # optional spec limits → Cpk + red plot lines
      hi: "hi_limit"
```

Verify: `docusearch ingest` → `docusearch data ls` shows the columns → `docusearch data plot <name>`.
**If this works, you're done — do not write code.**

## 2. Next: convert the odd format to CSV

If it's a spreadsheet, a JSON/NDJSON log, a database export, an instrument dump with a known
exporter, etc. — the cheapest path is a **one-off conversion to a wide or long CSV**, then ingest per
step 1. Write a tiny throwaway converter (stdlib `csv` + `json`/`openpyxl`/etc.), put the CSV under
the data source, and stop. Help the user pick **wide** (one column per metric) vs **long/tidy** (a
`metric,value[,group,lo,hi]` layout) — long is best when metrics share limits/units.

## 3. Only if truly non-standard: write a parser behind the ingest seam

Reserve this for a binary/nested format that can't be flattened to a table cheaply (like STDF).
Mirror `_write_stdf` / `_write_table` in `src/docusearch/ingest.py`:

1. **Produce a `dataset.Dataset`.** Write `extract_<fmt>(path) -> dataset.Dataset` (or parse to rows
   then build `Column`s). Reuse before reinventing: stdlib → an existing project dep → a
   well-maintained PyPI package (pin it in the lockfile + an extra) → hand-rolled, in that order.
2. **Add a routing branch.** In `run_ingest`, route the extension to a `_write_<fmt>` that calls
   `store.add_data_column(...)` for each numeric column and writes a document + a searchable summary
   chunk — copy the shape of `_write_table`. Keep everything else untouched (the seam).
3. **TDD.** Failing test first: a small fixture file → parse → assert the columns/values. Then the
   minimal code to pass. `ruff` + `mypy` clean; drop NaN/inf (`analytics._finite` handles values).
4. **No corpus-specific tokens** in `src/` (paths, IDs, answers). Windows-first (pathlib, no
   POSIX-only). One implementation per concept; no `utils.py` dumping ground.

Verify end to end: ingest → `data ls` → `data plot`. Then hand off to **write-data-report** to build
the analysis/plots the user actually wanted.

## Checklist before you finish
- [ ] Did step 1 or 2 already solve it? (Prefer no new code.)
- [ ] `docusearch data ls` shows the expected columns with the right n / limits.
- [ ] A round-trip test exists; `ruff` + `mypy` clean; the full suite still passes.
