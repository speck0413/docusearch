---
name: write-data-report
description: Analyze, plot, and report on data already in the docusearch data store (STDF or any ingested CSV/table). List columns, pull values, classify distributions and compare runs on the CPU, render plots, and assemble a cited report. Use when a user wants charts, a distribution/shift analysis, or a report over stored data.
---

# Analyze + report over the docusearch data store

The data is already stored as numeric columns (see **write-data-parser** to get it in). This skill
turns it into plots and a report. Everything numeric runs **on the CPU** — you invoke the scripts and
give a disposition; you don't eyeball raw numbers.

## Get the data

Over the **MCP server** (an agent) or the equivalent CLI/REST (a human/web UI):

| Want | MCP tool | CLI | REST |
|------|----------|-----|------|
| list columns | `list_data()` | `docusearch data ls [glob]` | `GET /v1/data/columns` |
| raw values | `data_values(column_id)` | — | `GET /v1/data/columns/{id}/values` |
| a plot | `data_plot(column_id, kind, by_group)` | `docusearch data plot <col> --kind …` | `GET /v1/data/columns/{id}/plot` |

For STDF specifically, the richer tools still apply: `stdf_plot`, `stdf_audit` (the 6-tab dashboard),
`stdf_site_compare`, `stdf_trend` — use those when the data is STDF.

## Analyze on the CPU (no AI in the numbers)

Pull values and call `docusearch.analytics` directly for a custom analysis:

- `classify_distribution(values)` → shape (`normal` / `bimodal` / `long-tail` / `outliers` /
  `discrete` / …) + skew/kurtosis/outlier-fraction.
- `compare_distributions(a, b)` → run-to-run verdict on the **two-sample KS** vs its sample-size-aware
  critical value: `shifted` (mean moved), `uncorrelated` (shape changed), plus `qq_r2`, `z_shift`, `ks`.
- `site_dispersion({group: values})` → site-to-site agreement (`site_shift`, `spread_sigma`).
- `capability(values, lo, hi)` → Cpl/Cpu/Cpk; `summary_stats(values)` → n/mean/median/std/min/max.
- `render_plot(kind, y=…|series=…, vlines=[lo,hi], backend=…)` → a self-contained plot fragment
  (matplotlib PNG or a plotly div). `include_js=False` for all-but-the-first plotly plot on a page.

Push the "which columns are interesting" judgement onto these — sort/rank by `qq_r2` (worst
correlated), `abs(z_shift)` (biggest shift), or `cpk` (worst capability), and only surface the few
that matter. Give a disposition (accept / investigate) **only if the user asks** for one.

## Build the report

Two paths:

1. **A cited prose+figures report** (the usual): assemble an answer spec and render with the report
   builder, exactly like the `docusearch` research skill — embed plot fragments as figures, cite any
   catalog claims `[D:doc#chunk]`, general knowledge `[GK]`. `docusearch report --spec s.yaml
   --format html|pdf|docx|pptx|xlsx`. The citation guard refuses hallucinated references.

2. **A bespoke themed dashboard for a data type** (like the STDF audit): mirror `stdf_analytics`'s
   builders — compute with `analytics`, lay out cards with `report.themed_page(...)` + the shared
   `_ANALYTICS_CSS` (tabs, chips, sortable/expandable tables, `.plotgrid`), and render plots with
   `render_plot`. Keep it deterministic (temperature 0 / fixed inputs → byte-identical output; give
   plotly a stable div id) and escape every user string (`html.escape`) — the red team greps for
   injection. Save under the config's `tmp_dir/reports/`, never in the repo.

## Rules
- Deterministic + reproducible (R-SRCH-5): same inputs → identical output; record seeds for anything
  sampled.
- Non-finite values are dropped by `analytics._finite`; report `n` excluding them ("3 of 96 invalid,
  excluded") rather than crashing.
- All generated output under `tmp/`. `ruff` + `mypy` clean; a test for any new builder.
- Reuse before reinventing — the analytics/plot/report engines already exist; call them, don't
  re-implement stats or charting.
