"""A format-agnostic **columnar dataset** (Phase 10) — the shared shape behind the data engine.

`docusearch`'s data store is not STDF-specific: STDF is just one thing a CSV (or any table) can hold.
This module defines a :class:`Dataset` of named numeric :class:`Column`s plus optional *semantic
roles* (spec limits, a group column like site, key columns for row identity), and a :func:`read_csv`
that turns any CSV into one. The **analytics** engine (`classify_distribution` /
`compare_distributions` / `render_plot` / `site_dispersion`) already operates on plain numeric
columns, so a Dataset feeds it directly — the same audit/Explore dashboard serves STDF and arbitrary
CSV alike. STDF's parser produces the same shape (a later increment), so there is one audit engine.

Two CSV layouts, chosen by the config role-map:
- **wide** (default): each numeric column is its own metric; rows are observations.
- **long / tidy**: a ``label`` column names the metric and a ``value`` column holds the reading, with
  optional ``group`` (e.g. site) and ``lo``/``hi`` limit columns — how test-like data lands in a CSV.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Column:
    """One metric: its finite numeric ``values`` (aligned with ``groups`` when a group role is set),
    the inferred ``kind`` (numeric | categorical | text), and optional spec ``lo``/``hi``/``units``."""

    name: str
    values: list[float] = field(default_factory=list)
    kind: str = "numeric"
    lo: float | None = None
    hi: float | None = None
    units: str = ""
    groups: list[str] = field(default_factory=list)  # parallel to values when a group role is set

    def group_values(self) -> dict[str, list[float]]:
        """``{group -> its values}`` (empty if no group role) — the input to a group/site comparison."""
        out: dict[str, list[float]] = {}
        for v, g in zip(self.values, self.groups, strict=False):
            out.setdefault(g, []).append(v)
        return out


@dataclass
class Dataset:
    """A named table of :class:`Column`s + the semantic roles that let a test-like table get the
    richer audit (limits→Cpk, group→site compare, keys→row identity). Arbitrary data just carries
    numeric columns and gets per-column distributions."""

    name: str
    columns: list[Column] = field(default_factory=list)
    group_role: str = ""          # the column used as the group/site dimension, if any
    key_columns: list[str] = field(default_factory=list)  # row-identity columns, if any

    def numeric(self) -> list[Column]:
        return [c for c in self.columns if c.kind == "numeric" and c.values]

    def column(self, name: str) -> Column | None:
        return next((c for c in self.columns if c.name == name), None)


_MISSING = {"", "na", "n/a", "nan", "null", "none", "-",
            "inf", "+inf", "-inf", "infinity", "-infinity"}  # non-finite tokens are "missing" too


def _as_float(cell: str) -> float | None:
    s = cell.strip()
    if s.lower() in _MISSING:
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def _infer_kind(cells: Sequence[str]) -> str:
    """numeric if most non-missing cells parse as finite floats; else categorical (few distinct) or
    text. A column that is mostly numbers is numeric even with a few stray non-numbers."""
    nonmissing = [c for c in cells if c.strip().lower() not in _MISSING]
    if not nonmissing:
        return "text"
    numeric = sum(1 for c in nonmissing if _as_float(c) is not None)
    if numeric >= 0.9 * len(nonmissing):
        return "numeric"
    distinct = len({c.strip() for c in nonmissing})
    return "categorical" if distinct <= max(20, len(nonmissing) // 20) else "text"


# standard delimiter by file extension; anything else needs an explicit delimiter (or fixed widths)
_DELIM_BY_EXT = {".csv": ",", ".tsv": "\t", ".tab": "\t", ".psv": "|", ".pipe": "|"}
# friendly spellings a user can put in config for hard-to-type delimiters
_DELIM_ALIASES = {"tab": "\t", r"\t": "\t", "space": " ", r"\s": " ", "pipe": "|",
                  "semicolon": ";", "comma": ","}


def _resolve_delimiter(path: Path | str, delimiter: str | None) -> str:
    if delimiter:
        return _DELIM_ALIASES.get(delimiter.strip().lower(), delimiter)
    return _DELIM_BY_EXT.get(Path(path).suffix.lower(), ",")


def _split_fixed(line: str, widths: Sequence[int]) -> list[str]:
    """Split a fixed-width line into **exactly** ``len(widths)`` fields. A short (ragged) line yields
    empty strings for the fields it doesn't reach — which `_infer_kind` treats as missing — rather than
    a ragged field count that would misalign columns or silently drop one (red-team #M3). Content past
    the declared widths is ignored: the widths define the columns."""
    out, i = [], 0
    for w in widths:
        # only take a field the line FULLY contains — a straddling/short field is missing (""), never a
        # truncated fragment mistaken for a real value (red-team #M3)
        out.append(line[i:i + w].strip() if i + w <= len(line) else "")
        i += w
    return out


def _read_rows(
    path: Path | str, *, delimiter: str | None = None, fixed_widths: Sequence[int] | None = None,
) -> tuple[list[str], list[list[str]]]:
    """Parse a delimited **or** fixed-width table into (header, body). Delimiter defaults by extension
    (``.csv``→``,``, ``.tsv``→tab, ``.psv``→``|``) unless given; ``fixed_widths`` parses columns at
    fixed character positions instead (no delimiter)."""
    if fixed_widths:
        with Path(path).open(encoding="utf-8-sig") as fh:
            lines = [ln.rstrip("\r\n") for ln in fh if ln.strip()]
        rows = [_split_fixed(ln, fixed_widths) for ln in lines]
    else:
        with Path(path).open(newline="", encoding="utf-8-sig") as fh:
            reader = csv.reader(fh, delimiter=_resolve_delimiter(path, delimiter))
            rows = [row for row in reader if any(cell.strip() for cell in row)]
    if not rows:
        return [], []
    header = [h.strip() for h in rows[0]]
    body = [r + [""] * (len(header) - len(r)) for r in rows[1:]]  # pad short rows
    return header, body


def read_table(
    path: Path | str,
    *,
    delimiter: str | None = None,
    fixed_widths: Sequence[int] | None = None,
    name: str = "",
    label_column: str = "",
    value_column: str = "",
    group_column: str = "",
    lo_column: str = "",
    hi_column: str = "",
    units_column: str = "",
    key_columns: Iterable[str] = (),
) -> Dataset:
    """Read **any delimited or fixed-width table** into a :class:`Dataset`. ``delimiter`` defaults by
    extension (csv/tsv/psv) or takes an explicit char / alias (``tab``, ``pipe``, …); ``fixed_widths``
    parses fixed-column-width files instead. **Long/tidy** when ``label_column`` + ``value_column`` are
    given (each distinct label → a Column; optional ``group``/``lo``/``hi``/``units`` columns fill the
    roles) — the shape test-like data takes in a table. **Wide** otherwise: every numeric column
    becomes a metric, and ``group_column`` (if given) tags each numeric observation by that value."""
    header, body = _read_rows(path, delimiter=delimiter, fixed_widths=fixed_widths)
    ds_name = name or Path(path).stem
    if not header:
        return Dataset(ds_name)
    # disambiguate duplicate column names (``x``, ``x`` → ``x``, ``x_2``) so each keeps its own
    # position — a name-keyed index would otherwise silently drop all but the last (red-team #M4)
    seen: dict[str, int] = {}
    uniq: list[str] = []
    for h in header:
        seen[h] = seen.get(h, 0) + 1
        uniq.append(h if seen[h] == 1 else f"{h}_{seen[h]}")
    header = uniq
    idx = {h: i for i, h in enumerate(header)}

    def col(row: list[str], colname: str) -> str:
        i = idx.get(colname)
        return row[i] if i is not None and i < len(row) else ""

    if label_column and value_column:
        return _read_long(ds_name, body, col, label_column, value_column, group_column,
                          lo_column, hi_column, units_column, list(key_columns))
    return _read_wide(ds_name, header, body, idx, group_column, list(key_columns))


def _read_long(
    name: str, body: list[list[str]], col: Callable[[list[str], str], str],
    label_column: str, value_column: str,
    group_column: str, lo_column: str, hi_column: str, units_column: str, keys: list[str],
) -> Dataset:
    cols: dict[str, Column] = {}
    order: list[str] = []
    for row in body:
        label = col(row, label_column).strip()
        val = _as_float(col(row, value_column))
        if not label or val is None:
            continue
        if label not in cols:
            cols[label] = Column(name=label)
            order.append(label)
            if lo_column:
                cols[label].lo = _as_float(col(row, lo_column))
            if hi_column:
                cols[label].hi = _as_float(col(row, hi_column))
            if units_column:
                cols[label].units = col(row, units_column).strip()
        c = cols[label]
        c.values.append(val)
        if group_column:
            c.groups.append(col(row, group_column).strip())
    return Dataset(name, [cols[k] for k in order], group_role=group_column, key_columns=keys)


def _read_wide(
    name: str, header: list[str], body: list[list[str]], idx: dict[str, int],
    group_column: str, keys: list[str],
) -> Dataset:
    group_vals = (
        [row[idx[group_column]].strip() if idx[group_column] < len(row) else "" for row in body]
        if group_column and group_column in idx else []
    )
    columns: list[Column] = []
    for h in header:
        cells = [row[idx[h]] if idx[h] < len(row) else "" for row in body]
        kind = _infer_kind(cells)
        if h == group_column:
            kind = "categorical"
        c = Column(name=h, kind=kind)
        if kind == "numeric":
            for cell, g in zip(cells, group_vals or [""] * len(cells), strict=False):
                v = _as_float(cell)
                if v is not None:
                    c.values.append(v)
                    if group_column:
                        c.groups.append(g)
        columns.append(c)
    return Dataset(name, columns, group_role=group_column, key_columns=keys)
