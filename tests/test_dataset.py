"""Format-agnostic columnar Dataset + CSV reader (Phase 10) — the general data engine's shape.
STDF is just one thing a CSV can hold; arbitrary numeric CSVs work the same way."""

from __future__ import annotations

from pathlib import Path

from docusearch import analytics, dataset


def _write(p: Path, text: str) -> Path:
    p.write_text(text, encoding="utf-8")
    return p


def test_wide_csv_infers_column_kinds(tmp_path: Path) -> None:
    csv = _write(tmp_path / "w.csv",
                 "vmin,iddq,site,label\n"
                 "0.71,1e-6,1,partA\n0.72,2e-6,1,partB\n0.70,1.5e-6,2,partC\n0.69,3e-6,2,partD\n")
    ds = dataset.read_table(csv)
    kinds = {c.name: c.kind for c in ds.columns}
    assert kinds["vmin"] == "numeric" and kinds["iddq"] == "numeric"
    assert kinds["label"] in ("categorical", "text")  # non-numeric
    vmin = ds.column("vmin")
    assert vmin is not None and vmin.values == [0.71, 0.72, 0.70, 0.69]
    assert [c.name for c in ds.numeric()] == ["vmin", "iddq", "site"]  # site is numeric here


def test_wide_csv_with_group_column_tags_observations(tmp_path: Path) -> None:
    csv = _write(tmp_path / "g.csv",
                 "vmin,site\n0.71,1\n0.72,1\n0.90,2\n0.91,2\n")
    ds = dataset.read_table(csv, group_column="site")
    assert ds.group_role == "site"
    vmin = ds.column("vmin")
    assert vmin is not None
    groups = vmin.group_values()
    assert groups["1"] == [0.71, 0.72] and groups["2"] == [0.90, 0.91]
    assert ds.column("site").kind == "categorical"  # the group column isn't a metric


def test_long_csv_pivots_to_metrics_with_limits_and_groups(tmp_path: Path) -> None:
    csv = _write(tmp_path / "l.csv",
                 "test,value,lo,hi,units,site\n"
                 "VMIN,0.71,0.70,0.85,V,1\nVMIN,0.72,0.70,0.85,V,1\n"
                 "VMIN,0.90,0.70,0.85,V,2\nIDDQ,1e-6,0,2e-6,A,1\nIDDQ,2e-6,0,2e-6,A,2\n")
    ds = dataset.read_table(csv, label_column="test", value_column="value",
                          lo_column="lo", hi_column="hi", units_column="units", group_column="site")
    vmin = ds.column("VMIN")
    assert vmin is not None
    assert vmin.values == [0.71, 0.72, 0.90] and vmin.lo == 0.70 and vmin.hi == 0.85
    assert vmin.units == "V" and vmin.group_values()["2"] == [0.90]
    assert ds.column("IDDQ").hi == 2e-6


def test_missing_and_nonfinite_values_dropped(tmp_path: Path) -> None:
    csv = _write(tmp_path / "m.csv", "x\n1.0\n\nNA\nnan\ninf\n2.0\n")
    ds = dataset.read_table(csv)
    assert ds.column("x").values == [1.0, 2.0]  # blank/NA/nan/inf all dropped


def test_dataset_feeds_analytics_engine(tmp_path: Path) -> None:
    # a plain numeric CSV column flows straight into the shared distribution engine — no STDF needed
    rows = "\n".join(str(round(0.001 * i, 4)) for i in range(-300, 300))
    csv = _write(tmp_path / "d.csv", "measurement\n" + rows + "\n")
    ds = dataset.read_table(csv)
    col = ds.column("measurement")
    assert col is not None and len(col.values) == 600
    shape = analytics.classify_distribution(col.values)["shape"]
    assert shape in analytics.DISTRIBUTION_SHAPES  # classified like any numeric series
    cmp = analytics.compare_distributions(col.values, [v + 0.5 for v in col.values])
    assert cmp["shifted"] is True  # run-to-run compare works on CSV data too


def test_empty_csv_is_safe(tmp_path: Path) -> None:
    assert dataset.read_table(_write(tmp_path / "e.csv", "")).columns == []
    header_only = dataset.read_table(_write(tmp_path / "h.csv", "a,b,c\n"))
    assert [c.name for c in header_only.columns] == ["a", "b", "c"]
    assert header_only.numeric() == []  # no rows → no numeric values


def test_tsv_delimiter_by_extension(tmp_path: Path) -> None:
    tsv = _write(tmp_path / "w.tsv", "a\tb\n1\t2\n3\t4\n")
    ds = dataset.read_table(tsv)
    assert ds.column("a").values == [1.0, 3.0] and ds.column("b").values == [2.0, 4.0]


def test_explicit_delimiter_and_alias(tmp_path: Path) -> None:
    psv = _write(tmp_path / "d.txt", "x|y\n1|2\n3|4\n")
    ds = dataset.read_table(psv, delimiter="pipe")  # a .txt needs an explicit delimiter
    assert ds.column("x").values == [1.0, 3.0]
    semi = _write(tmp_path / "s.txt", "x;y\n5;6\n")
    assert dataset.read_table(semi, delimiter=";").column("y").values == [6.0]


def test_fixed_width_columns(tmp_path: Path) -> None:
    # columns at fixed character positions (widths 6, 6, 4); no delimiter
    fw = _write(tmp_path / "f.txt", "vmin  iddq  st  \n0.71  1e-6  1   \n0.72  2e-6  2   \n")
    ds = dataset.read_table(fw, fixed_widths=[6, 6, 4])
    assert [c.name for c in ds.columns] == ["vmin", "iddq", "st"]
    assert ds.column("vmin").values == [0.71, 0.72] and ds.column("iddq").values == [1e-6, 2e-6]


def test_phase10_redteam_ragged_and_duplicate_columns(tmp_path: Path) -> None:
    from docusearch import dataset

    # M3: a ragged fixed-width line yields exactly len(widths) fields; a field the line doesn't FULLY
    # contain is missing ("") — never a truncated fragment mistaken for a real value
    assert dataset._split_fixed("abcde", [5, 5]) == ["abcde", ""]         # noqa: SLF001 (2nd absent)
    assert dataset._split_fixed("abcdefg", [5, 5]) == ["abcde", ""]       # noqa: SLF001 ("fg" truncated → "")
    assert dataset._split_fixed("abcdefghij", [3, 3]) == ["abc", "def"]   # noqa: SLF001 (tail ignored)

    # M4: duplicate column names are disambiguated so BOTH keep their own data (was: first silently lost)
    f = tmp_path / "dup.csv"
    f.write_text("x,x\n1,2\n3,4\n", encoding="utf-8")
    ds = dataset.read_table(f)
    by = {c.name: list(c.values) for c in ds.columns}
    assert set(by) == {"x", "x_2"} and by["x"] == [1.0, 3.0] and by["x_2"] == [2.0, 4.0]
