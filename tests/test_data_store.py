"""Generic columnar data store (Phase 10, migration 5): store any CSV/table's columns + values so
non-AI tools can query and plot them — not STDF-specific."""

from __future__ import annotations

from pathlib import Path

from docusearch.store import Store


def _doc(store: Store, path: str) -> int:
    return store.add_document(path=path, source="csv-src", title=path, fmt="csv")


def test_add_and_query_data_columns(tmp_path: Path) -> None:
    with Store.open(str(tmp_path / "c.db")) as store:
        doc = _doc(store, "meas.csv")
        cid = store.add_data_column(
            doc_id=doc, dataset="meas", name="vmin", kind="numeric", units="V",
            lo=0.70, hi=0.85, values=[0.71, 0.72, 0.70], groups=["1", "1", "2"])
        store.add_data_column(
            doc_id=doc, dataset="meas", name="label", kind="categorical", units="",
            lo=None, hi=None, values=[])
        cols = store.data_columns(doc_id=doc)
        assert [c["name"] for c in cols] == ["vmin", "label"]
        vmin = next(c for c in cols if c["name"] == "vmin")
        assert vmin["kind"] == "numeric" and vmin["lo"] == 0.70 and vmin["n"] == 3
        vals = store.data_values(column_id=cid)
        assert [v["value"] for v in vals] == [0.71, 0.72, 0.70]
        assert [v["grp"] for v in vals] == ["1", "1", "2"]
        assert store.count_data_values() == 3


def test_delete_document_clears_data_tables(tmp_path: Path) -> None:
    with Store.open(str(tmp_path / "c.db")) as store:
        doc = _doc(store, "x.csv")
        store.add_data_column(doc_id=doc, dataset="x", name="a", kind="numeric", units="",
                              lo=None, hi=None, values=[1.0, 2.0, 3.0])
        assert store.count_data_values() == 3
        store.delete_document(doc)
        assert store.count_data_values() == 0 and store.data_columns() == []


def test_store_opens_at_current_schema(tmp_path: Path) -> None:
    from docusearch.store import SCHEMA_VERSION
    with Store.open(str(tmp_path / "c.db")) as store:
        assert store._read_version() == SCHEMA_VERSION  # noqa: SLF001
        names = {r[0] for r in store._conn.execute(  # noqa: SLF001
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        assert {"data_columns", "data_values", "feedback"} <= names
