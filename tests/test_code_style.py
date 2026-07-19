"""Phase 9 / GATE 9 — derive a house **style guide** from the indexed symbols: naming conventions
per symbol group, docstring coverage, typing discipline, and structure. No AI; pure statistics over
the symbols the parser already extracted."""

from __future__ import annotations

from docusearch import code_style
from docusearch.code_index import Symbol


def _fn(name: str, sig: str, doc: str = "d", lines: int = 4) -> Symbol:
    return Symbol("python", "function", name, name, sig, doc, 1, lines, "", "m.py")


def _cls(name: str, doc: str = "d") -> Symbol:
    return Symbol("python", "class", name, name, f"class {name}", doc, 1, 10, "", "m.py")


def _meth(cls: str, name: str, sig: str, doc: str = "d") -> Symbol:
    return Symbol("python", "method", name, f"{cls}.{name}", sig, doc, 2, 5, cls, "m.py")


def test_name_convention_classifier() -> None:
    assert code_style.name_convention("open_store") == "snake_case"
    assert code_style.name_convention("connect") == "snake_case"       # single lower word
    assert code_style.name_convention("readTable") == "camelCase"
    assert code_style.name_convention("DataStore") == "PascalCase"
    assert code_style.name_convention("Client") == "PascalCase"        # single capitalised word
    assert code_style.name_convention("MAX_SIZE") == "SCREAMING_SNAKE"


def test_derive_python_style() -> None:
    syms = [
        _fn("open_store", "def open_store(path: str) -> bool"),
        _fn("read_table", "def read_table(p: str) -> list"),
        _fn("write_row", "def write_row(r: dict) -> None"),
        _fn("badName", "def badName(x)", doc=""),           # one camelCase outlier, no docstring/annots
        _cls("DataStore"),
        _cls("CodeIndex"),
        _meth("DataStore", "open", "def open(self) -> DataStore"),
        _meth("DataStore", "close", "def close(self) -> None"),
    ]
    g = code_style.derive_style(syms, "python")
    assert g.language == "python"
    assert g.counts == {"function": 4, "class": 2, "method": 2}

    # callables = functions + methods; 5 of 6 are snake_case → dominant snake_case
    assert g.naming["callable"].convention == "snake_case"
    assert round(g.naming["callable"].rate, 2) == round(5 / 6, 2)
    assert g.naming["type"].convention == "PascalCase" and g.naming["type"].rate == 1.0

    # docstrings: 5 of 6 callables have one; both types do
    assert round(g.docstring_coverage["callable"], 2) == round(5 / 6, 2)
    assert g.docstring_coverage["type"] == 1.0

    # return annotations (python `->`): 5 of 6 callables
    assert g.return_annotation_rate is not None and round(g.return_annotation_rate, 2) == round(5 / 6, 2)
    assert g.avg_methods_per_type == 1.0  # 2 methods / 2 types


def test_derive_all_groups_by_language() -> None:
    syms = [
        _fn("a", "def a() -> int"),
        Symbol("javascript", "function", "b", "b", "function b()", "", 1, 2, "", "u.js"),
        Symbol("javascript", "function", "c", "c", "function c()", "", 1, 2, "", "u.js"),
    ]
    guides = code_style.derive_all(syms)
    assert [g.language for g in guides] == ["javascript", "python"]  # most symbols first
    assert code_style.derive_style(syms, "javascript").return_annotation_rate is None  # not py/ts


def test_style_guide_html_renders() -> None:
    g = code_style.derive_style([_fn("open_store", "def open_store() -> int"), _cls("DataStore")],
                                "python")
    html = code_style.style_guide_html([g])
    assert html.startswith("<!doctype html>")
    assert "Style guide" in html and "snake_case" in html and "PascalCase" in html
    assert "python" in html


def test_empty_symbols_is_clean() -> None:
    g = code_style.derive_style([], "python")
    assert g.counts == {} and g.naming == {} and g.avg_lines == 0.0
    assert "no symbols" in code_style.style_guide_html([g]).lower() or code_style.style_guide_html([g])
