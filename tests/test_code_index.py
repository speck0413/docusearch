"""Phase 9 / GATE 9 — language-aware source-code symbol extraction (tree-sitter).

`code_index.parse_symbols(text, language)` turns a source file into retrievable **symbols** (functions,
classes, methods, …) with a qualified name, signature, docstring, and line span — the substrate the
code store searches over and the style guide is later derived from. Multi-language from day one.
"""

from __future__ import annotations

from docusearch import code_index

PY = '''\
from __future__ import annotations


def greet(name: str) -> str:
    """Say hi to someone."""
    return f"hi {name}"


class Store:
    """A tiny store."""

    def open(self, path: str) -> "Store":
        # not a docstring
        return self
'''


def _by_qual(text: str, language: str) -> dict[str, code_index.Symbol]:
    return {s.qualname: s for s in code_index.parse_symbols(text, language, path="x")}


def test_python_functions_classes_methods() -> None:
    syms = _by_qual(PY, "python")
    assert set(syms) == {"greet", "Store", "Store.open"}

    g = syms["greet"]
    assert g.kind == "function" and g.name == "greet" and g.parent == ""
    assert g.language == "python"
    assert "def greet(name: str) -> str" in g.signature and "return" not in g.signature
    assert g.docstring == "Say hi to someone."
    assert g.start_line == 4 and g.end_line == 6

    c = syms["Store"]
    assert c.kind == "class" and c.docstring == "A tiny store."

    m = syms["Store.open"]
    assert m.kind == "method" and m.name == "open" and m.parent == "Store"
    assert "def open(self, path: str)" in m.signature
    assert m.docstring == ""  # the comment inside the body is NOT a docstring


def test_signature_is_header_only_no_body() -> None:
    # a multi-line signature keeps its lines but stops at the body
    src = 'def f(\n    a: int,\n    b: int,\n) -> int:\n    return a + b\n'
    (f,) = code_index.parse_symbols(src, "python", path="x")
    assert "return" not in f.signature
    assert "a: int" in f.signature and "b: int" in f.signature


def test_multi_language_smoke() -> None:
    js = _by_qual("function f(a) { return a }\nclass C { m() {} }\n", "javascript")
    assert js["f"].kind == "function" and js["C.m"].kind == "method"

    go = _by_qual("package main\nfunc Add(a int, b int) int { return a + b }\n", "go")
    assert go["Add"].kind == "function" and "Add(a int, b int) int" in go["Add"].signature

    rs = _by_qual("fn add(a: i32) -> i32 { a }\n", "rust")
    assert rs["add"].kind == "function"

    ja = _by_qual("class C { void m(int a) {} }\n", "java")
    assert ja["C"].kind == "class" and ja["C.m"].kind == "method"


def test_detect_language_by_extension() -> None:
    assert code_index.detect_language("a/b/foo.py") == "python"
    assert code_index.detect_language("foo.ts") == "typescript"
    assert code_index.detect_language("foo.tsx") == "typescript"
    assert code_index.detect_language("foo.js") == "javascript"
    assert code_index.detect_language("foo.go") == "go"
    assert code_index.detect_language("foo.rs") == "rust"
    assert code_index.detect_language("foo.java") == "java"
    assert code_index.detect_language("README.md") is None
    assert code_index.detect_language("noext") is None


def test_unsupported_language_raises() -> None:
    import pytest
    with pytest.raises(ValueError, match="unsupported"):
        code_index.parse_symbols("x = 1", "cobol", path="x")


def test_syntax_errors_are_tolerated() -> None:
    # tree-sitter is error-tolerant: a broken tail should not lose the good symbol before it
    src = 'def ok():\n    return 1\n\ndef broken(  :\n'
    syms = _by_qual(src, "python")
    assert "ok" in syms


def test_deeply_nested_is_fast_and_complete() -> None:
    # red-team #H2: qualname computation must be O(1) per symbol (scope carried during the walk),
    # not an O(depth) ancestor re-walk — 400 nested functions used to take seconds and drop symbols
    import time
    src = "".join(f"function f{i}() {{" for i in range(400)) + "}" * 400
    t0 = time.perf_counter()
    syms = code_index.parse_symbols(src, "javascript", path="x")
    assert time.perf_counter() - t0 < 2.0  # was multiple seconds under the O(n·depth) walk
    assert len(syms) == 400  # every nested function still found
    assert syms[0].qualname == "f0"  # outermost
    assert any(s.qualname.endswith(".f399") for s in syms)  # deepest nesting preserved in qualname
