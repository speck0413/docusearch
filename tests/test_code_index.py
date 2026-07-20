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


def _by_qual_path(text: str, language: str, path: str) -> dict[str, code_index.Symbol]:
    return {s.qualname: s for s in code_index.parse_symbols(text, language, path=path)}


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
    assert code_index.detect_language("foo.c") == "c"
    assert code_index.detect_language("foo.h") == "c"
    assert code_index.detect_language("foo.cpp") == "cpp"
    assert code_index.detect_language("foo.hpp") == "cpp"
    assert code_index.detect_language("foo.cs") == "csharp"
    assert code_index.detect_language("foo.bas") == "vba"
    assert code_index.detect_language("Sheet1.cls") == "vba"
    assert code_index.detect_language("README.md") is None
    assert code_index.detect_language("noext") is None


def test_c_symbols() -> None:
    src = ("int add(int a, int b) { return a + b; }\n"
           "static char *dup_str(const char *s) { return 0; }\n"
           "struct Point { int x; int y; };\n"
           "enum Color { RED, GREEN };\n")
    syms = _by_qual(src, "c")
    assert syms["add"].kind == "function" and "int add(int a, int b)" in syms["add"].signature
    assert syms["dup_str"].kind == "function"        # pointer return: name still resolved
    assert syms["Point"].kind == "struct" and syms["Color"].kind == "enum"


def test_cpp_symbols_methods_and_namespace() -> None:
    src = ("class Foo {\npublic:\n  int bar(int x) { return x; }\n  void baz() const {}\n};\n"
           "void Foo::qux() {}\n"
           "namespace ns { int helper() { return 0; } }\n")
    syms = _by_qual(src, "cpp")
    assert syms["Foo"].kind == "class"
    assert syms["Foo.bar"].kind == "method" and syms["Foo.bar"].parent == "Foo"
    assert syms["Foo.baz"].kind == "method"
    assert syms["ns"].kind == "namespace"
    assert syms["ns.helper"].kind == "function" and syms["ns.helper"].parent == "ns"


def test_csharp_symbols() -> None:
    src = ("namespace N {\n  public class C {\n    public void M(int a) {}\n"
           "    public int P { get; set; }\n  }\n  interface I { void F(); }\n}\n")
    syms = _by_qual(src, "csharp")
    assert syms["N"].kind == "namespace"
    assert syms["N.C"].kind == "class"
    assert syms["N.C.M"].kind == "method" and syms["N.C.M"].parent == "N.C"
    assert syms["N.C.P"].kind == "property"
    assert syms["N.I"].kind == "interface"


def test_vba_symbols() -> None:
    src = ("' Greet a person.\n"
           "Public Function Greet(name As String) As String\n"
           "    Greet = \"hi \" & name\n"
           "End Function\n\n"
           "Private Sub DoWork()\n"
           "    Debug.Print 1\n"
           "End Sub\n\n"
           "Public Property Get Count() As Long\n"
           "    Count = 3\n"
           "End Property\n")
    syms = _by_qual(src, "vba")
    assert syms["Greet"].kind == "function"
    assert "Function Greet(name As String) As String" in syms["Greet"].signature
    assert syms["Greet"].docstring == "Greet a person."
    assert syms["Greet"].start_line == 2 and syms["Greet"].end_line == 4
    assert syms["DoWork"].kind == "function"
    assert syms["Count"].kind == "property"


def test_vba_class_module_members_are_methods() -> None:
    # a .cls file is a class module: a synthetic class from the file stem, procedures are its methods
    src = "Public Sub Run()\nEnd Sub\n"
    syms = _by_qual_path(src, "vba", "Widget.cls")
    assert syms["Widget"].kind == "class"
    assert syms["Widget.Run"].kind == "method" and syms["Widget.Run"].parent == "Widget"


def test_vba_unterminated_blocks_are_fast() -> None:
    # red-team #H1: a file full of declarations missing their `End` used to be O(n^2) (a plain typo,
    # 50k lines → 75s). The single-pass parser closes them at the next decl / EOF in linear time.
    import time
    src = "\n".join(f"Public Sub P{i}()" for i in range(20000))  # 20k Subs, zero End Sub
    t0 = time.perf_counter()
    syms = code_index.parse_symbols(src, "vba", path="m.bas")
    assert time.perf_counter() - t0 < 1.0  # was tens of seconds
    assert len(syms) == 20000  # every declaration still extracted


def test_c_deep_declarator_still_resolves() -> None:
    # red-team #L1: the declarator-descent bound must be well past any realistic nesting depth
    src = "int " + "(" * 60 + "deep" + ")" * 60 + "(void) { return 0; }"
    assert any(s.name == "deep" for s in code_index.parse_symbols(src, "c", path="x.c"))


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
