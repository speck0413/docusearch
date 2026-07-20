"""Language-aware source-code symbol extraction (Phase 9 / GATE 9) — the code engine's parser.

A source file is parsed with **tree-sitter** (error-tolerant, one grammar per language, each shipping
its compiled parser in its own wheel → offline) into retrievable :class:`Symbol` units — functions,
classes, methods, structs, … — each with a qualified name (``Class.method``), the header
**signature**, a **docstring** (Python: the leading string; others: the leading comment block), and a
1-based line span. This is the substrate the ``code`` store searches over and the style guide is
later derived from.

Adding a language is one pinned grammar in the ``code`` extra + one :data:`LANG_SPECS` row — the walk,
qualname nesting, signature slicing, and comment-docstring logic are all language-agnostic.
"""

from __future__ import annotations

import importlib
import re
from dataclasses import dataclass, field
from functools import cache
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tree_sitter import Node, Parser


@dataclass(frozen=True)
class Symbol:
    """One extracted code unit — a function / class / method / struct / interface / …."""

    language: str
    kind: str        # function | method | class | struct | interface | trait | enum | type
    name: str        # the bare identifier, e.g. "open"
    qualname: str    # dotted path through enclosing definitions, e.g. "Store.open"
    signature: str   # the header line(s), body stripped
    docstring: str   # "" when none
    start_line: int  # 1-based, inclusive
    end_line: int
    parent: str = ""   # enclosing symbol's qualname, or ""
    path: str = ""     # source file (set by the extractor)


@dataclass(frozen=True)
class _Spec:
    module: str                 # grammar package, e.g. "tree_sitter_python"
    lang_attr: str              # attribute returning the PyCapsule, e.g. "language"
    kinds: dict[str, str] = field(default_factory=dict)  # tree-sitter node type -> base kind
    # node types that are only a real definition when they carry a body (a C/C++ struct/class used as
    # a type or forward-declared has no body — skip it so it isn't double-counted as a symbol)
    require_body: frozenset[str] = frozenset()


# One row per supported language. `kinds` maps the grammar's definition node types to a base kind;
# a base "function" becomes "method" when it nests directly inside a class-like symbol (below).
LANG_SPECS: dict[str, _Spec] = {
    "python": _Spec("tree_sitter_python", "language",
                    {"function_definition": "function", "class_definition": "class"}),
    "javascript": _Spec("tree_sitter_javascript", "language",
                         {"function_declaration": "function",
                          "generator_function_declaration": "function",
                          "class_declaration": "class", "method_definition": "method"}),
    "typescript": _Spec("tree_sitter_typescript", "language_typescript",
                         {"function_declaration": "function", "class_declaration": "class",
                          "abstract_class_declaration": "class", "method_definition": "method",
                          "interface_declaration": "interface", "enum_declaration": "enum"}),
    "go": _Spec("tree_sitter_go", "language",
                {"function_declaration": "function", "method_declaration": "method",
                 "type_spec": "type"}),
    "rust": _Spec("tree_sitter_rust", "language",
                  {"function_item": "function", "struct_item": "struct", "enum_item": "enum",
                   "trait_item": "trait"}),
    "java": _Spec("tree_sitter_java", "language",
                  {"class_declaration": "class", "interface_declaration": "interface",
                   "enum_declaration": "enum", "method_declaration": "method",
                   "constructor_declaration": "method"}),
    "c": _Spec("tree_sitter_c", "language",
               {"function_definition": "function", "struct_specifier": "struct",
                "union_specifier": "struct", "enum_specifier": "enum", "type_definition": "type"},
               require_body=frozenset({"struct_specifier", "union_specifier", "enum_specifier"})),
    "cpp": _Spec("tree_sitter_cpp", "language",
                 {"function_definition": "function", "class_specifier": "class",
                  "struct_specifier": "struct", "union_specifier": "struct",
                  "enum_specifier": "enum", "namespace_definition": "namespace"},
                 require_body=frozenset({"class_specifier", "struct_specifier",
                                         "union_specifier", "enum_specifier"})),
    "csharp": _Spec("tree_sitter_c_sharp", "language",
                    {"class_declaration": "class", "interface_declaration": "interface",
                     "struct_declaration": "struct", "enum_declaration": "enum",
                     "record_declaration": "class", "method_declaration": "method",
                     "constructor_declaration": "method", "destructor_declaration": "method",
                     "property_declaration": "property",
                     "namespace_declaration": "namespace",
                     "file_scoped_namespace_declaration": "namespace"}),
    # "vba" has no tree-sitter grammar — it is parsed by _parse_vba (a line-based extractor) instead.
}

# File extension → language.
_EXT: dict[str, str] = {
    ".py": "python", ".pyi": "python",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".go": "go", ".rs": "rust", ".java": "java",
    ".c": "c", ".h": "c",
    ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".c++": "cpp",
    ".hpp": "cpp", ".hh": "cpp", ".hxx": "cpp", ".h++": "cpp",
    ".cs": "csharp",
    ".vba": "vba", ".bas": "vba", ".cls": "vba", ".frm": "vba",
}

# Kinds whose scope makes a nested "function" a "method", and defines a qualname boundary.
_CLASS_LIKE = frozenset({"class", "struct", "interface", "trait", "enum", "impl"})
# Node types that begin a definition's body — everything before the first one is the signature.
_BODY_TYPES = frozenset({
    "block", "statement_block", "class_body", "interface_body", "enum_body", "enum_class_body",
    "declaration_list", "field_declaration_list", "enum_variant_list",
    "compound_statement", "enumerator_list", "enum_member_declaration_list", "accessor_list",
})
_COMMENT_TYPES = frozenset({"comment", "line_comment", "block_comment"})
# Declarator leaves that name a C/C++ definition (the name is buried in the declarator, not a field).
_C_NAME_NODES = frozenset({
    "identifier", "field_identifier", "type_identifier", "qualified_identifier",
    "destructor_name", "operator_name",
})


def detect_language(path: str) -> str | None:
    """The language for a file path by extension, or ``None`` if unsupported."""
    return _EXT.get(PurePosixPath(path).suffix.lower())


def symbol_from_row(row: object) -> Symbol:
    """Rebuild a :class:`Symbol` from a ``code_symbols`` store row (for the style-guide deriver)."""
    r = row  # a sqlite3.Row (mapping access)
    return Symbol(
        language=str(r["language"]), kind=str(r["kind"]), name=str(r["name"]),  # type: ignore[index]
        qualname=str(r["qualname"]), signature=str(r["signature"] or ""),        # type: ignore[index]
        docstring=str(r["docstring"] or ""), start_line=int(r["start_line"] or 0),  # type: ignore[index]
        end_line=int(r["end_line"] or 0), parent=str(r["parent"] or ""),         # type: ignore[index]
        path=str(r["path"] or ""),                                               # type: ignore[index]
    )


@cache
def _parser(language: str) -> Parser:
    from tree_sitter import Language, Parser

    spec = LANG_SPECS[language]
    module = importlib.import_module(spec.module)
    return Parser(Language(getattr(module, spec.lang_attr)()))


def _name(node: Node) -> str | None:
    field_node = node.child_by_field_name("name")
    if field_node is not None and field_node.text:
        return field_node.text.decode("utf-8", "replace")
    # C/C++: no "name" field — descend the `declarator` chain to the identifier that names it.
    cur = node.child_by_field_name("declarator")
    for _ in range(512):  # bounded (guards a pathological tree) but well past any real nesting depth
        if cur is None:
            return None
        if cur.type in _C_NAME_NODES:
            return cur.text.decode("utf-8", "replace").strip() if cur.text else None
        nxt = cur.child_by_field_name("declarator")
        if nxt is None:  # pointer/array/parenthesized wrappers: step into the next declarator child
            nxt = next((c for c in cur.children if c.type.endswith("declarator")
                        or c.type in _C_NAME_NODES), None)
        cur = nxt
    return None


def _signature(node: Node, src: bytes) -> str:
    body_start = node.end_byte
    for child in node.children:
        if child.type in _BODY_TYPES:
            body_start = child.start_byte
            break
    text = src[node.start_byte:body_start].decode("utf-8", "replace").strip()
    text = text.rstrip("{").rstrip()
    if text.endswith(":"):  # Python's def/class colon
        text = text[:-1].rstrip()
    return text


def _python_docstring(node: Node, src: bytes) -> str:
    body = next((c for c in node.children if c.type == "block"), None)
    if body is None or not body.named_children:
        return ""
    first = body.named_children[0]
    if first.type != "expression_statement" or not first.named_children:
        return ""
    string_node = first.named_children[0]
    if string_node.type != "string":
        return ""
    content = next((c for c in string_node.children if c.type == "string_content"), None)
    raw = (content.text if content is not None else string_node.text) or b""
    return " ".join(raw.decode("utf-8", "replace").split())


def _comment_docstring(node: Node, src: bytes) -> str:
    """Contiguous comment lines immediately above a definition (JS/TS/Go/Rust/Java doc comments)."""
    lines: list[str] = []
    sib = node.prev_sibling
    expected = node.start_point[0]
    while sib is not None and sib.type in _COMMENT_TYPES and sib.end_point[0] == expected - 1:
        raw = (sib.text or b"").decode("utf-8", "replace")
        lines.append(raw)
        expected = sib.start_point[0]
        sib = sib.prev_sibling
    lines.reverse()
    cleaned = []
    for raw in lines:
        for ln in raw.splitlines():
            ln = ln.strip().lstrip("/").lstrip("*").lstrip("#").strip()
            ln = ln.removeprefix("<").strip() if ln.startswith("<") else ln
            cleaned.append(ln)
    return " ".join(" ".join(cleaned).split())


def parse_symbols(text: str, language: str, *, path: str = "") -> list[Symbol]:
    """Every top-level and nested definition in ``text``, in source order.

    Raises ``ValueError`` for an unsupported ``language``. tree-sitter is error-tolerant, so a syntax
    error in one region does not lose the well-formed symbols around it."""
    if language == "vba":  # no tree-sitter grammar — a line-based parser handles VBA
        return _parse_vba(text, path)
    if language not in LANG_SPECS:
        raise ValueError(
            f"unsupported language: {language!r} (have: {', '.join([*LANG_SPECS, 'vba'])})")
    spec = LANG_SPECS[language]
    src = text.encode("utf-8")
    root = _parser(language).parse(src).root_node

    out: list[Symbol] = []
    # DFS carrying the enclosing named-definition scope as (parent_qualname, class_like) — computed
    # once as we descend, so a symbol's qualname/parent is O(1), not an O(depth) re-walk per symbol
    # (which was O(n·depth) tree-sitter FFI calls overall on deeply nested files — red-team #H2).
    stack: list[tuple[Node, str, bool]] = [(root, "", False)]
    while stack:
        node, parent, parent_class_like = stack.pop()
        child_parent, child_class_like = parent, parent_class_like
        has_body = node.type not in spec.require_body or any(
            c.type in _BODY_TYPES for c in node.children)
        if node.type in spec.kinds and has_body:
            name = _name(node)
            if name:
                qualname = f"{parent}.{name}" if parent else name
                kind = spec.kinds[node.type]
                if kind == "function" and parent_class_like:
                    kind = "method"
                doc = _python_docstring(node, src) if language == "python" \
                    else _comment_docstring(node, src)
                out.append(Symbol(
                    language=language, kind=kind, name=name, qualname=qualname,
                    signature=_signature(node, src), docstring=doc,
                    start_line=node.start_point[0] + 1, end_line=node.end_point[0] + 1,
                    parent=parent, path=path,
                ))
                child_parent, child_class_like = qualname, spec.kinds[node.type] in _CLASS_LIKE
        for child in reversed(node.children):  # DFS, preserving source order on emit
            stack.append((child, child_parent, child_class_like))

    out.sort(key=lambda s: (s.start_line, s.qualname))
    return out


# ---- VBA (no tree-sitter grammar exists — a line-based extractor; VBA procedures never nest) -------

_VBA_DECL = re.compile(
    r"^\s*(?:(?:Public|Private|Friend|Global)\s+)?(?:Static\s+)?"
    r"(?P<kw>Sub|Function|Property\s+(?:Get|Let|Set)|Type|Enum)\s+(?P<name>[A-Za-z_]\w*)",
    re.IGNORECASE,
)
_VBA_END = re.compile(r"^\s*End\s+(?P<kw>Sub|Function|Property|Type|Enum)\b", re.IGNORECASE)
_VBA_KIND = {"sub": "function", "function": "function", "property": "property",
             "type": "type", "enum": "enum"}


def _vba_leading_comment(lines: list[str], i: int) -> str:
    """Contiguous ``'`` / ``Rem`` comment lines immediately above line ``i`` — a VBA docstring."""
    out: list[str] = []
    k = i - 1
    while k >= 0:
        s = lines[k].strip()
        if s.startswith("'"):
            out.append(s.lstrip("'").strip())
        elif s[:4].lower() == "rem ":
            out.append(s[4:].strip())
        else:
            break
        k -= 1
    out.reverse()
    return " ".join(" ".join(out).split())


def _parse_vba(text: str, path: str) -> list[Symbol]:
    """Extract Sub / Function / Property / Type / Enum from VBA. A ``.cls`` file is a class module: a
    synthetic class (from the file stem) is emitted and its procedures become methods."""
    lines = text.splitlines()
    n = len(lines)
    is_cls = PurePosixPath(path).suffix.lower() == ".cls"
    cls = PurePosixPath(path).stem if is_cls else ""
    out: list[Symbol] = []
    if cls:
        out.append(Symbol("vba", "class", cls, cls, f"Class {cls}", "", 1, n or 1, "", path))

    def _emit(start_i: int, base: str, name: str, signature: str, end_i: int) -> None:
        kind = _VBA_KIND[base]
        if cls and kind == "function":
            kind = "method"
        qual = f"{cls}.{name}" if cls else name
        out.append(Symbol("vba", kind, name, qual, signature, _vba_leading_comment(lines, start_i),
                          start_i + 1, end_i + 1, cls, path))

    # Single pass (O(n)): at most one procedure/block is open at a time (VBA doesn't nest them). A
    # missing `End` is closed at the next declaration or at EOF — never a per-declaration re-scan to
    # the end of the file (red-team #H1: that was O(n²) on unterminated blocks).
    pending: tuple[int, str, str, str] | None = None  # (start_i, base, name, signature)
    i = 0
    while i < n:
        line = lines[i]
        if pending is not None:
            end_m = _VBA_END.match(line)
            if end_m and end_m.group("kw").lower() == pending[1]:
                _emit(*pending, end_i=i)
                pending = None
                i += 1
                continue
            if _VBA_DECL.match(line):  # a new decl before the End: close the old one at the prior line
                _emit(*pending, end_i=i - 1)
                pending = None
                continue  # re-read this line as the new declaration
            i += 1
            continue
        m = _VBA_DECL.match(line)
        if not m:
            i += 1
            continue
        base = m.group("kw").split()[0].lower()  # sub | function | property | type | enum
        sig_parts = [line.strip()]  # signature: join VBA's ` _` line continuations
        j = i
        while sig_parts[-1].endswith("_") and j + 1 < n:
            sig_parts[-1] = sig_parts[-1][:-1].rstrip()
            j += 1
            sig_parts.append(lines[j].strip())
        pending = (i, base, m.group("name"), " ".join(sig_parts).strip())
        i = j + 1
    if pending is not None:  # unterminated block runs to the end of the file
        _emit(*pending, end_i=n - 1)
    return out
