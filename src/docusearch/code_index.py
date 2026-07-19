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
}

# File extension → language.
_EXT: dict[str, str] = {
    ".py": "python", ".pyi": "python",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".go": "go", ".rs": "rust", ".java": "java",
}

# Kinds whose scope makes a nested "function" a "method", and defines a qualname boundary.
_CLASS_LIKE = frozenset({"class", "struct", "interface", "trait", "enum", "impl"})
# Node types that begin a definition's body — everything before the first one is the signature.
_BODY_TYPES = frozenset({
    "block", "statement_block", "class_body", "interface_body", "enum_body", "enum_class_body",
    "declaration_list", "field_declaration_list", "enum_variant_list",
})
_COMMENT_TYPES = frozenset({"comment", "line_comment", "block_comment"})


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
    return field_node.text.decode("utf-8", "replace") if field_node and field_node.text else None


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
    if language not in LANG_SPECS:
        raise ValueError(f"unsupported language: {language!r} (have: {', '.join(LANG_SPECS)})")
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
        if node.type in spec.kinds:
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
