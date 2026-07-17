"""Citation grammar: parse / resolve / verify (R-CIT-1, §11)."""

from __future__ import annotations

from docusearch import citations


def test_parse_gk() -> None:
    cites = citations.parse("The sky is blue. [GK]")
    assert len(cites) == 1
    assert cites[0].kind == "GK"
    assert cites[0].doc_id is None and cites[0].chunk_id is None


def test_parse_doc() -> None:
    cites = citations.parse("SPI runs at 1 MHz. [D:812#90312]")
    assert len(cites) == 1
    c = cites[0]
    assert c.kind == "doc" and c.doc_id == 812 and c.chunk_id == 90312
    assert c.raw == "[D:812#90312]"


def test_parse_mixed_and_ordered() -> None:
    text = "Fact one [D:1#2]. General thing [GK]. Fact two [D:3#4]."
    cites = citations.parse(text)
    assert [(c.kind, c.doc_id, c.chunk_id) for c in cites] == [
        ("doc", 1, 2),
        ("GK", None, None),
        ("doc", 3, 4),
    ]


def test_parse_none() -> None:
    assert citations.parse("no citations here") == []


def test_parse_ignores_malformed() -> None:
    # missing #chunk, or non-numeric -> not a valid citation
    assert citations.parse("[D:812] [D:x#y] [D:1#]") == []


def test_resolve_doc_url() -> None:
    c = citations.Citation(kind="doc", raw="[D:812#90312]", doc_id=812, chunk_id=90312)
    assert (
        citations.resolve(c, "http://host:8321") == "http://host:8321/v1/documents/812?chunk=90312"
    )


def test_resolve_trims_trailing_slash() -> None:
    c = citations.Citation(kind="doc", raw="[D:1#2]", doc_id=1, chunk_id=2)
    assert citations.resolve(c, "http://host:8321/") == "http://host:8321/v1/documents/1?chunk=2"


def test_resolve_gk_is_none() -> None:
    assert citations.resolve(citations.Citation(kind="GK", raw="[GK]"), "http://h") is None


def test_verify_all_allowed_is_empty() -> None:
    text = "A [D:1#2]. B [D:1#3]. C [GK]."
    assert citations.verify(text, {2, 3}) == []


def test_verify_flags_out_of_evidence() -> None:
    text = "A [D:1#2]. B [D:9#999]."  # chunk 999 not in evidence
    violations = citations.verify(text, {2, 3})
    assert len(violations) == 1
    assert violations[0].chunk_id == 999


def test_verify_ignores_gk() -> None:
    assert citations.verify("only general knowledge [GK]", set()) == []


def test_citation_error_is_exported() -> None:
    assert issubclass(citations.CitationError, Exception)


def test_render_references_numbers_and_dedupes() -> None:
    text = "A [D:1#2]. B [D:1#2]. C [D:3#4]. D [GK]."
    body, refs = citations.render_references(text, "http://host:8321")
    # each distinct catalog citation numbered once, superscripted; GK left inline
    assert "[GK]" in body
    assert refs[0].startswith("1.")
    assert "http://host:8321/v1/documents/1?chunk=2" in refs[0]
    assert len(refs) == 2  # [D:1#2] and [D:3#4], deduped
