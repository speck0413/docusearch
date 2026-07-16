"""Heading-aware, code-preserving chunker (R-ING-4, §7.6)."""

from __future__ import annotations

from docusearch.ingest import ExtractedDoc, Segment, chunk_document


def _doc(*segments: Segment) -> ExtractedDoc:
    return ExtractedDoc(title="t", segments=list(segments))


def test_single_paragraph_is_one_chunk() -> None:
    doc = _doc(Segment("body", "a short paragraph of text", "Intro"))
    chunks = chunk_document(doc, chunk_tokens=350, overlap=40)
    assert len(chunks) == 1
    assert chunks[0].kind == "body"
    assert chunks[0].locator == "Intro"
    assert chunks[0].text == "a short paragraph of text"


def test_code_segment_is_its_own_whole_chunk() -> None:
    code = "def foo():\n    return 1\n" * 100  # long
    doc = _doc(
        Segment("body", "before the code", "API"),
        Segment("code", code, "API"),
        Segment("body", "after the code", "API"),
    )
    chunks = chunk_document(doc, chunk_tokens=350, overlap=40)
    kinds = [c.kind for c in chunks]
    assert kinds == ["body", "code", "body"]
    assert chunks[1].text == code  # never split, byte-for-byte
    assert chunks[1].locator == "API"


def test_long_code_block_never_split() -> None:
    code = " ".join(f"tok{i}" for i in range(1000))
    doc = _doc(Segment("code", code, "H"))
    chunks = chunk_document(doc, chunk_tokens=50, overlap=5)
    assert len(chunks) == 1
    assert chunks[0].kind == "code"


def test_heading_change_starts_new_chunk() -> None:
    doc = _doc(
        Segment("body", "content under one", "A"),
        Segment("body", "content under two", "B"),
    )
    chunks = chunk_document(doc, chunk_tokens=350, overlap=40)
    assert [c.locator for c in chunks] == ["A", "B"]


def test_long_body_splits_with_overlap() -> None:
    words = " ".join(f"w{i}" for i in range(20))
    doc = _doc(Segment("body", words, "H"))
    chunks = chunk_document(doc, chunk_tokens=10, overlap=3)
    assert len(chunks) >= 2
    assert all(len(c.text.split()) <= 10 for c in chunks)
    # overlap: each chunk begins with the last 3 words of the previous one
    first, second = chunks[0].text.split(), chunks[1].text.split()
    assert second[:3] == first[-3:]


def test_no_pure_overlap_tail_chunk() -> None:
    # exactly chunk_tokens words -> a single chunk, not one + an overlap-only tail
    words = " ".join(f"w{i}" for i in range(10))
    doc = _doc(Segment("body", words, "H"))
    chunks = chunk_document(doc, chunk_tokens=10, overlap=3)
    assert len(chunks) == 1


def test_table_segment_becomes_body_chunk() -> None:
    doc = _doc(Segment("table", "Reg | Val\nCTRL | 0x1", "Registers"))
    chunks = chunk_document(doc, chunk_tokens=350, overlap=40)
    assert len(chunks) == 1
    assert chunks[0].kind == "body"
    assert "CTRL | 0x1" in chunks[0].text


def test_ords_are_sequential() -> None:
    doc = _doc(
        Segment("body", "one", "A"),
        Segment("code", "x = 1", "A"),
        Segment("body", "two", "B"),
    )
    chunks = chunk_document(doc, chunk_tokens=350, overlap=40)
    assert [c.ord for c in chunks] == [0, 1, 2]


def test_empty_doc_yields_no_chunks() -> None:
    assert chunk_document(_doc(), chunk_tokens=350, overlap=40) == []


def test_overlap_larger_than_target_is_safe() -> None:
    # pathological config must not loop forever
    words = " ".join(f"w{i}" for i in range(50))
    doc = _doc(Segment("body", words, "H"))
    chunks = chunk_document(doc, chunk_tokens=5, overlap=99)
    assert chunks  # terminates and produces chunks
