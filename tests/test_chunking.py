"""Tests for chunking: short entries stay whole, long ones split with overlap."""

from __future__ import annotations

from journal.chunking import chunk_body, split_paragraphs


def test_short_entry_is_whole():
    body = "A short entry.\n\nWith two paragraphs."
    chunks = chunk_body(body, target_words=400)
    assert len(chunks) == 1
    assert "short entry" in chunks[0] and "two paragraphs" in chunks[0]


def test_empty_body():
    assert chunk_body("") == []
    assert chunk_body("   \n\n  ") == []


def test_long_entry_splits():
    paras = [" ".join(["word"] * 50) for _ in range(10)]  # 500 words, 10 paras
    body = "\n\n".join(paras)
    chunks = chunk_body(body, target_words=120, overlap_paras=1)
    assert len(chunks) > 1
    # Every chunk should respect the budget reasonably (allow one para overflow).
    for c in chunks:
        assert len(c.split()) <= 120 + 50


def test_overlap_carries_paragraph():
    paras = [f"para{i} " + " ".join(["w"] * 40) for i in range(6)]
    body = "\n\n".join(paras)
    chunks = chunk_body(body, target_words=80, overlap_paras=1)
    # Consecutive chunks should share a paragraph marker due to overlap.
    joined_markers = [[t for t in c.split() if t.startswith("para")] for c in chunks]
    # At least one marker from chunk i reappears in chunk i+1.
    shared = any(
        set(joined_markers[i]) & set(joined_markers[i + 1])
        for i in range(len(chunks) - 1)
    )
    assert shared


def test_no_overlap_when_zero():
    paras = [f"para{i} " + " ".join(["w"] * 40) for i in range(6)]
    body = "\n\n".join(paras)
    chunks = chunk_body(body, target_words=80, overlap_paras=0)
    markers = [[t for t in c.split() if t.startswith("para")] for c in chunks]
    shared = any(
        set(markers[i]) & set(markers[i + 1]) for i in range(len(chunks) - 1)
    )
    assert not shared


def test_split_paragraphs_strips_and_drops_blanks():
    assert split_paragraphs("a\n\n\n  b  \n\n") == ["a", "b"]
