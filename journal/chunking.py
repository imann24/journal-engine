"""Chunking: keep short entries whole; split long ones by paragraph with overlap.

Each chunk inherits its entry's date downstream (see ingest.py). This module is
pure text -> list[str]; no dates, no I/O, so it is easy to test.
"""

from __future__ import annotations

import re

from . import config


def split_paragraphs(body: str) -> list[str]:
    """Split on blank lines into non-empty, stripped paragraphs."""
    return [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]


def chunk_body(
    body: str,
    target_words: int | None = None,
    overlap_paras: int | None = None,
) -> list[str]:
    """Return a list of chunk texts.

    - Entries at or under `target_words` are returned whole (one chunk).
    - Longer entries are split on paragraph boundaries, accumulating paragraphs
      up to the word budget, with `overlap_paras` paragraphs carried into the
      next chunk for continuity.
    """
    target_words = target_words or config.CHUNK_TARGET_WORDS
    overlap_paras = config.CHUNK_OVERLAP_PARAS if overlap_paras is None else overlap_paras

    paras = split_paragraphs(body)
    if not paras:
        return []

    if sum(len(p.split()) for p in paras) <= target_words:
        return ["\n\n".join(paras)]

    chunks: list[str] = []
    cur: list[str] = []
    cur_words = 0
    for p in paras:
        w = len(p.split())
        if cur_words + w > target_words and cur:
            chunks.append("\n\n".join(cur))
            cur = cur[-overlap_paras:] if overlap_paras else []
            cur_words = sum(len(c.split()) for c in cur)
        cur.append(p)
        cur_words += w
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks
