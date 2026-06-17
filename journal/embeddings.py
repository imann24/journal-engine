"""Embeddings via local Ollama (bge-m3). Dense vectors only here; the sparse/
lexical side of hybrid search is handled by LanceDB's BM25 full-text index.
"""

from __future__ import annotations

import ollama

from . import config


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch-embed a list of strings via Ollama. Order is preserved."""
    if not texts:
        return []
    out: list[list[float]] = []
    for i in range(0, len(texts), config.EMBED_BATCH):
        batch = texts[i : i + config.EMBED_BATCH]
        resp = ollama.embed(model=config.EMBED_MODEL, input=batch)
        out.extend(resp["embeddings"])
    return out


def embed_one(text: str) -> list[float]:
    return embed_texts([text])[0]
