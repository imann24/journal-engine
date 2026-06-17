"""Hybrid retrieval: dense (vector) + sparse (BM25 FTS), fused with reciprocal-
rank fusion, with an optional date-range prefilter.
"""

from __future__ import annotations

from . import config, store
from .embeddings import embed_one


def date_filter(date_from: str | None, date_to: str | None) -> str | None:
    """Build a LanceDB SQL clause over date_int from ISO date strings."""
    clauses = []
    if date_from:
        clauses.append(f"date_int >= {int(date_from.replace('-', ''))}")
    if date_to:
        clauses.append(f"date_int <= {int(date_to.replace('-', ''))}")
    return " AND ".join(clauses) if clauses else None


def hybrid_search(
    query: str,
    k: int = 8,
    date_from: str | None = None,
    date_to: str | None = None,
    pool: int = 40,
    rrf_k: int = 60,
    tbl=None,
) -> list[dict]:
    tbl = tbl or store.open_or_create()
    if store.count_rows(tbl) == 0:
        return []
    flt = date_filter(date_from, date_to)

    # Dense side
    qvec = embed_one(query)
    vq = tbl.search(qvec).limit(pool)
    if flt:
        vq = vq.where(flt, prefilter=True)
    dense = vq.to_list()

    # Sparse / lexical side (BM25). FTS may be unavailable on a tiny/empty index.
    try:
        fq = tbl.search(query, query_type="fts").limit(pool)
        if flt:
            fq = fq.where(flt, prefilter=True)
        sparse = fq.to_list()
    except Exception:
        sparse = []

    # Reciprocal-rank fusion
    scores: dict[str, float] = {}
    sestore: dict[str, dict] = {}
    for ranked in (dense, sparse):
        for rank, row in enumerate(ranked):
            cid = row["id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
            sestore[cid] = row
    top = sorted(scores, key=scores.get, reverse=True)[:k]
    return [sestore[cid] for cid in top]
