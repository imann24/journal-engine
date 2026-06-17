"""Enrichment: a re-runnable LLM pass that tags each *entry* with mood (1-5),
topics, people, and places, written onto every chunk of that entry as queryable
columns. Idempotent: only entries flagged unenriched are processed, so it can be
stopped and resumed, and re-ingesting an entry re-flags it for re-tagging.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable

from . import store
from .llm import chat

ENRICH_SYSTEM = (
    "Extract structured metadata from a single journal excerpt. Respond with "
    "ONLY a JSON object, no prose, no code fences, with keys: "
    "mood (integer 1-5, 1=very low, 5=very high), "
    "topics (array of 1-4 short lowercase theme tags), "
    "people (array of first names mentioned), "
    "places (array of place names mentioned)."
)


def _safe_json(raw: str) -> dict:
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def _clean_list(v) -> str:
    if not isinstance(v, list):
        v = [v] if v else []
    items = [str(x).strip().lower() for x in v if str(x).strip()]
    # de-dup while preserving order
    seen, out = set(), []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return ", ".join(out)


def _clean_mood(v) -> int:
    try:
        m = int(v)
    except (ValueError, TypeError):
        return 3
    return min(5, max(1, m))


def pending_entries(tbl) -> int:
    """How many distinct entries still need enrichment."""
    try:
        df = store.table_to_df(tbl)
    except Exception:
        return 0
    if df.empty:
        return 0
    return df[~df["enriched"]]["entry_id"].nunique()


def enrich(
    limit: int | None = None,
    tbl=None,
    progress: Callable[[int, int], None] | None = None,
) -> int:
    """Enrich up to `limit` unenriched entries. Returns the number enriched."""
    tbl = tbl or store.open_or_create()
    try:
        df = store.table_to_df(tbl)
    except Exception:
        return 0
    if df.empty:
        return 0

    todo = df[~df["enriched"]]
    entry_ids = list(dict.fromkeys(todo["entry_id"].tolist()))  # stable order
    if limit:
        entry_ids = entry_ids[:limit]
    if not entry_ids:
        return 0

    total = len(entry_ids)
    for n, entry_id in enumerate(entry_ids, 1):
        rows = df[df["entry_id"] == entry_id].sort_values("chunk_index")
        # Enrich on the whole entry (joined chunks), capped for the prompt.
        full_text = "\n\n".join(rows["text"].tolist())[:6000]
        meta = _safe_json(chat(full_text, system=ENRICH_SYSTEM))
        values = {
            "mood": _clean_mood(meta.get("mood", 3)),
            "topics": _clean_list(meta.get("topics", [])),
            "people": _clean_list(meta.get("people", [])),
            "places": _clean_list(meta.get("places", [])),
            "enriched": True,
        }
        tbl.update(where=f"entry_id = '{store.sql_quote(entry_id)}'", values=values)
        if progress:
            progress(n, total)
    return total
