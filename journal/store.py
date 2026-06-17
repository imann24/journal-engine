"""LanceDB storage layer: an explicit schema, an open-or-create helper, and the
dedup bookkeeping that makes ingestion idempotent and incremental.

Storage is per-chunk, but dedup and enrichment are keyed to the *entry*
(entry_id + content_hash). Re-ingesting an unchanged entry is a no-op; an edited
entry has its old chunks deleted and replaced; only new/changed entries get
embedded, and enrichment only touches entries flagged unenriched.
"""

from __future__ import annotations

import hashlib

import lancedb
import pyarrow as pa

from . import config


def content_hash(body: str) -> str:
    """Stable hash of an entry's normalized text — the dedup key."""
    normalized = body.strip().replace("\r\n", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def schema() -> pa.Schema:
    """Explicit Arrow schema so we can create an empty table before any data
    exists (the web UI may ingest the very first entry into a fresh DB)."""
    return pa.schema(
        [
            pa.field("id", pa.string()),            # chunk id: "<entry_id>::<i>"
            pa.field("entry_id", pa.string()),      # stable per-entry id
            pa.field("source", pa.string()),        # file path or "paste"/"upload"
            pa.field("date", pa.string()),          # ISO YYYY-MM-DD
            pa.field("date_int", pa.int64()),       # YYYYMMDD for range filters
            pa.field("date_source", pa.string()),   # filename|header|mtime|manual|today
            pa.field("text", pa.string()),
            pa.field("word_count", pa.int32()),
            pa.field("content_hash", pa.string()),
            pa.field("chunk_index", pa.int32()),
            pa.field("ingested_at", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), config.EMBED_DIM)),
            # Enrichment columns (mood == 0 / enriched == False => not yet tagged)
            pa.field("mood", pa.int32()),
            pa.field("topics", pa.string()),
            pa.field("people", pa.string()),
            pa.field("places", pa.string()),
            pa.field("enriched", pa.bool_()),
        ]
    )


def connect():
    return lancedb.connect(config.DB_PATH)


def table_to_df(tbl):
    """Read a whole table into pandas via Arrow.

    lancedb's own `to_pandas()` needs the optional `pylance` package; going
    through Arrow uses only pyarrow, which we already depend on.
    """
    return tbl.to_arrow().to_pandas()


def _table_names(db) -> list[str]:
    """List table names across lancedb versions (newer list_tables() returns a
    response object; older table_names() returns a plain list)."""
    try:
        resp = db.list_tables()
        return list(getattr(resp, "tables", resp))
    except Exception:
        return list(db.table_names())


def open_or_create(db=None):
    """Return the entries table, creating an empty one (with FTS index) if needed."""
    db = db or connect()
    if config.TABLE in _table_names(db):
        return db.open_table(config.TABLE)
    tbl = db.create_table(config.TABLE, schema=schema())
    _ensure_fts(tbl)
    return tbl


def _ensure_fts(tbl) -> None:
    try:
        tbl.create_fts_index("text", replace=True)
    except Exception:
        # FTS index can't be built on an empty table on some backends; it gets
        # (re)built after the first ingest via rebuild_fts().
        pass


def rebuild_fts(tbl) -> None:
    """Rebuild the BM25 full-text index — call after inserts/deletes."""
    try:
        tbl.create_fts_index("text", replace=True)
    except Exception:
        pass


def existing_entry_hashes(tbl) -> dict[str, str]:
    """Map entry_id -> content_hash for everything currently indexed.

    Used to decide, without embedding anything, whether each incoming entry is
    new, changed, or unchanged.
    """
    try:
        df = table_to_df(tbl)
    except Exception:
        return {}
    if df.empty or "entry_id" not in df.columns:
        return {}
    return dict(zip(df["entry_id"], df["content_hash"]))


def sql_quote(value: str) -> str:
    """Escape a string for inline use in a LanceDB SQL filter."""
    return value.replace("'", "''")


def delete_entry(tbl, entry_id: str) -> None:
    tbl.delete(f"entry_id = '{sql_quote(entry_id)}'")


def count_rows(tbl) -> int:
    try:
        return tbl.count_rows()
    except Exception:
        return 0


# --------------------------------------------------------------------------- #
# Entry management (list / remove)
# --------------------------------------------------------------------------- #
def list_entries(tbl):
    """One row per entry: entry_id, date, date_source, source, chunks.
    Sorted by date. Returns an empty DataFrame when nothing is indexed."""
    df = table_to_df(tbl)
    if df.empty:
        return df
    first = (
        df.sort_values("chunk_index")
        .groupby("entry_id", as_index=False)
        .first()[["entry_id", "date", "date_source", "source"]]
    )
    counts = df.groupby("entry_id").size().rename("chunks").reset_index()
    out = first.merge(counts, on="entry_id")
    return out.sort_values(["date", "entry_id"]).reset_index(drop=True)


def entry_ids_in_range(tbl, date_from: str | None = None,
                       date_to: str | None = None) -> list[str]:
    df = table_to_df(tbl)
    if df.empty:
        return []
    if date_from:
        df = df[df["date_int"] >= int(date_from.replace("-", ""))]
    if date_to:
        df = df[df["date_int"] <= int(date_to.replace("-", ""))]
    return sorted(df["entry_id"].unique().tolist())


def delete_entries(tbl, entry_ids) -> int:
    """Delete the given entries (all their chunks) and rebuild the FTS index.
    Returns the number of entries removed."""
    ids = [e for e in dict.fromkeys(entry_ids) if e]
    for eid in ids:
        delete_entry(tbl, eid)
    if ids:
        rebuild_fts(tbl)
    return len(ids)


def delete_all(tbl=None):
    """Remove every entry, in place, keeping the table and schema.

    Deleting rows (rather than dropping/recreating the table) keeps any existing
    open table handle valid — important because the web UI caches one.
    """
    tbl = tbl or open_or_create()
    try:
        tbl.delete("true")           # always-true predicate -> delete all rows
    except Exception:
        # Fallback for older backends: drop and recreate.
        db = connect()
        if config.TABLE in _table_names(db):
            db.drop_table(config.TABLE)
        return open_or_create(db)
    rebuild_fts(tbl)
    return tbl
