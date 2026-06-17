"""Analytics over the indexed + enriched columns. Pure dataframe work, no LLM.

Counts are reported at the *entry* level (deduping chunks) so a long entry split
into several chunks counts once. Returns plain pandas objects so both the CLI and
the Streamlit dashboard can consume them.
"""

from __future__ import annotations

import pandas as pd

from . import store


def load_frame(tbl=None, date_from: str | None = None,
              date_to: str | None = None) -> pd.DataFrame:
    """One row per entry (first chunk's metadata), optionally date-filtered."""
    tbl = tbl or store.open_or_create()
    try:
        df = store.table_to_df(tbl)
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df

    if date_from:
        df = df[df["date_int"] >= int(date_from.replace("-", ""))]
    if date_to:
        df = df[df["date_int"] <= int(date_to.replace("-", ""))]
    if df.empty:
        return df

    # Collapse to one row per entry.
    entries = (
        df.sort_values("chunk_index")
        .groupby("entry_id", as_index=False)
        .first()
    )
    entries["year"] = entries["date"].str[:4]
    return entries


def entries_per_year(entries: pd.DataFrame) -> pd.Series:
    if entries.empty:
        return pd.Series(dtype=int)
    return entries.groupby("year").size().sort_index()


def mean_mood_per_year(entries: pd.DataFrame) -> pd.Series:
    if entries.empty or "mood" not in entries:
        return pd.Series(dtype=float)
    m = entries[entries["mood"] > 0]
    if m.empty:
        return pd.Series(dtype=float)
    return m.groupby("year")["mood"].mean().round(2).sort_index()


def top_tokens(entries: pd.DataFrame, column: str, n: int = 12) -> pd.Series:
    if entries.empty or column not in entries:
        return pd.Series(dtype=int)
    tokens = (
        entries[column]
        .dropna()
        .str.split(",")
        .explode()
        .str.strip()
        .replace("", pd.NA)
        .dropna()
    )
    return tokens.value_counts().head(n)


def has_enrichment(entries: pd.DataFrame) -> bool:
    return not entries.empty and "mood" in entries and (entries["mood"] > 0).any()


def print_stats(tbl=None) -> None:
    """CLI text report."""
    entries = load_frame(tbl)
    if entries.empty:
        print("No entries indexed yet. Run `ingest` first.")
        return

    print(f"\n{len(entries)} entries indexed.")
    print("\n=== Entries per year ===")
    print(entries_per_year(entries).to_string())

    if has_enrichment(entries):
        print("\n=== Mean mood per year (1-5) ===")
        print(mean_mood_per_year(entries).to_string())
        for label, col in (("people", "people"), ("places", "places"),
                           ("topics", "topics")):
            print(f"\n=== Top {label} ===")
            s = top_tokens(entries, col)
            print(s.to_string() if not s.empty else "(none)")
    else:
        print("\n(Run `enrich` to unlock mood / people / place / topic analytics.)")
