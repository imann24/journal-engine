"""Chart-ready derived insights over the entry frame: "on this day" lookbacks,
writing cadence and streaks, mood swings, and per-entity (people/places/topics)
catalogs and timelines.

Pure pandas/stdlib — no LLM, no Ollama — so everything here is instant and
deterministic. All functions take the one-row-per-entry frame produced by
``stats.load_frame`` so the CLI and the web UI share one code path. Mood values
of 0 mean "not enriched yet" and are excluded from averages.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd


def _dated(entries: pd.DataFrame) -> pd.DataFrame:
    d = entries.copy()
    d["dt"] = pd.to_datetime(d["date"], errors="coerce")
    return d.dropna(subset=["dt"])


def full_text_map(df_all: pd.DataFrame) -> pd.Series:
    """entry_id -> full entry text (chunks rejoined in order).

    ``load_frame`` keeps only the first chunk's text per entry; use this (from
    the raw per-chunk frame) wherever the whole entry is displayed or quoted.
    """
    if df_all.empty:
        return pd.Series(dtype=str)
    return (
        df_all.sort_values("chunk_index")
        .groupby("entry_id")["text"]
        .apply("\n\n".join)
    )


# --------------------------------------------------------------------------- #
# Cadence: on this day, streaks, rhythm
# --------------------------------------------------------------------------- #
def on_this_day(entries: pd.DataFrame, today: date | None = None,
                window: int = 0) -> pd.DataFrame:
    """Entries from any year whose month-day falls within ±window days of today,
    newest first. ``window`` widens the net for sparse journals."""
    if entries.empty:
        return entries
    today = today or date.today()
    keys = {
        (today + timedelta(days=off)).strftime("%m-%d")
        for off in range(-window, window + 1)
    }
    d = _dated(entries)
    hits = d[d["dt"].dt.strftime("%m-%d").isin(keys)]
    return hits.sort_values("date", ascending=False).reset_index(drop=True)


def cadence_stats(entries: pd.DataFrame, today: date | None = None,
                  total_words: int | None = None) -> dict:
    """Corpus-wide writing-rhythm facts: span, streaks, gaps.

    ``total_words`` should be summed over the per-chunk frame (the entry frame
    only carries the first chunk's word_count); falls back to that if omitted.
    """
    if entries.empty:
        return {}
    today = today or date.today()
    d = _dated(entries)
    if d.empty:
        return {}
    days = sorted(d["dt"].dt.date.unique())

    longest_streak = streak = 1
    longest_gap = 0
    gaps = []
    for prev, cur in zip(days, days[1:]):
        delta = (cur - prev).days
        if delta == 1:
            streak += 1
            longest_streak = max(longest_streak, streak)
        else:
            streak = 1
        gaps.append(delta)
        longest_gap = max(longest_gap, delta)

    return {
        "entries": int(len(entries)),
        "distinct_days": len(days),
        "first": days[0].isoformat(),
        "last": days[-1].isoformat(),
        "span_days": (days[-1] - days[0]).days,
        "years": int(d["dt"].dt.year.nunique()),
        "longest_streak": longest_streak,
        "longest_gap": longest_gap,
        "median_gap": float(pd.Series(gaps).median()) if gaps else 0.0,
        "current_gap": (today - days[-1]).days,
        "total_words": int(total_words if total_words is not None
                           else entries["word_count"].sum()),
    }


def writing_rhythm(entries: pd.DataFrame) -> pd.DataFrame:
    """Year × month entry counts (all 12 month columns, zero-filled) — the
    calendar-heatmap frame."""
    if entries.empty:
        return pd.DataFrame()
    d = _dated(entries)
    if d.empty:
        return pd.DataFrame()
    counts = (
        d.groupby([d["dt"].dt.year, d["dt"].dt.month])
        .size()
        .unstack(fill_value=0)
    )
    counts = counts.reindex(columns=range(1, 13), fill_value=0)
    counts.index.name = "year"
    counts.columns = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    return counts


# --------------------------------------------------------------------------- #
# Mood dynamics
# --------------------------------------------------------------------------- #
def mood_swings(entries: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Largest mood deltas between consecutive enriched entries — the days most
    worth re-reading. Columns: prev_date, prev_mood, date, mood, delta."""
    if entries.empty or "mood" not in entries:
        return pd.DataFrame()
    d = _dated(entries)
    d = d[d["mood"] > 0].sort_values("dt")
    if len(d) < 2:
        return pd.DataFrame()
    d = d.assign(
        prev_date=d["date"].shift(),
        prev_mood=d["mood"].shift(),
        prev_entry_id=d["entry_id"].shift(),
    )
    d = d.dropna(subset=["prev_mood"])
    d["delta"] = d["mood"] - d["prev_mood"]
    d = d.reindex(d["delta"].abs().sort_values(ascending=False).index)
    cols = ["prev_date", "prev_mood", "prev_entry_id", "date", "mood",
            "delta", "entry_id"]
    return d.head(n)[cols].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Entities: people / places / topics
# --------------------------------------------------------------------------- #
def _entity_pairs(entries: pd.DataFrame, column: str) -> pd.DataFrame:
    """(entry_id, date, year, mood, token) rows, one per entity mention."""
    if entries.empty or column not in entries:
        return pd.DataFrame()
    d = entries[["entry_id", "date", "year", "mood", column]].copy()
    d[column] = d[column].fillna("").str.split(",")
    d = d.explode(column)
    d["token"] = d[column].str.strip()
    d = d[d["token"] != ""]
    return d.drop(columns=[column]).reset_index(drop=True)


def entity_catalog(entries: pd.DataFrame, column: str,
                   min_count: int = 1) -> pd.DataFrame:
    """Every person/place/topic with mentions, first/last seen and mean mood,
    most-mentioned first."""
    pairs = _entity_pairs(entries, column)
    if pairs.empty:
        return pd.DataFrame()
    moody = pairs[pairs["mood"] > 0]
    mood = moody.groupby("token")["mood"].mean().round(2)
    cat = (
        pairs.groupby("token")
        .agg(mentions=("entry_id", "nunique"),
             first_seen=("date", "min"),
             last_seen=("date", "max"))
        .reset_index()
    )
    cat["mean_mood"] = cat["token"].map(mood)
    cat = cat[cat["mentions"] >= min_count]
    return cat.sort_values(
        ["mentions", "last_seen"], ascending=[False, False]
    ).reset_index(drop=True)


def entity_timeline(entries: pd.DataFrame, column: str,
                    token: str) -> pd.DataFrame:
    """Per-year mentions + mean mood for one entity."""
    pairs = _entity_pairs(entries, column)
    if pairs.empty:
        return pd.DataFrame()
    sub = pairs[pairs["token"] == token]
    if sub.empty:
        return pd.DataFrame()
    moody = sub[sub["mood"] > 0]
    out = (
        sub.groupby("year")
        .agg(mentions=("entry_id", "nunique"))
        .join(moody.groupby("year")["mood"].mean().round(2).rename("mean_mood"))
        .reset_index()
        .sort_values("year")
    )
    return out.reset_index(drop=True)


def entity_entries(entries: pd.DataFrame, column: str,
                   token: str) -> pd.DataFrame:
    """The entries mentioning an entity, newest first."""
    pairs = _entity_pairs(entries, column)
    if pairs.empty:
        return pd.DataFrame()
    ids = set(pairs[pairs["token"] == token]["entry_id"])
    sub = entries[entries["entry_id"].isin(ids)]
    return sub.sort_values("date", ascending=False).reset_index(drop=True)


def co_mentions(entries: pd.DataFrame, column: str, token: str,
                other_column: str, n: int = 8) -> pd.Series:
    """What co-occurs with an entity: top tokens of ``other_column`` within the
    entries that mention ``token`` (the token itself excluded)."""
    pairs = _entity_pairs(entries, column)
    if pairs.empty:
        return pd.Series(dtype=int)
    ids = set(pairs[pairs["token"] == token]["entry_id"])
    others = _entity_pairs(entries[entries["entry_id"].isin(ids)], other_column)
    if others.empty:
        return pd.Series(dtype=int)
    counts = others[others["token"] != token]["token"].value_counts()
    return counts.head(n)


# --------------------------------------------------------------------------- #
# Highlights: a handful of notable, plainly-worded facts for the Home tab
# --------------------------------------------------------------------------- #
def highlights(entries: pd.DataFrame, total_words: int | None = None,
               n: int = 6) -> list[str]:
    if entries.empty:
        return []
    d = _dated(entries)
    if d.empty:
        return []
    out: list[str] = []
    cad = cadence_stats(entries, total_words=total_words)

    out.append(
        f"{cad['entries']} entries across {cad['years']} year(s) — "
        f"{cad['total_words']:,} words between {cad['first']} and {cad['last']}."
    )

    monthly = d.groupby(d["dt"].dt.to_period("M")).size()
    if not monthly.empty:
        busiest = monthly.idxmax()
        out.append(
            f"Your most prolific month was {busiest.strftime('%B %Y')} "
            f"({int(monthly.max())} entries)."
        )

    if cad["longest_gap"] >= 30:
        gaps = d.sort_values("dt")
        prev = gaps["dt"].shift()
        deltas = (gaps["dt"] - prev).dt.days
        i = deltas.idxmax()
        out.append(
            f"Longest quiet stretch: {int(deltas.loc[i])} days, ending when you "
            f"wrote again on {gaps.loc[i, 'date']}."
        )

    longest = entries.loc[entries["word_count"].idxmax()]
    out.append(
        f"Your longest entry opens on {longest['date']} "
        f"({int(longest['word_count'])}+ words)."
    )

    moody = d[d["mood"] > 0]
    if len(moody) >= 3:
        best = moody.loc[moody["mood"].idxmax()]
        low = moody.loc[moody["mood"].idxmin()]
        out.append(
            f"Brightest day on record: {best['date']} (mood {int(best['mood'])}/5); "
            f"heaviest: {low['date']} (mood {int(low['mood'])}/5)."
        )

    people = entity_catalog(entries, "people")
    if not people.empty:
        recent_cut = d["dt"].max() - pd.Timedelta(days=180)
        newcomers = people[pd.to_datetime(people["first_seen"]) >= recent_cut]
        if not newcomers.empty:
            names = ", ".join(newcomers["token"].head(3))
            out.append(f"Newer faces in the story: {names}.")

    return out[:n]
