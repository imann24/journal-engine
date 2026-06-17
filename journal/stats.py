"""Analytics over the indexed + enriched columns. Pure dataframe work, no LLM.

Counts are reported at the *entry* level (deduping chunks) so a long entry split
into several chunks counts once. Returns plain pandas objects so both the CLI and
the Streamlit dashboard can consume them.
"""

from __future__ import annotations

import re

import pandas as pd

from . import store


SIGNAL_LEXICONS = {
    "gratitude": {
        "grateful", "gratitude", "thankful", "appreciate", "appreciated",
        "blessing", "blessed", "lucky",
    },
    "strain": {
        "stress", "stressed", "overwhelmed", "anxious", "anxiety", "panic",
        "afraid", "worried", "worry", "pressure", "exhausted", "tired",
        "hard", "heavy", "sad", "angry", "frustrated",
    },
    "connection": {
        "friend", "friends", "family", "mom", "dad", "partner", "together",
        "talked", "called", "met", "shared", "love", "loved", "held",
    },
    "growth": {
        "learned", "realized", "noticed", "understood", "practice", "change",
        "changed", "growing", "growth", "progress", "try", "trying",
    },
    "body": {
        "body", "breath", "breathing", "sleep", "slept", "tired", "energy",
        "walk", "walked", "run", "ran", "exercise", "hungry", "rest",
    },
    "presence": {
        "present", "quiet", "still", "pause", "paused", "breathe", "breath",
        "noticed", "listen", "listened", "aware", "awareness", "mindful",
    },
    "agency": {
        "choose", "chose", "choice", "decide", "decided", "boundary",
        "boundaries", "ask", "asked", "make", "made", "plan", "planned",
    },
    "uncertainty": {
        "maybe", "uncertain", "unsure", "confused", "question", "wonder",
        "wondering", "doubt", "doubtful", "unknown", "stuck",
    },
    "joy": {
        "joy", "happy", "delight", "delighted", "laughed", "laughing",
        "fun", "beautiful", "peace", "peaceful", "good", "excited",
    },
    "restoration": {
        "rest", "rested", "restore", "restored", "repair", "repaired",
        "heal", "healing", "soft", "gentle", "slow", "ease", "safe",
    },
}

VALUE_LEXICONS = {
    "rest": {"rest", "sleep", "slow", "quiet", "ease", "recover", "tired"},
    "connection": {"together", "friend", "family", "call", "called", "love"},
    "clarity": {"clear", "clarity", "understand", "realized", "decide"},
    "safety": {"safe", "secure", "home", "steady", "stable", "trust"},
    "meaning": {"meaning", "purpose", "values", "important", "matter"},
    "freedom": {"free", "freedom", "choice", "space", "permission"},
}

PAST_WORDS = {"was", "were", "had", "remember", "remembered", "used"}
PRESENT_WORDS = {"am", "is", "are", "now", "today", "currently", "feel"}
FUTURE_WORDS = {"will", "going", "tomorrow", "soon", "hope", "plan", "want"}


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z']+", str(text).lower())


def _token_series(entries: pd.DataFrame, column: str) -> pd.Series:
    if entries.empty or column not in entries:
        return pd.Series(dtype=str)
    return (
        entries[column]
        .dropna()
        .str.split(",")
        .explode()
        .str.strip()
        .replace("", pd.NA)
        .dropna()
    )


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
    tokens = _token_series(entries, column)
    return tokens.value_counts().head(n)


def has_enrichment(entries: pd.DataFrame) -> bool:
    return not entries.empty and "mood" in entries and (entries["mood"] > 0).any()


def insight_frame(entries: pd.DataFrame) -> pd.DataFrame:
    """Derived mindfulness signals, one row per entry.

    These are transparent lexical indicators rather than clinical measures. They
    are meant to reveal writing patterns worth reflecting on, not diagnose state.
    """
    if entries.empty:
        return pd.DataFrame()

    rows = []
    for _, entry in entries.iterrows():
        words = _words(entry.get("text", ""))
        word_set = set(words)
        word_count = max(1, int(entry.get("word_count") or len(words) or 1))
        signal_counts = {
            name: sum(1 for word in words if word in lexicon)
            for name, lexicon in SIGNAL_LEXICONS.items()
        }
        value_counts = {
            name: sum(1 for word in words if word in lexicon)
            for name, lexicon in VALUE_LEXICONS.items()
        }
        dominant_signal = max(signal_counts, key=signal_counts.get)
        if signal_counts[dominant_signal] == 0:
            dominant_signal = "unlabeled"

        time_counts = {
            "past": len(word_set & PAST_WORDS),
            "present": len(word_set & PRESENT_WORDS),
            "future": len(word_set & FUTURE_WORDS),
        }
        attention_lens = max(time_counts, key=time_counts.get)
        if time_counts[attention_lens] == 0:
            attention_lens = "reflective"

        row = {
            "entry_id": entry.get("entry_id"),
            "date": entry.get("date"),
            "year": entry.get("year"),
            "mood": entry.get("mood", 0),
            "word_count": word_count,
            "dominant_signal": dominant_signal,
            "attention_lens": attention_lens,
        }
        for name, count in signal_counts.items():
            row[f"signal_{name}"] = count
            row[f"signal_{name}_rate"] = round(count * 100 / word_count, 2)
        for name, count in value_counts.items():
            row[f"need_{name}"] = count
        rows.append(row)
    return pd.DataFrame(rows)


def signal_totals(insights: pd.DataFrame, n: int = 8) -> pd.Series:
    if insights.empty:
        return pd.Series(dtype=int)
    cols = [
        c for c in insights.columns
        if c.startswith("signal_") and not c.endswith("_rate")
    ]
    if not cols:
        return pd.Series(dtype=int)
    totals = insights[cols].sum().sort_values(ascending=False)
    totals.index = totals.index.str.removeprefix("signal_")
    return totals[totals > 0].head(n)


def needs_totals(insights: pd.DataFrame, n: int = 6) -> pd.Series:
    if insights.empty:
        return pd.Series(dtype=int)
    cols = [c for c in insights.columns if c.startswith("need_")]
    if not cols:
        return pd.Series(dtype=int)
    totals = insights[cols].sum().sort_values(ascending=False)
    totals.index = totals.index.str.removeprefix("need_")
    return totals[totals > 0].head(n)


def topic_mood(entries: pd.DataFrame, min_count: int = 2) -> pd.DataFrame:
    if entries.empty or not has_enrichment(entries):
        return pd.DataFrame()
    rows = []
    for _, entry in entries[entries["mood"] > 0].iterrows():
        topics = [t.strip() for t in str(entry.get("topics", "")).split(",") if t.strip()]
        for topic in topics:
            rows.append({
                "topic": topic,
                "entry_id": entry.get("entry_id"),
                "mood": entry.get("mood"),
            })
    if not rows:
        return pd.DataFrame()
    exploded = pd.DataFrame(rows)
    grouped = (
        exploded.groupby("topic")
        .agg(entries=("entry_id", "nunique"), mean_mood=("mood", "mean"))
        .reset_index()
    )
    grouped = grouped[grouped["entries"] >= min_count]
    if grouped.empty:
        return pd.DataFrame()
    grouped["mean_mood"] = grouped["mean_mood"].round(2)
    return grouped.sort_values(["entries", "mean_mood"], ascending=[False, False])


def year_signal_matrix(insights: pd.DataFrame) -> pd.DataFrame:
    if insights.empty or "year" not in insights:
        return pd.DataFrame()
    cols = [c for c in insights.columns if c.startswith("signal_") and c.endswith("_rate")]
    if not cols:
        return pd.DataFrame()
    matrix = insights.groupby("year")[cols].mean().round(2)
    matrix.columns = matrix.columns.str.removeprefix("signal_").str.removesuffix("_rate")
    return matrix


def reflective_prompts(entries: pd.DataFrame, insights: pd.DataFrame) -> list[str]:
    if entries.empty:
        return []
    prompts = []
    totals = signal_totals(insights, n=3)
    needs = needs_totals(insights, n=2)
    if not totals.empty:
        signal = totals.index[0]
        prompts.append(
            f"When {signal} appears in these entries, what is it asking you to "
            "notice before you act?"
        )
    if not needs.empty:
        need = needs.index[0]
        prompts.append(
            f"What small, kind choice would honor your need for {need} this week?"
        )
    if has_enrichment(entries):
        low = entries[entries["mood"] > 0].sort_values("mood").head(1)
        if not low.empty:
            date = low.iloc[0]["date"]
            prompts.append(
                f"Looking back at {date}, what support would you offer that "
                "version of yourself now?"
            )
    if not prompts:
        prompts.append(
            "What pattern in this period feels most alive, and what would it be "
            "like to meet it with patience?"
        )
    return prompts[:3]


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

    insights = insight_frame(entries)
    signals = signal_totals(insights)
    if not signals.empty:
        print("\n=== Mindfulness signals ===")
        print(signals.to_string())
    prompts = reflective_prompts(entries, insights)
    if prompts:
        print("\n=== Reflective prompts ===")
        for prompt in prompts:
            print(f"- {prompt}")
