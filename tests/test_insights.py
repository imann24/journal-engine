"""Insights are pure pandas — no Ollama, no LanceDB — so we exercise them on a
small synthetic entry frame with known streaks, gaps, moods, and entities."""

from __future__ import annotations

from datetime import date

import pandas as pd

from journal.insights import (
    cadence_stats,
    co_mentions,
    entity_catalog,
    entity_entries,
    entity_timeline,
    full_text_map,
    highlights,
    mood_swings,
    on_this_day,
    writing_rhythm,
)


def _entries():
    rows = [
        # a 3-day streak in July 2023
        ("e1", "2023-07-14", 5, "max, anna", "travel, music", 100),
        ("e2", "2023-07-15", 4, "max", "travel", 80),
        ("e3", "2023-07-16", 1, "anna", "work", 120),
        # a long gap, then 2024
        ("e4", "2024-07-14", 3, "max", "work, music", 60),
        ("e5", "2024-12-01", 0, "", "", 500),  # unenriched
    ]
    df = pd.DataFrame(
        rows, columns=["entry_id", "date", "mood", "people", "topics",
                       "word_count"],
    )
    df["year"] = df["date"].str[:4]
    df["text"] = "body of " + df["entry_id"]
    return df


def test_on_this_day_matches_month_day_across_years():
    otd = on_this_day(_entries(), today=date(2026, 7, 14))
    assert otd["entry_id"].tolist() == ["e4", "e1"]  # newest first

    # window widens the net across a month boundary safely
    wide = on_this_day(_entries(), today=date(2026, 7, 14), window=2)
    assert set(wide["entry_id"]) == {"e1", "e2", "e3", "e4"}


def test_cadence_stats_streaks_and_gaps():
    cad = cadence_stats(_entries(), today=date(2024, 12, 11), total_words=860)
    assert cad["entries"] == 5
    assert cad["longest_streak"] == 3
    assert cad["longest_gap"] == 364  # 2023-07-16 -> 2024-07-14
    assert cad["current_gap"] == 10
    assert cad["first"] == "2023-07-14" and cad["last"] == "2024-12-01"
    assert cad["total_words"] == 860


def test_writing_rhythm_is_year_by_month():
    rhythm = writing_rhythm(_entries())
    assert list(rhythm.index) == [2023, 2024]
    assert rhythm.loc[2023, "Jul"] == 3
    assert rhythm.loc[2024, "Dec"] == 1
    assert rhythm.loc[2024, "Jan"] == 0  # zero-filled months


def test_mood_swings_ranked_by_delta_and_skip_unenriched():
    swings = mood_swings(_entries())
    top = swings.iloc[0]
    assert (top["prev_date"], top["date"]) == ("2023-07-15", "2023-07-16")
    assert top["delta"] == -3
    assert "e5" not in swings["entry_id"].tolist()  # mood 0 excluded


def test_entity_catalog_counts_and_mood():
    cat = entity_catalog(_entries(), "people")
    max_row = cat[cat["token"] == "max"].iloc[0]
    assert max_row["mentions"] == 3
    assert max_row["first_seen"] == "2023-07-14"
    assert max_row["last_seen"] == "2024-07-14"
    assert max_row["mean_mood"] == 4.0  # (5 + 4 + 3) / 3
    anna_row = cat[cat["token"] == "anna"].iloc[0]
    assert anna_row["mean_mood"] == 3.0  # (5 + 1) / 2


def test_entity_timeline_and_entries_and_co_mentions():
    tl = entity_timeline(_entries(), "people", "max")
    assert tl[tl["year"] == "2023"]["mentions"].iloc[0] == 2

    ents = entity_entries(_entries(), "people", "anna")
    assert ents["entry_id"].tolist() == ["e3", "e1"]  # newest first

    co = co_mentions(_entries(), "people", "max", "topics")
    assert co.index[0] in ("travel", "music", "work")
    assert co["travel"] == 2


def test_highlights_are_plain_sentences():
    facts = highlights(_entries(), total_words=860)
    assert facts and all(isinstance(f, str) for f in facts)
    assert "5 entries" in facts[0]
    assert any("quiet stretch" in f for f in facts)


def test_full_text_map_rejoins_chunks_in_order():
    df_all = pd.DataFrame([
        {"entry_id": "e1", "chunk_index": 1, "text": "second"},
        {"entry_id": "e1", "chunk_index": 0, "text": "first"},
        {"entry_id": "e2", "chunk_index": 0, "text": "solo"},
    ])
    texts = full_text_map(df_all)
    assert texts["e1"] == "first\n\nsecond"
    assert texts["e2"] == "solo"
