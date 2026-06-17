from __future__ import annotations

import pandas as pd

from journal.stats import (
    insight_frame,
    needs_totals,
    reflective_prompts,
    signal_totals,
    topic_mood,
    year_signal_matrix,
)


def _entries():
    return pd.DataFrame(
        [
            {
                "entry_id": "e1",
                "date": "2024-01-01",
                "year": "2024",
                "text": "I feel grateful and peaceful after a quiet walk with a friend.",
                "word_count": 12,
                "mood": 5,
                "topics": "friendship, rest",
            },
            {
                "entry_id": "e2",
                "date": "2024-01-02",
                "year": "2024",
                "text": "I am stressed and tired, but I noticed I need rest and space.",
                "word_count": 13,
                "mood": 2,
                "topics": "work, rest",
            },
            {
                "entry_id": "e3",
                "date": "2025-01-01",
                "year": "2025",
                "text": "I will choose a boundary and make a plan with clarity.",
                "word_count": 11,
                "mood": 4,
                "topics": "work, agency",
            },
        ]
    )


def test_insight_frame_extracts_signals_and_lens():
    insights = insight_frame(_entries())

    first = insights[insights["entry_id"] == "e1"].iloc[0]
    assert first["signal_gratitude"] == 1
    assert first["signal_connection"] >= 1

    third = insights[insights["entry_id"] == "e3"].iloc[0]
    assert third["dominant_signal"] == "agency"
    assert third["attention_lens"] == "future"


def test_totals_matrix_and_prompts_are_populated():
    entries = _entries()
    insights = insight_frame(entries)

    assert signal_totals(insights).index[0] in {"agency", "strain", "presence"}
    assert needs_totals(insights).loc["rest"] >= 2
    assert "2024" in year_signal_matrix(insights).index
    assert reflective_prompts(entries, insights)


def test_topic_mood_groups_topics_with_min_count():
    grouped = topic_mood(_entries(), min_count=2)

    assert grouped.set_index("topic").loc["rest", "entries"] == 2
    assert grouped.set_index("topic").loc["work", "mean_mood"] == 3.0
