"""Digest caching + grounding plumbing, with the LLM monkeypatched (tests never
require Ollama). The cache must hit on identical inputs, miss when content
changes, and honor force=True."""

from __future__ import annotations

import pandas as pd
import pytest

from journal import config
from journal import digest as digest_mod


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "db"))
    calls = []

    def fake_chat(prompt, system=None, model=None, **kw):
        calls.append(prompt)
        return "A reflective digest citing (2024-01-01)."

    # digest.py imported `chat` into its namespace; patch it there.
    monkeypatch.setattr(digest_mod, "chat", fake_chat)
    return calls


def _entries(hash_suffix=""):
    return pd.DataFrame([
        {"entry_id": "e1", "date": "2024-01-01", "mood": 4,
         "topics": "work", "people": "max", "word_count": 10,
         "text": "went to work", "content_hash": "h1" + hash_suffix},
        {"entry_id": "e2", "date": "2024-02-01", "mood": 2,
         "topics": "rest", "people": "", "word_count": 12,
         "text": "took a rest day", "content_hash": "h2"},
    ])


def test_compose_caches_by_period_model_and_content(env):
    calls = env
    first = digest_mod.compose(_entries(), date_from="2024-01-01",
                               date_to="2024-12-31", model="m1")
    assert not first.cached and len(calls) == 1
    assert first.n_entries == 2 and first.n_sampled == 2

    again = digest_mod.compose(_entries(), date_from="2024-01-01",
                               date_to="2024-12-31", model="m1")
    assert again.cached and again.text == first.text
    assert len(calls) == 1  # no second LLM call

    # editing an entry (new content hash) invalidates the cache
    edited = digest_mod.compose(_entries(hash_suffix="x"),
                                date_from="2024-01-01",
                                date_to="2024-12-31", model="m1")
    assert not edited.cached and len(calls) == 2

    # a different model composes fresh too
    other = digest_mod.compose(_entries(), date_from="2024-01-01",
                               date_to="2024-12-31", model="m2")
    assert not other.cached and len(calls) == 3


def test_force_recomposes(env):
    calls = env
    digest_mod.compose(_entries(), date_from="a", date_to="b", model="m")
    forced = digest_mod.compose(_entries(), date_from="a", date_to="b",
                                model="m", force=True)
    assert not forced.cached and len(calls) == 2


def test_prompt_grounds_in_dates_texts_and_overview(env):
    calls = env
    texts = pd.Series({"e1": "FULL TEXT ONE", "e2": "FULL TEXT TWO"})
    digest_mod.compose(_entries(), texts=texts, date_from="2024-01-01",
                       date_to="2024-12-31", model="m")
    prompt = calls[0]
    assert "[2024-01-01] FULL TEXT ONE" in prompt
    assert "[2024-02-01] FULL TEXT TWO" in prompt
    assert "2 entries in this period" in prompt
    assert "work" in prompt  # topics reach the overview line


def test_empty_frame_short_circuits(env):
    res = digest_mod.compose(pd.DataFrame(), date_from="a", date_to="b")
    assert res.n_entries == 0 and "No entries" in res.text
    assert env == []  # no LLM call


def test_sample_entries_keeps_ends_and_caps():
    entries = pd.DataFrame({
        "entry_id": [f"e{i}" for i in range(40)],
        "date": [f"2024-01-{i + 1:02d}" for i in range(31)]
        + [f"2024-02-{i + 1:02d}" for i in range(9)],
    })
    sampled = digest_mod.sample_entries(entries, max_n=10)
    assert len(sampled) == 10
    assert sampled["date"].iloc[0] == "2024-01-01"
    assert sampled["date"].iloc[-1] == "2024-02-09"
