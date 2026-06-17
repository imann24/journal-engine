"""Tests that ingestion is idempotent and incremental — re-running never
duplicates, unchanged entries are skipped, and edited entries are replaced.

Ollama is not available in CI, so embeddings are monkeypatched with deterministic
fake vectors; everything else (dating, dedup, LanceDB I/O) is exercised for real.
"""

from __future__ import annotations

import pytest

from journal import config, ingest as ingest_mod, store


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "db"))
    monkeypatch.setattr(config, "EMBED_DIM", 8)

    def fake_embed(texts):
        return [[float(len(t) % 7)] * 8 for t in texts]

    # ingest.py imported embed_texts into its namespace; patch it there.
    monkeypatch.setattr(ingest_mod, "embed_texts", fake_embed)
    return tmp_path


def _entries(tbl):
    df = store.table_to_df(tbl)
    return df["entry_id"].nunique(), len(df)


def test_paste_dedup(fresh_db):
    s1 = ingest_mod.ingest_paste("Hello world, a journal entry.", "2020-01-01")
    assert s1.added == 1
    tbl = store.open_or_create()
    assert _entries(tbl)[0] == 1

    # Same content again -> skipped, no new rows.
    s2 = ingest_mod.ingest_paste("Hello world, a journal entry.", "2020-01-01")
    assert s2.added == 0 and s2.skipped == 1
    assert _entries(tbl)[0] == 1


def test_dir_reingest_is_noop(fresh_db):
    d = fresh_db / "journals"
    d.mkdir()
    (d / "2013-05-04.txt").write_text("First entry about Max in Boston.")
    (d / "2014-06-01.txt").write_text("Second entry, feeling good.")

    s1 = ingest_mod.ingest_dir(str(d))
    assert s1.added == 2
    tbl = store.open_or_create()
    n_entries, n_rows = _entries(tbl)
    assert n_entries == 2

    # Re-ingest unchanged -> all skipped, identical row count.
    s2 = ingest_mod.ingest_dir(str(d))
    assert s2.skipped == 2 and s2.added == 0
    assert _entries(tbl) == (n_entries, n_rows)


def test_edited_entry_is_replaced_not_duplicated(fresh_db):
    d = fresh_db / "journals"
    d.mkdir()
    f = d / "2013-05-04.txt"
    f.write_text("Original text.")
    ingest_mod.ingest_dir(str(d))

    f.write_text("Edited and longer text now, still one entry.")
    s = ingest_mod.ingest_dir(str(d))
    assert s.updated == 1 and s.added == 0

    tbl = store.open_or_create()
    n_entries, _ = _entries(tbl)
    assert n_entries == 1
    df = store.table_to_df(tbl)
    assert "Edited and longer" in df.iloc[0]["text"]
    # The edited entry is re-flagged for enrichment.
    assert bool(df["enriched"].iloc[0]) is False


def test_date_source_recorded(fresh_db):
    d = fresh_db / "journals"
    d.mkdir()
    (d / "2013-05-04.txt").write_text("Filename-dated entry.")
    (d / "untitled.txt").write_text("May 4, 2013\nHeader-dated entry.")
    s = ingest_mod.ingest_dir(str(d))
    assert s.date_sources.get("filename") == 1
    assert s.date_sources.get("header") == 1


def test_batch_split(fresh_db):
    blob = "Entry one about home.\n\n---\n\nEntry two about work."
    s = ingest_mod.ingest_batch(blob)
    assert s.added == 2
