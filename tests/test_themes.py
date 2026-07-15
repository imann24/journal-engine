"""Theme discovery on synthetic vectors: two well-separated groups with distinct
vocabularies must come back as two coherently-labeled themes, deterministically.
No Ollama needed (naming via the model is optional and not tested here)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from journal.themes import (
    cluster_entries,
    default_k,
    entry_matrix,
    theme_year_share,
)


def _synthetic():
    ids, dates, vecs, texts = [], [], [], []
    rng = np.random.default_rng(42)
    for i in range(6):  # music cluster near (1, 0, 0, 0)
        ids.append(f"m{i}")
        dates.append(f"2023-0{i + 1}-01")
        vecs.append(np.array([1.0, 0.0, 0.0, 0.0]) + rng.normal(0, 0.02, 4))
        texts.append("piano practice concert melody piano rehearsal song")
    for i in range(6):  # running cluster near (0, 1, 0, 0)
        ids.append(f"r{i}")
        dates.append(f"2024-0{i + 1}-01")
        vecs.append(np.array([0.0, 1.0, 0.0, 0.0]) + rng.normal(0, 0.02, 4))
        texts.append("marathon training race stretch marathon shoes running")
    X = np.array(vecs)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return ids, dates, X, texts


def test_clusters_separate_and_labels_are_distinctive():
    ids, dates, X, texts = _synthetic()
    themes, assign = cluster_entries(ids, dates, X, texts, k=2)

    assert len(themes) == 2
    by_prefix = {
        t.theme_id: {i[0] for i in t.entry_ids} for t in themes
    }
    # each theme holds only one prefix (m or r) — clean separation
    assert all(len(prefixes) == 1 for prefixes in by_prefix.values())

    labels = " ".join(t.label for t in themes)
    assert ("piano" in labels or "melody" in labels or "concert" in labels)
    assert ("marathon" in labels or "race" in labels or "running" in labels)


def test_clustering_is_deterministic():
    ids, dates, X, texts = _synthetic()
    _, a1 = cluster_entries(ids, dates, X, texts, k=2, seed=0)
    _, a2 = cluster_entries(ids, dates, X, texts, k=2, seed=0)
    assert a1["theme_id"].tolist() == a2["theme_id"].tolist()


def test_theme_year_share_rows_sum_to_one():
    ids, dates, X, texts = _synthetic()
    _, assign = cluster_entries(ids, dates, X, texts, k=2)
    share = theme_year_share(assign)
    assert list(share.index) == ["2023", "2024"]
    assert np.allclose(share.sum(axis=1), 1.0, atol=0.01)


def test_entry_matrix_weights_and_normalizes():
    df_all = pd.DataFrame([
        {"entry_id": "e1", "chunk_index": 0, "date": "2024-01-01",
         "text": "a", "word_count": 3, "vector": [1.0, 0.0]},
        {"entry_id": "e1", "chunk_index": 1, "date": "2024-01-01",
         "text": "b", "word_count": 1, "vector": [0.0, 1.0]},
    ])
    ids, dates, X, texts = entry_matrix(df_all)
    assert ids == ["e1"] and dates == ["2024-01-01"]
    assert np.isclose(np.linalg.norm(X[0]), 1.0)
    assert X[0][0] > X[0][1]  # weighted toward the longer chunk
    assert texts == ["a\n\nb"]


def test_default_k_is_bounded():
    assert default_k(4) == 2
    assert 2 <= default_k(100) <= 10
    assert default_k(100000) == 10
