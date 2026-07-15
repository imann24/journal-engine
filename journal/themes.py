"""Semantic theme discovery over the vectors we already store.

Clusters per-entry bge-m3 embeddings with a small deterministic k-means
(seeded k-means++, numpy only — numpy already ships as a hard dependency of
pandas/pyarrow, so nothing new is installed), then labels each cluster with its
most distinctive words relative to the whole corpus. No LLM and no network:
themes come straight out of the index. ``name_themes`` optionally asks the
local Ollama model for friendlier titles.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import store

_TOKEN_RE = re.compile(r"[a-zA-Z']{3,}")

# Function words plus journal filler that would otherwise dominate every label.
_STOPWORDS = {
    "the", "and", "was", "that", "for", "with", "this", "but", "not", "are",
    "have", "had", "you", "she", "they", "them", "then", "there", "some",
    "would", "really", "today", "don't", "didn't", "i'm", "it's", "its",
    "get", "got", "going", "went", "one", "did", "can", "could", "will",
    "much", "more", "also", "think", "know", "feel", "felt", "feeling",
    "because", "after", "before", "day", "time", "now", "still", "even",
    "things", "thing", "lot", "bit", "how", "into", "just", "like", "about",
    "all", "out", "when", "what", "been", "were", "over", "back", "again",
    "very", "good", "new", "see", "way", "make", "made", "want", "wanted",
    "little", "last", "from", "which", "while", "where", "who", "your",
    "our", "his", "her", "him", "has", "does", "doesn't", "won't", "can't",
    "myself", "being", "than", "too", "off", "down", "most", "other",
    "something", "someone", "maybe", "though", "around", "here", "should",
    "need", "needs", "trying", "kind", "sort", "actually", "right", "first",
}


@dataclass
class Theme:
    theme_id: int
    label: str
    terms: list[str]
    size: int
    entry_ids: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Corpus -> per-entry matrix
# --------------------------------------------------------------------------- #
def entry_matrix(df_all: pd.DataFrame):
    """(ids, dates, X, texts) — one L2-normalized, word-count-weighted mean
    vector per entry, plus the joined text (capped) for labeling."""
    if df_all.empty or "vector" not in df_all:
        return [], [], np.zeros((0, 0)), []
    ids, dates, vecs, texts = [], [], [], []
    for eid, grp in df_all.sort_values("chunk_index").groupby("entry_id", sort=False):
        mat = np.array([np.asarray(v, dtype=np.float64) for v in grp["vector"]])
        w = grp["word_count"].to_numpy(dtype=np.float64)
        w = np.where(w > 0, w, 1.0)
        mean = (mat * w[:, None]).sum(axis=0) / w.sum()
        norm = np.linalg.norm(mean)
        if norm == 0:
            continue
        ids.append(str(eid))
        dates.append(str(grp["date"].iloc[0]))
        vecs.append(mean / norm)
        texts.append("\n\n".join(grp["text"].tolist())[:4000])
    X = np.array(vecs) if vecs else np.zeros((0, 0))
    return ids, dates, X, texts


# --------------------------------------------------------------------------- #
# Deterministic k-means (seeded k-means++)
# --------------------------------------------------------------------------- #
def _kmeans(X: np.ndarray, k: int, seed: int = 0, iters: int = 60) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(X)
    centers = [X[int(rng.integers(n))]]
    for _ in range(k - 1):
        d2 = np.min(
            ((X[:, None, :] - np.array(centers)[None, :, :]) ** 2).sum(-1), axis=1
        )
        total = d2.sum()
        probs = d2 / total if total > 0 else np.full(n, 1.0 / n)
        centers.append(X[int(rng.choice(n, p=probs))])
    C = np.array(centers)
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        dists = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        labels = dists.argmin(axis=1)
        newC = np.array([
            X[labels == j].mean(axis=0) if (labels == j).any() else C[j]
            for j in range(k)
        ])
        if np.allclose(newC, C):
            break
        C = newC
    return labels


def default_k(n_entries: int) -> int:
    """A gentle default: grows slowly with corpus size, capped for readability."""
    return max(2, min(10, round((n_entries / 4) ** 0.5) + 1))


# --------------------------------------------------------------------------- #
# Labeling: distinctive words per cluster vs. the whole corpus
# --------------------------------------------------------------------------- #
def _word_counts(texts: list[str]) -> pd.Series:
    words = []
    for t in texts:
        words.extend(
            w for w in (m.group(0).lower() for m in _TOKEN_RE.finditer(t))
            if w not in _STOPWORDS
        )
    return pd.Series(words).value_counts() if words else pd.Series(dtype=int)


def _distinctive_terms(cluster_texts: list[str], corpus: pd.Series,
                       n: int = 4) -> list[str]:
    local = _word_counts(cluster_texts)
    if local.empty:
        return []
    local = local[local >= 2] if (local >= 2).any() else local
    l_tot = local.sum() or 1
    c_tot = corpus.sum() or 1
    score = (local / l_tot) * np.log(
        (local / l_tot) / (corpus.reindex(local.index).fillna(0) / c_tot + 1e-9)
        + 1e-9
    )
    return score.sort_values(ascending=False).head(n).index.tolist()


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def cluster_entries(ids: list[str], dates: list[str], X: np.ndarray,
                    texts: list[str], k: int | None = None,
                    seed: int = 0) -> tuple[list[Theme], pd.DataFrame]:
    """Cluster pre-built entry vectors. Returns (themes, assignments) where
    assignments has one row per entry: entry_id, date, year, theme_id, label."""
    n = len(ids)
    if n == 0 or X.size == 0:
        return [], pd.DataFrame()
    k = min(k or default_k(n), n)
    labels = _kmeans(X, k, seed=seed)
    corpus = _word_counts(texts)

    themes: list[Theme] = []
    for j in range(k):
        members = [i for i in range(n) if labels[i] == j]
        if not members:
            continue
        terms = _distinctive_terms([texts[i] for i in members], corpus)
        themes.append(Theme(
            theme_id=j,
            label=", ".join(terms[:3]) if terms else f"theme {j + 1}",
            terms=terms,
            size=len(members),
            entry_ids=[ids[i] for i in members],
        ))
    themes.sort(key=lambda t: -t.size)

    # Similar clusters can land on the same distinctive words; disambiguate so
    # the theme river never silently merges two themes under one label.
    seen: dict[str, int] = {}
    for t in themes:
        seen[t.label] = seen.get(t.label, 0) + 1
        if seen[t.label] > 1:
            t.label = f"{t.label} ({seen[t.label]})"

    label_of = {t.theme_id: t.label for t in themes}
    assign = pd.DataFrame({
        "entry_id": ids,
        "date": dates,
        "year": [str(d)[:4] for d in dates],
        "theme_id": labels,
    })
    assign["label"] = assign["theme_id"].map(label_of)
    return themes, assign


def discover(tbl=None, k: int | None = None,
             seed: int = 0) -> tuple[list[Theme], pd.DataFrame]:
    """Load the corpus and cluster it. The convenience entry point for the CLI
    and the web UI."""
    tbl = tbl or store.open_or_create()
    try:
        df_all = store.table_to_df(tbl)
    except Exception:
        return [], pd.DataFrame()
    ids, dates, X, texts = entry_matrix(df_all)
    return cluster_entries(ids, dates, X, texts, k=k, seed=seed)


def theme_year_share(assign: pd.DataFrame) -> pd.DataFrame:
    """Year × theme share-of-entries (rows sum to 1) — the theme-river frame."""
    if assign.empty:
        return pd.DataFrame()
    counts = pd.crosstab(assign["year"], assign["label"])
    return counts.div(counts.sum(axis=1), axis=0).round(3)


def name_themes(themes: list[Theme], texts_by_id: dict[str, str],
                model: str | None = None) -> list[Theme]:
    """Optionally ask the local model for a friendlier 2-4 word title per theme.
    Grounded in the distinctive terms + two short excerpts; falls back to the
    term label on any failure. Mutates and returns ``themes``."""
    from .llm import chat  # imported lazily so themes stays usable without Ollama

    for t in themes:
        excerpts = [
            texts_by_id.get(eid, "")[:300] for eid in t.entry_ids[:2]
        ]
        prompt = (
            f"Distinctive words: {', '.join(t.terms)}\n\n"
            "Excerpts from this group of journal entries:\n"
            + "\n---\n".join(e for e in excerpts if e)
            + "\n\nGive a 2-4 word title for this recurring theme in the "
            "journal. Respond with ONLY the title, no quotes or punctuation."
        )
        try:
            title = chat(prompt, model=model).strip().strip('"').strip(".")
            if 0 < len(title) <= 60:
                t.label = title
        except Exception:
            pass
    return themes
