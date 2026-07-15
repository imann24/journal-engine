"""Grounded period reflections ("month / year in review").

Samples a date range's entries, asks the local Ollama model for a short
narrative that cites entry dates, and caches the result in SQLite (alongside
the conversations DB, inside the gitignored LanceDB folder) so recomposing a
period is instant. The cache key covers the period, the model, and the
entries' content hashes, so editing or re-ingesting anything in the range
invalidates the digest automatically.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import config
from .llm import chat

DIGEST_SYSTEM = (
    "You write short reflective digests of a person's own journal entries, "
    "addressed to them as 'you'. Ground every claim strictly in the excerpts "
    "provided — never invent events, people, or feelings. Cite entry dates in "
    "parentheses, e.g. (2019-07-14). Shape: a brief narrative of the period "
    "(what happened, what occupied their mind), then a short paragraph on any "
    "shifts or throughlines actually visible in the excerpts, and end with one "
    "gentle, concrete question worth sitting with. Warm but honest — no "
    "diagnoses, no advice lists, no flattery. Keep it under ~350 words."
)

MAX_EXCERPTS = 14
EXCERPT_CHARS = 700


@dataclass
class Digest:
    text: str
    cached: bool
    n_entries: int
    n_sampled: int
    model: str


# --------------------------------------------------------------------------- #
# Cache (SQLite, lives in the gitignored DB folder — never committed)
# --------------------------------------------------------------------------- #
def _conn() -> sqlite3.Connection:
    db_dir = Path(config.DB_PATH)
    db_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_dir / "insights.db"))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS digests ("
        " cache_key TEXT PRIMARY KEY, date_from TEXT, date_to TEXT,"
        " model TEXT, text TEXT NOT NULL, created_at TEXT NOT NULL)"
    )
    return conn


def cache_key(entries: pd.DataFrame, date_from: str, date_to: str,
              model: str) -> str:
    h = hashlib.sha256(f"{date_from}|{date_to}|{model}".encode("utf-8"))
    hashes = entries.get("content_hash", entries["entry_id"])
    for eid, ch in sorted(zip(entries["entry_id"], hashes)):
        h.update(f"{eid}:{ch}".encode("utf-8"))
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Sampling + prompt
# --------------------------------------------------------------------------- #
def sample_entries(entries: pd.DataFrame, max_n: int = MAX_EXCERPTS) -> pd.DataFrame:
    """Evenly spaced sample across the period (always keeps first and last),
    oldest first, so long ranges still fit in one prompt."""
    d = entries.sort_values("date").reset_index(drop=True)
    n = len(d)
    if n <= max_n:
        return d
    idx = sorted({round(i * (n - 1) / (max_n - 1)) for i in range(max_n)})
    return d.iloc[idx].reset_index(drop=True)


def _context_block(entries: pd.DataFrame) -> str:
    """Compact stats header so the model can mention scale honestly."""
    parts = [f"{len(entries)} entries in this period"]
    moody = entries[entries["mood"] > 0] if "mood" in entries else pd.DataFrame()
    if len(moody) >= 3:
        parts.append(f"mean mood {moody['mood'].mean():.1f}/5")
    for col in ("topics", "people"):
        if col in entries:
            tokens = (
                entries[col].dropna().str.split(",").explode().str.strip()
            )
            tokens = tokens[tokens != ""]
            if not tokens.empty:
                top = ", ".join(tokens.value_counts().head(5).index)
                parts.append(f"recurring {col}: {top}")
    return "; ".join(parts)


def compose(entries: pd.DataFrame, texts: pd.Series | dict | None = None,
            date_from: str = "", date_to: str = "", model: str | None = None,
            force: bool = False) -> Digest:
    """Compose (or fetch from cache) the reflection for a pre-filtered entry
    frame. ``texts`` optionally maps entry_id -> full text (falls back to the
    frame's first-chunk text)."""
    model = model or config.CHAT_MODEL
    if entries.empty:
        return Digest("No entries in this period.", False, 0, 0, model)

    key = cache_key(entries, date_from, date_to, model)
    conn = _conn()
    try:
        if not force:
            row = conn.execute(
                "SELECT text FROM digests WHERE cache_key = ?", (key,)
            ).fetchone()
            if row:
                return Digest(row[0], True, len(entries), 0, model)

        sampled = sample_entries(entries)
        blocks = []
        for _, r in sampled.iterrows():
            body = None
            if texts is not None:
                try:
                    body = texts[r["entry_id"]]
                except (KeyError, IndexError):
                    body = None
            body = body if body else r.get("text", "")
            blocks.append(f"[{r['date']}] {str(body)[:EXCERPT_CHARS]}")

        prompt = (
            f"PERIOD: {date_from or sampled['date'].iloc[0]} to "
            f"{date_to or sampled['date'].iloc[-1]}\n"
            f"OVERVIEW: {_context_block(entries)}\n\n"
            "JOURNAL EXCERPTS (a sample across the period):\n"
            + "\n\n".join(blocks)
            + "\n\nWrite the reflection:"
        )
        text = chat(prompt, system=DIGEST_SYSTEM, model=model)

        conn.execute(
            "INSERT OR REPLACE INTO digests VALUES (?, ?, ?, ?, ?, ?)",
            (key, date_from, date_to, model, text, datetime.now().isoformat()),
        )
        conn.commit()
        return Digest(text, False, len(entries), len(sampled), model)
    finally:
        conn.close()
