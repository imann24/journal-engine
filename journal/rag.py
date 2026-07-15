"""RAG question answering. Retrieves with hybrid_search, grounds the model in the
excerpts, cites entry dates, and refuses gracefully when coverage is thin.

Two seamlessness upgrades over plain one-shot RAG:

* **Multi-turn.** ``ask`` accepts the prior chat ``history`` so follow-ups keep
  their context; short follow-ups also borrow the previous question for
  retrieval (so "what about later that year?" still finds the right entries).
* **Auto date filter.** With ``auto_dates=True`` and no explicit range, years
  mentioned in the question ("in 2019", "between 2019 and 2021", "last year")
  become a date_int prefilter automatically. Conservative by design: only
  explicit years and last/this year, never guessed seasons or months.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from .llm import chat_messages
from .search import hybrid_search

RAG_SYSTEM = (
    "You answer questions strictly from the user's own journal excerpts provided "
    "below. Use only what is present — do not invent events, names, or dates. "
    "Attribute claims to their entry date in parentheses, e.g. (2019-07-14). If "
    "the excerpts do not cover the question, say so plainly rather than guessing."
)

_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_HISTORY_TURNS = 8  # most recent messages carried into the model context


@dataclass
class Answer:
    text: str
    cited_dates: list[str] = field(default_factory=list)
    excerpts: list[dict] = field(default_factory=list)
    date_from: str | None = None
    date_to: str | None = None
    auto_dated: bool = False


def infer_date_range(question: str,
                     today: date | None = None) -> tuple[str | None, str | None]:
    """Explicit years (or last/this year) in the question -> (from, to) ISO
    dates spanning those calendar years; (None, None) when nothing explicit."""
    today = today or date.today()
    q = question.lower()
    if "last year" in q:
        y = today.year - 1
        return f"{y}-01-01", f"{y}-12-31"
    if "this year" in q:
        y = today.year
        return f"{y}-01-01", f"{y}-12-31"
    years = [int(m.group(0)) for m in _YEAR_RE.finditer(question)]
    years = [y for y in years if 1950 <= y <= today.year + 1]
    if years:
        return f"{min(years)}-01-01", f"{max(years)}-12-31"
    return None, None


def _retrieval_query(question: str, history: list[dict] | None) -> str:
    """Short follow-ups ("and then?", "what about her?") retrieve poorly on
    their own; borrow the previous user question for the search."""
    if history and len(question.split()) <= 5:
        prev = [m["content"] for m in history if m.get("role") == "user"]
        if prev:
            return f"{prev[-1]} {question}"
    return question


def ask(
    question: str,
    k: int = 8,
    date_from: str | None = None,
    date_to: str | None = None,
    tbl=None,
    model: str | None = None,
    history: list[dict] | None = None,
    auto_dates: bool = False,
) -> Answer:
    auto_dated = False
    if auto_dates and not (date_from or date_to):
        date_from, date_to = infer_date_range(question)
        auto_dated = bool(date_from or date_to)

    hits = hybrid_search(
        _retrieval_query(question, history),
        k=k, date_from=date_from, date_to=date_to, tbl=tbl,
    )
    if not hits:
        return Answer(
            "No relevant entries found for that query"
            + (" in the given date range." if (date_from or date_to) else "."),
            [], [], date_from, date_to, auto_dated,
        )

    ordered = sorted(hits, key=lambda r: r["date"])
    context = "\n\n".join(f"[{h['date']}] {h['text']}" for h in ordered)
    prompt = (
        f"JOURNAL EXCERPTS:\n{context}\n\n"
        f"QUESTION: {question}\n\nAnswer, citing entry dates:"
    )

    messages = [{"role": "system", "content": RAG_SYSTEM}]
    for m in (history or [])[-_HISTORY_TURNS:]:
        if m.get("role") in ("user", "assistant") and m.get("content"):
            messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": prompt})

    text = chat_messages(messages, model=model)
    cited = sorted({h["date"] for h in hits})
    return Answer(text, cited, ordered, date_from, date_to, auto_dated)
