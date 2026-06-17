"""RAG question answering. Retrieves with hybrid_search, grounds the model in the
excerpts, cites entry dates, and refuses gracefully when coverage is thin.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .llm import chat
from .search import hybrid_search

RAG_SYSTEM = (
    "You answer questions strictly from the user's own journal excerpts provided "
    "below. Use only what is present — do not invent events, names, or dates. "
    "Attribute claims to their entry date in parentheses, e.g. (2019-07-14). If "
    "the excerpts do not cover the question, say so plainly rather than guessing."
)


@dataclass
class Answer:
    text: str
    cited_dates: list[str] = field(default_factory=list)
    excerpts: list[dict] = field(default_factory=list)


def ask(
    question: str,
    k: int = 8,
    date_from: str | None = None,
    date_to: str | None = None,
    tbl=None,
) -> Answer:
    hits = hybrid_search(question, k=k, date_from=date_from, date_to=date_to, tbl=tbl)
    if not hits:
        return Answer(
            "No relevant entries found for that query"
            + (" in the given date range." if (date_from or date_to) else "."),
            [], [],
        )

    ordered = sorted(hits, key=lambda r: r["date"])
    context = "\n\n".join(f"[{h['date']}] {h['text']}" for h in ordered)
    prompt = (
        f"JOURNAL EXCERPTS:\n{context}\n\n"
        f"QUESTION: {question}\n\nAnswer, citing entry dates:"
    )
    text = chat(prompt, system=RAG_SYSTEM)
    cited = sorted({h["date"] for h in hits})
    return Answer(text, cited, ordered)
