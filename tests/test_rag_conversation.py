"""Multi-turn RAG plumbing: date inference from the question, follow-up
retrieval queries, and history reaching the model. Search + LLM are
monkeypatched — no Ollama, no LanceDB."""

from __future__ import annotations

from datetime import date

from journal import rag


def test_infer_date_range_single_year():
    assert rag.infer_date_range("what happened in 2019?") == \
        ("2019-01-01", "2019-12-31")


def test_infer_date_range_span():
    assert rag.infer_date_range("between 2019 and 2021, how was work?") == \
        ("2019-01-01", "2021-12-31")


def test_infer_date_range_relative_years():
    today = date(2026, 7, 15)
    assert rag.infer_date_range("how was last year?", today=today) == \
        ("2025-01-01", "2025-12-31")
    assert rag.infer_date_range("what has this year felt like?", today=today) == \
        ("2026-01-01", "2026-12-31")


def test_infer_date_range_ignores_non_years():
    assert rag.infer_date_range("what did I write about running?") == (None, None)
    # implausible numbers (room 2150, $1200) are not years
    assert rag.infer_date_range("we stayed in room 2150") == (None, None)


def _capture(monkeypatch):
    seen = {}

    def fake_search(query, k=8, date_from=None, date_to=None, tbl=None):
        seen["query"] = query
        seen["date_from"], seen["date_to"] = date_from, date_to
        return [{"id": "c1", "entry_id": "e1", "date": "2019-03-01",
                 "text": "an excerpt"}]

    def fake_chat_messages(messages, model=None, **kw):
        seen["messages"] = messages
        return "grounded answer"

    monkeypatch.setattr(rag, "hybrid_search", fake_search)
    monkeypatch.setattr(rag, "chat_messages", fake_chat_messages)
    return seen


def test_ask_applies_auto_dates_and_reports_them(monkeypatch):
    seen = _capture(monkeypatch)
    ans = rag.ask("how was work in 2019?", auto_dates=True)
    assert seen["date_from"] == "2019-01-01" and seen["date_to"] == "2019-12-31"
    assert ans.auto_dated and ans.date_from == "2019-01-01"


def test_explicit_dates_win_over_auto(monkeypatch):
    seen = _capture(monkeypatch)
    ans = rag.ask("how was work in 2019?", date_from="2020-01-01",
                  auto_dates=True)
    assert seen["date_from"] == "2020-01-01"
    assert not ans.auto_dated


def test_history_reaches_the_model_and_short_followups_borrow_context(monkeypatch):
    seen = _capture(monkeypatch)
    history = [
        {"role": "user", "content": "how did I feel about the move to Berlin?"},
        {"role": "assistant", "content": "you were excited (2019-03-01)"},
    ]
    rag.ask("and afterwards?", history=history)

    # short follow-up borrows the previous question for retrieval
    assert "Berlin" in seen["query"] and "afterwards" in seen["query"]

    roles = [m["role"] for m in seen["messages"]]
    assert roles == ["system", "user", "assistant", "user"]
    assert seen["messages"][1]["content"].startswith("how did I feel")
    # the final user message carries the excerpts + the new question
    assert "an excerpt" in seen["messages"][-1]["content"]
    assert "and afterwards?" in seen["messages"][-1]["content"]
