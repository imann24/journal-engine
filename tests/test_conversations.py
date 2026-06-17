"""Tests for query/conversation persistence."""

from __future__ import annotations

import tempfile
from pathlib import Path
import pytest

from journal import config, conversations


@pytest.fixture
def temp_db(monkeypatch):
    """Create a temporary directory for DB storage and point config.DB_PATH to it."""
    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setattr(config, "DB_PATH", tmpdir)
        conversations.init_db()
        yield Path(tmpdir)


def test_init_db(temp_db):
    """Ensure the db file is created and initialized."""
    db_file = temp_db / "conversations.db"
    assert db_file.exists()


def test_save_and_retrieve_conversation(temp_db):
    """Test saving a new conversation and retrieving it."""
    conv_id = "test-conv-1"
    messages = [
        {"role": "user", "content": "hello", "excerpts": None},
        {"role": "assistant", "content": "hi there", "excerpts": [{"date": "2026-06-17", "text": "excerpt"}]},
    ]

    conversations.save_conversation(
        conversation_id=conv_id,
        title="Test Title",
        date_from="2026-01-01",
        date_to="2026-12-31",
        k=5,
        messages=messages,
    )

    # List conversations
    convs = conversations.list_conversations()
    assert len(convs) == 1
    assert convs[0]["id"] == conv_id
    assert convs[0]["title"] == "Test Title"

    # Get conversation
    conv = conversations.get_conversation(conv_id)
    assert conv is not None
    assert conv["title"] == "Test Title"
    assert conv["date_from"] == "2026-01-01"
    assert conv["date_to"] == "2026-12-31"
    assert conv["k"] == 5
    assert len(conv["messages"]) == 2
    assert conv["messages"][0]["role"] == "user"
    assert conv["messages"][0]["content"] == "hello"
    assert conv["messages"][0]["excerpts"] is None
    assert conv["messages"][1]["role"] == "assistant"
    assert conv["messages"][1]["content"] == "hi there"
    assert conv["messages"][1]["excerpts"] == [{"date": "2026-06-17", "text": "excerpt"}]


def test_update_existing_conversation(temp_db):
    """Test that saving with an existing id updates the conversation."""
    conv_id = "test-conv-1"

    conversations.save_conversation(
        conversation_id=conv_id,
        title="Initial Title",
        date_from=None,
        date_to=None,
        k=8,
        messages=[{"role": "user", "content": "query 1", "excerpts": None}],
    )

    # Update it
    conversations.save_conversation(
        conversation_id=conv_id,
        title="Updated Title",
        date_from="2026-06-01",
        date_to=None,
        k=10,
        messages=[
            {"role": "user", "content": "query 1", "excerpts": None},
            {"role": "assistant", "content": "response 1", "excerpts": None},
            {"role": "user", "content": "query 2", "excerpts": None},
        ],
    )

    conv = conversations.get_conversation(conv_id)
    assert conv is not None
    assert conv["title"] == "Updated Title"
    assert conv["date_from"] == "2026-06-01"
    assert conv["k"] == 10
    assert len(conv["messages"]) == 3
    assert conv["messages"][2]["content"] == "query 2"


def test_delete_conversation(temp_db):
    """Test deleting a conversation removes it and its messages."""
    conv_id = "test-conv-1"
    conversations.save_conversation(
        conversation_id=conv_id,
        title="To Delete",
        date_from=None,
        date_to=None,
        k=5,
        messages=[{"role": "user", "content": "msg", "excerpts": None}],
    )

    assert len(conversations.list_conversations()) == 1
    conversations.delete_conversation(conv_id)
    assert len(conversations.list_conversations()) == 0
    assert conversations.get_conversation(conv_id) is None


def test_rename_conversation(temp_db):
    """Test renaming a conversation's title."""
    conv_id = "test-conv-1"
    conversations.save_conversation(
        conversation_id=conv_id,
        title="Old Title",
        date_from=None,
        date_to=None,
        k=5,
        messages=[{"role": "user", "content": "msg", "excerpts": None}],
    )

    conversations.rename_conversation(conv_id, "New Title")
    conv = conversations.get_conversation(conv_id)
    assert conv is not None
    assert conv["title"] == "New Title"
