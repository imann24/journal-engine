"""Persistence layer for queries/conversations.
Allows saving chat sessions, metadata, and excerpts to a local SQLite database in the LanceDB folder.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from . import config


def get_db_path() -> Path:
    db_dir = Path(config.DB_PATH)
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / "conversations.db"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(get_db_path()))
    # Enable foreign keys for cascade deletes
    conn.execute("PRAGMA foreign_keys = ON;")
    # Return rows as dicts for convenience
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initialize the database tables if they do not exist."""
    conn = get_connection()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                date_from TEXT,
                date_to TEXT,
                k INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                excerpts_json TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            );
        """)
        conn.commit()
    finally:
        conn.close()


def list_conversations() -> list[dict[str, Any]]:
    """List all saved conversations, sorted by last update time (descending)."""
    conn = get_connection()
    try:
        cursor = conn.execute(
            "SELECT id, title, date_from, date_to, k, created_at, updated_at "
            "FROM conversations ORDER BY updated_at DESC"
        )
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def get_conversation(conversation_id: str) -> dict[str, Any] | None:
    """Retrieve metadata and message history of a specific conversation."""
    conn = get_connection()
    try:
        cursor = conn.execute(
            "SELECT id, title, date_from, date_to, k, created_at, updated_at "
            "FROM conversations WHERE id = ?",
            (conversation_id,)
        )
        row = cursor.fetchone()
        if not row:
            return None

        conv = dict(row)

        cursor_msgs = conn.execute(
            "SELECT role, content, excerpts_json, created_at "
            "FROM messages WHERE conversation_id = ? ORDER BY id ASC",
            (conversation_id,)
        )
        messages = []
        for msg_row in cursor_msgs.fetchall():
            msg = dict(msg_row)
            if msg["excerpts_json"]:
                msg["excerpts"] = json.loads(msg["excerpts_json"])
            else:
                msg["excerpts"] = None
            messages.append(msg)

        conv["messages"] = messages
        return conv
    finally:
        conn.close()


def save_conversation(
    conversation_id: str,
    title: str,
    date_from: str | None,
    date_to: str | None,
    k: int,
    messages: list[dict[str, Any]]
) -> None:
    """Create or update a conversation and overwrite its message history."""
    conn = get_connection()
    try:
        now = datetime.now().isoformat()

        # Check if conversation already exists to keep its created_at timestamp
        cursor = conn.execute("SELECT created_at FROM conversations WHERE id = ?", (conversation_id,))
        row = cursor.fetchone()

        if row:
            created_at = row["created_at"]
            conn.execute(
                "UPDATE conversations SET title = ?, date_from = ?, date_to = ?, k = ?, updated_at = ? "
                "WHERE id = ?",
                (title, date_from, date_to, k, now, conversation_id)
            )
        else:
            created_at = now
            conn.execute(
                "INSERT INTO conversations (id, title, date_from, date_to, k, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (conversation_id, title, date_from, date_to, k, created_at, now)
            )

        # Clear existing messages and insert new list
        conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))

        for msg in messages:
            excerpts_json = None
            if "excerpts" in msg and msg["excerpts"] is not None:
                excerpts_json = json.dumps(msg["excerpts"])

            conn.execute(
                "INSERT INTO messages (conversation_id, role, content, excerpts_json, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (conversation_id, msg["role"], msg["content"], excerpts_json, now)
            )

        conn.commit()
    finally:
        conn.close()


def delete_conversation(conversation_id: str) -> None:
    """Delete a conversation (cascading to its messages)."""
    conn = get_connection()
    try:
        conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        conn.commit()
    finally:
        conn.close()


def rename_conversation(conversation_id: str, new_title: str) -> None:
    """Rename a conversation's title."""
    conn = get_connection()
    try:
        now = datetime.now().isoformat()
        conn.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
            (new_title, now, conversation_id)
        )
        conn.commit()
    finally:
        conn.close()
