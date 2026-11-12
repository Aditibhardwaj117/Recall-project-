"""SQLite persistence layer for Recall.

Every timestamp is stored as an ISO-8601 UTC string ending in "Z".
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

DB_PATH = Path(os.getenv("RECALL_DB_PATH", "recall.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id         TEXT PRIMARY KEY,
    title      TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content         TEXT NOT NULL,
    created_at      TEXT NOT NULL,              -- UTC, e.g. 2026-11-12T10:42:03Z
    used_search     INTEGER NOT NULL DEFAULT 0, -- 1 if the web search tool ran
    sources         TEXT NOT NULL DEFAULT '[]', -- JSON: [{"title","url","domain"}]
    search_queries  TEXT NOT NULL DEFAULT '[]'  -- JSON: ["query", ...]
);

CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, id);
CREATE INDEX IF NOT EXISTS idx_conv_updated  ON conversations(updated_at DESC);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)


# ── conversations ────────────────────────────────────────────────────────────
def create_conversation(title: str = "New chat") -> str:
    cid, now = uuid.uuid4().hex, utcnow()
    with connect() as conn:
        conn.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?,?,?,?)",
            (cid, title, now, now),
        )
    return cid


def list_conversations(limit: int = 50) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM conversations ORDER BY updated_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_conversation(cid: str) -> dict | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM conversations WHERE id = ?", (cid,)).fetchone()
    return dict(row) if row else None


def rename_conversation(cid: str, title: str) -> None:
    with connect() as conn:
        conn.execute("UPDATE conversations SET title = ? WHERE id = ?", (title, cid))


def delete_conversation(cid: str) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM conversations WHERE id = ?", (cid,))


# ── messages ─────────────────────────────────────────────────────────────────
def add_message(
    cid: str,
    role: str,
    content: str,
    *,
    used_search: bool = False,
    sources: list[dict] | None = None,
    search_queries: list[str] | None = None,
    created_at: str | None = None,
) -> int:
    now = created_at or utcnow()
    with connect() as conn:
        cur = conn.execute(
            """INSERT INTO messages
               (conversation_id, role, content, created_at, used_search, sources, search_queries)
               VALUES (?,?,?,?,?,?,?)""",
            (
                cid, role, content, now, int(used_search),
                json.dumps(sources or []), json.dumps(search_queries or []),
            ),
        )
        conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, cid))
        return int(cur.lastrowid)


def _row_to_message(r: sqlite3.Row) -> dict:
    d = dict(r)
    d["sources"] = json.loads(d["sources"] or "[]")
    d["search_queries"] = json.loads(d["search_queries"] or "[]")
    d["used_search"] = bool(d["used_search"])
    return d


def get_messages(cid: str, limit: int | None = None) -> list[dict]:
    """Messages in chronological order. With `limit`, returns the most recent N."""
    with connect() as conn:
        if limit:
            rows = conn.execute(
                "SELECT * FROM (SELECT * FROM messages WHERE conversation_id = ? "
                "ORDER BY id DESC LIMIT ?) ORDER BY id ASC",
                (cid, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id ASC", (cid,)
            ).fetchall()
    return [_row_to_message(r) for r in rows]


def get_user_questions(cid: str) -> list[dict]:
    """Every question the user asked in this conversation, with its UTC timestamp."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT content, created_at FROM messages "
            "WHERE conversation_id = ? AND role = 'user' ORDER BY id ASC",
            (cid,),
        ).fetchall()
    return [dict(r) for r in rows]


# ── cross-conversation recall ────────────────────────────────────────────────
_STOP = set(
    "the a an and or but for with about from that this what when where which who whom how why "
    "are was were is be been being have has had do does did can could would should will "
    "you your me my our their his her its into onto over under than then there here "
    "tell give show please just also more some any".split()
)


def _keywords(text: str, limit: int = 6) -> list[str]:
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-']{2,}", text.lower())
    seen, out = set(), []
    for w in words:
        if w not in _STOP and w not in seen:
            seen.add(w)
            out.append(w)
    return out[:limit]


def search_past_questions(
    text: str, exclude_cid: str | None = None, limit: int = 4
) -> list[dict]:
    """Find earlier questions (in other conversations) related to `text`.

    Lightweight keyword scoring. Returns question, UTC time, and a short answer snippet.
    """
    kws = _keywords(text)
    if not kws:
        return []
    clause = " + ".join(["(LOWER(content) LIKE ?)"] * len(kws))
    params: list = [f"%{k}%" for k in kws]
    sql = (
        f"SELECT id, conversation_id, content, created_at, ({clause}) AS score "
        "FROM messages WHERE role = 'user' "
    )
    if exclude_cid:
        sql += "AND conversation_id != ? "
        params.append(exclude_cid)
    sql += "AND score > 0 ORDER BY score DESC, id DESC LIMIT ?"
    params.append(limit)

    results = []
    with connect() as conn:
        for r in conn.execute(sql, params).fetchall():
            ans = conn.execute(
                "SELECT content FROM messages WHERE conversation_id = ? AND id > ? "
                "AND role = 'assistant' ORDER BY id ASC LIMIT 1",
                (r["conversation_id"], r["id"]),
            ).fetchone()
            results.append(
                {
                    "question": r["content"],
                    "asked_at": r["created_at"],
                    "answer_snippet": (ans["content"][:300] if ans else ""),
                }
            )
    return results
