import importlib
import re

import pytest


@pytest.fixture()
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("RECALL_DB_PATH", str(tmp_path / "t.db"))
    import db as mod
    importlib.reload(mod)
    mod.init_db()
    return mod


def test_messages_roundtrip_with_utc_and_sources(db):
    cid = db.create_conversation("t")
    db.add_message(cid, "user", "Who won the 2022 World Cup?")
    db.add_message(cid, "assistant", "Argentina.", used_search=True,
                   sources=[{"title": "FIFA", "url": "https://fifa.com", "domain": "fifa.com"}],
                   search_queries=["2022 world cup winner"])
    msgs = db.get_messages(cid)
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", msgs[0]["created_at"])
    assert msgs[1]["sources"][0]["domain"] == "fifa.com"
    assert msgs[1]["used_search"] is True


def test_question_log_and_limit(db):
    cid = db.create_conversation("t")
    for i in range(5):
        db.add_message(cid, "user", f"q{i}")
        db.add_message(cid, "assistant", f"a{i}")
    assert [q["content"] for q in db.get_user_questions(cid)] == [f"q{i}" for i in range(5)]
    assert len(db.get_messages(cid, limit=3)) == 3


def test_cross_conversation_recall(db):
    a = db.create_conversation("a")
    db.add_message(a, "user", "What is the capital of Australia?")
    db.add_message(a, "assistant", "Canberra.")
    b = db.create_conversation("b")
    hits = db.search_past_questions("Tell me more about Australia's capital", exclude_cid=b)
    assert hits and hits[0]["answer_snippet"] == "Canberra."
    assert db.search_past_questions("Australia", exclude_cid=a) == []


def test_delete_cascades(db):
    cid = db.create_conversation("t")
    db.add_message(cid, "user", "hi")
    db.delete_conversation(cid)
    assert db.get_messages(cid) == []
