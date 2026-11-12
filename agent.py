"""Recall's agent: OpenAI Responses API + exactly one tool (web search) + SQLite memory."""
from __future__ import annotations

import os
from typing import Iterator
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from dotenv import load_dotenv
from openai import BadRequestError, OpenAI

import db

load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
HISTORY_MESSAGES = int(os.getenv("RECALL_HISTORY_MESSAGES", "24"))
QUESTION_LOG_MAX = 60

_client: OpenAI | None = None


def client() -> OpenAI:
    global _client
    if _client is None:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set. Add it to your .env file.")
        _client = OpenAI()
    return _client


SYSTEM_PROMPT = """\
You are Recall, a fast, accurate research assistant with persistent memory and one tool: web search.

WHEN TO SEARCH
- Search for anything that may have changed or that you can't be sure of: news, prices, scores, \
weather, releases, software versions, laws, office-holders, "latest/current/today", people or \
companies you don't know well, and any claim where an error would matter.
- Do NOT search for stable knowledge: math, definitions, grammar, coding help, well-established \
history or science, writing/rewriting tasks, or questions about this conversation itself.
- Prefer one precise search; use more only if the first is insufficient. Favor primary and \
reputable sources (official sites, major news agencies, peer-reviewed or government sources) over \
blogs and forums.

ANSWER STYLE
- Lead with the direct answer, then give only the supporting detail needed. Be concise.
- Summarize in your own words; never paste long passages from pages. Quote at most a short phrase.
- Give absolute dates for time-sensitive facts ("as of 12 Nov 2026"). Resolve "today", "yesterday" \
and similar against the current UTC time below.
- If sources disagree or evidence is thin, say so. If you don't know and search didn't help, say that.
- Never invent URLs, titles, quotes or sources. Only cite pages the search tool actually returned.

MEMORY
- You are given the full list of questions the user asked in this conversation with UTC timestamps, \
plus related questions from earlier conversations. Use them to answer "what did I ask", "when did \
I ask", and to resolve follow-ups such as "and what about its price?" or "compare it with the first one".
- Treat follow-ups as continuing the prior topic. When you search for a follow-up, make the query \
self-contained (include the entity names from earlier turns).
- Report timestamps in UTC unless the user asks otherwise.
"""


def _fmt_ts(ts: str) -> str:
    return ts.replace("T", " ").replace("Z", " UTC")


def build_instructions(cid: str, user_text: str) -> str:
    parts = [SYSTEM_PROMPT, f"CURRENT UTC TIME: {_fmt_ts(db.utcnow())}"]

    questions = db.get_user_questions(cid)
    if questions:
        shown = questions[-QUESTION_LOG_MAX:]
        offset = len(questions) - len(shown)
        lines = [f"{offset + i + 1}. [{_fmt_ts(q['created_at'])}] {q['content']}" for i, q in enumerate(shown)]
        parts.append("QUESTIONS ASKED IN THIS CONVERSATION (oldest first):\n" + "\n".join(lines))

    cid_hits = db.search_past_questions(user_text, exclude_cid=cid)
    if cid_hits:
        lines = [
            f"- [{_fmt_ts(h['asked_at'])}] Q: {h['question']}"
            + (f"\n  A (truncated): {h['answer_snippet']}" if h["answer_snippet"] else "")
            for h in cid_hits
        ]
        parts.append(
            "POSSIBLY RELATED QUESTIONS FROM EARLIER CONVERSATIONS (context only; may be outdated, "
            "re-check time-sensitive facts):\n" + "\n".join(lines)
        )
    return "\n\n".join(parts)


def build_input(cid: str) -> list[dict]:
    msgs = db.get_messages(cid, limit=HISTORY_MESSAGES)
    return [{"role": m["role"], "content": m["content"]} for m in msgs]


# ── source extraction ────────────────────────────────────────────────────────
def _clean_url(url: str) -> str:
    p = urlparse(url)
    q = [(k, v) for k, v in parse_qsl(p.query) if not k.lower().startswith("utm_")]
    return urlunparse(p._replace(query=urlencode(q)))


def _domain(url: str) -> str:
    host = urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def extract_sources_and_queries(response) -> tuple[list[dict], list[str]]:
    """Pull real citations and search queries out of a completed response.

    Sources come ONLY from the tool's url_citation annotations, so they cannot be fabricated.
    """
    sources: list[dict] = []
    queries: list[str] = []
    seen: set[str] = set()
    for item in getattr(response, "output", None) or []:
        itype = getattr(item, "type", "")
        if itype == "web_search_call":
            q = getattr(getattr(item, "action", None), "query", None)
            if q and q not in queries:
                queries.append(q)
        elif itype == "message":
            for part in getattr(item, "content", None) or []:
                for ann in getattr(part, "annotations", None) or []:
                    if getattr(ann, "type", "") != "url_citation":
                        continue
                    url = _clean_url(getattr(ann, "url", "") or "")
                    if not url or url in seen:
                        continue
                    seen.add(url)
                    sources.append(
                        {"title": (getattr(ann, "title", "") or _domain(url)).strip(),
                         "url": url, "domain": _domain(url)}
                    )
    return sources, queries


# ── main entry point ─────────────────────────────────────────────────────────
def _stream(cid: str, user_text: str, tool_type: str):
    return client().responses.create(
        model=MODEL,
        instructions=build_instructions(cid, user_text),
        input=build_input(cid),
        tools=[{"type": tool_type, "search_context_size": "medium"}],  # the ONLY tool
        tool_choice="auto",   # model decides whether a search is needed
        max_tool_calls=3,
        store=False,          # memory lives in our SQLite, not on OpenAI's side
        stream=True,
    )


def stream_answer(cid: str, user_text: str) -> Iterator[dict]:
    """Persist the question, stream the answer, persist the answer.

    Yields dicts: {"type": "status"|"delta"|"done", ...}
    """
    db.add_message(cid, "user", user_text)  # timestamp = moment the question was asked

    try:
        stream = _stream(cid, user_text, "web_search")
    except BadRequestError:
        stream = _stream(cid, user_text, "web_search_preview")  # older API naming

    final = None
    buf: list[str] = []
    for ev in stream:
        t = getattr(ev, "type", "")
        if t in ("response.web_search_call.in_progress", "response.web_search_call.searching"):
            yield {"type": "status", "text": "Searching the web…"}
        elif t == "response.web_search_call.completed":
            yield {"type": "status", "text": "Reading results…"}
        elif t == "response.output_text.delta":
            buf.append(ev.delta)
            yield {"type": "delta", "text": ev.delta}
        elif t == "response.completed":
            final = ev.response
        elif t in ("response.failed", "response.incomplete"):
            err = getattr(getattr(ev.response, "error", None), "message", None)
            raise RuntimeError(err or "The model did not finish its answer.")
        elif t == "error":
            raise RuntimeError(getattr(ev, "message", "Unknown streaming error"))

    text = "".join(buf).strip() or "I couldn't produce an answer. Please try again."
    sources, queries = extract_sources_and_queries(final) if final else ([], [])
    used_search = bool(queries or sources)

    db.add_message(cid, "assistant", text, used_search=used_search,
                   sources=sources, search_queries=queries)
    yield {"type": "done", "text": text, "sources": sources,
           "used_search": used_search, "queries": queries}
