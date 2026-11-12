# Recall — web-search chat agent with persistent memory

A ChatGPT-style assistant built with **Streamlit**, the **OpenAI Responses API**, and **SQLite**.
It has exactly one tool, **web search**, and remembers every question you ask along with the UTC time you asked it.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate     # Python 3.11+
pip install -r requirements.txt
cp .env.example .env                                  # add your OPENAI_API_KEY
streamlit run app.py
python -m pytest -q tests                             # optional: unit tests (no API key needed)
```

## How each requirement is met

| Requirement | Implementation |
|---|---|
| Fast answers | Token streaming (`stream=True`), `gpt-4.1-mini` default, `search_context_size="medium"`, max 3 searches per turn, search skipped when not needed |
| Accurate answers | System prompt prioritizes primary/reputable sources, requires stating disagreement or uncertainty, absolute dates, and forbids invented facts or URLs |
| Searches only when needed | `tools=[web_search]` with `tool_choice="auto"`; prompt lists what warrants a search (news, prices, versions, "latest") and what doesn't (math, definitions, code, chat-recall) |
| No fabricated sources | The source list shown in the UI is built **only** from the tool's `url_citation` annotations (`agent.extract_sources_and_queries`), deduplicated, with tracking params stripped |
| Remembers what was asked and when | Every user message is saved with a UTC timestamp *before* the model is called |
| Conversation context | Last 24 messages are replayed each turn, plus a numbered, timestamped log of **all** questions in the conversation, so "what was my first question?" works even in long chats |
| Cross-conversation memory | Keyword-scored lookup of related past questions (with answer snippets and dates) is injected as context, flagged "may be outdated" |
| Persistence | SQLite (`recall.db`): `conversations` and `messages` (role, content, `created_at` UTC, `used_search`, `sources` JSON, `search_queries` JSON) |

## Architecture

```
app.py     Streamlit UI: sidebar history, chat, streaming, source chips, timestamps
agent.py   Prompt + memory assembly, Responses API call, stream handling, citation extraction
db.py      SQLite schema and queries (all UTC)
tests/     DB and citation-extraction tests
```

Flow per turn: save question (UTC) → build instructions (system prompt + current UTC time + question log + related past questions) → stream the model, which decides whether to search → extract citations → save answer with sources → re-render from the database.

## Design decisions

- **`store=False`**: memory lives in your SQLite file, not on OpenAI's servers, so the app owns its own history.
- **Question log in instructions, not just chat history**: the replay window is bounded for speed, while the log stays small and complete.
- **Sources from annotations only**: the model's prose can link anywhere, but the "sources" row reflects what the search tool actually returned.
- **Errors keep your question**: if the API fails, the question is already saved and you can resend it.

## Demo script (shows every requirement)

1. "What are the top news headlines today?" → searches, shows source chips and badge.
2. "Which of those is about technology?" → follow-up using context.
3. "Explain recursion in one paragraph" → badge shows *From memory* (no search).
4. "What was my first question, and when did I ask it?" → answers from the timestamped log.
5. Click **New chat**, ask "Remind me what I asked about news headlines earlier" → cross-conversation recall.

## Possible extensions

Full-text search (FTS5) or embeddings for past-question retrieval, per-user accounts, export to Markdown, domain allow-listing for sources.
