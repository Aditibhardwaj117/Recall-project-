"""Recall — a web-search chat agent with persistent memory (Streamlit UI)."""
from __future__ import annotations

import html
from datetime import datetime

import streamlit as st

import agent
import db

st.set_page_config(page_title="Recall", page_icon="🔎", layout="wide", initial_sidebar_state="expanded")
db.init_db()

ACCENT = "#0B7A75"

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600&display=swap');
html, body, [class*="st-"], .stMarkdown, button, textarea {{ font-family: 'Instrument Sans', system-ui, sans-serif; }}
#MainMenu, footer, [data-testid="stToolbar"] {{ visibility: hidden; }}
.block-container {{ max-width: 820px; padding-top: 2rem; padding-bottom: 6rem; }}

/* sidebar */
[data-testid="stSidebar"] .block-container {{ padding-top: 1.2rem; }}
.brand {{ font-size: 1.35rem; font-weight: 600; letter-spacing: -0.01em; margin-bottom: .1rem; }}
.brand span {{ color: {ACCENT}; }}
.tagline {{ font-size: .82rem; opacity: .65; margin-bottom: 1rem; }}
[data-testid="stSidebar"] .stButton > button {{
    justify-content: flex-start; text-align: left; border-radius: 10px; font-size: .9rem;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}}
[data-testid="stSidebar"] .stButton > button[kind="primary"] {{ background: {ACCENT}; border-color: {ACCENT}; color: #fff; }}

/* messages */
[data-testid="stChatMessage"] {{ border-radius: 14px; padding: .85rem 1rem; margin-bottom: .4rem; }}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {{ background: rgba(128,128,128,.10); }}
.meta {{ font-size: .75rem; opacity: .6; margin-top: .35rem; }}
.badge {{ display:inline-block; font-size:.72rem; padding:1px 9px; border-radius:99px;
          border:1px solid rgba(128,128,128,.35); margin-right:6px; }}
.badge.web {{ color:{ACCENT}; border-color:{ACCENT}; }}

/* sources */
.sources {{ display:flex; flex-wrap:wrap; gap:6px; margin-top:.6rem; }}
.src {{ display:inline-flex; flex-direction:column; max-width:250px; padding:6px 10px; border-radius:10px;
        background: rgba(128,128,128,.10); text-decoration:none !important; border:1px solid transparent; }}
.src:hover {{ border-color:{ACCENT}; }}
.src b {{ font-size:.78rem; font-weight:500; color:inherit; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.src i {{ font-size:.7rem; opacity:.6; font-style:normal; }}

.empty h2 {{ font-weight:600; letter-spacing:-0.02em; margin-bottom:.2rem; }}
.empty p {{ opacity:.7; margin-top:0; }}
@media (max-width: 640px) {{ .block-container {{ padding-left: .8rem; padding-right: .8rem; }} }}
</style>
""",
    unsafe_allow_html=True,
)


# ── helpers ──────────────────────────────────────────────────────────────────
def fmt_ts(ts: str) -> str:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).strftime("%d %b %Y, %H:%M UTC")
    except ValueError:
        return ts


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    chips = "".join(
        f'<a class="src" href="{html.escape(s["url"], quote=True)}" target="_blank" rel="noopener noreferrer">'
        f'<b>{html.escape(s["title"][:80])}</b><i>{html.escape(s["domain"])}</i></a>'
        for s in sources
    )
    st.markdown(f'<div class="sources">{chips}</div>', unsafe_allow_html=True)


def render_meta(msg: dict) -> None:
    badge = ""
    if msg["role"] == "assistant":
        badge = (
            '<span class="badge web">Searched the web</span>'
            if msg["used_search"]
            else '<span class="badge">From memory</span>'
        )
    label = "Asked" if msg["role"] == "user" else "Answered"
    st.markdown(f'<div class="meta">{badge}{label} {fmt_ts(msg["created_at"])}</div>', unsafe_allow_html=True)


def render_message(msg: dict) -> None:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_sources(msg["sources"])
        render_meta(msg)


def new_chat() -> None:
    st.session_state.cid = None


# ── state ────────────────────────────────────────────────────────────────────
st.session_state.setdefault("cid", None)
cid: str | None = st.session_state.cid

# ── sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="brand">Re<span>call</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="tagline">Web-searching chat that remembers what you asked, and when.</div>', unsafe_allow_html=True)
    st.button("New chat", on_click=new_chat, use_container_width=True, icon=":material/add:")

    convs = db.list_conversations()
    if convs:
        st.markdown("**History**")
    for c in convs:
        col_a, col_b = st.columns([5, 1], vertical_alignment="center")
        active = c["id"] == cid
        if col_a.button(c["title"], key=f"open_{c['id']}", type="primary" if active else "secondary",
                        use_container_width=True, help=f"Started {fmt_ts(c['created_at'])}"):
            st.session_state.cid = c["id"]
            st.rerun()
        if col_b.button("", key=f"del_{c['id']}", icon=":material/delete:", help="Delete conversation"):
            db.delete_conversation(c["id"])
            if active:
                st.session_state.cid = None
            st.rerun()

    if cid:
        qs = db.get_user_questions(cid)
        if qs:
            with st.expander(f"What you asked ({len(qs)})"):
                for q in qs:
                    st.markdown(f"**{fmt_ts(q['created_at'])}**  \n{q['content']}")

# ── main pane ────────────────────────────────────────────────────────────────
messages = db.get_messages(cid) if cid else []

if not messages:
    st.markdown(
        '<div class="empty"><h2>Ask anything.</h2>'
        "<p>Recall searches the web for current facts, answers the rest directly, "
        "and keeps a dated record of every question.</p></div>",
        unsafe_allow_html=True,
    )
    suggestions = [
        "What are today's top technology headlines?",
        "Explain how HTTPS works in simple terms",
        "What's the latest stable version of Python?",
    ]
    cols = st.columns(len(suggestions))
    for col, s in zip(cols, suggestions):
        if col.button(s, use_container_width=True, key=f"sugg_{s}"):
            st.session_state.pending = s
            st.rerun()

for m in messages:
    render_message(m)

prompt = st.chat_input("Ask a question, or follow up on a previous one…") or st.session_state.pop("pending", None)

if prompt:
    prompt = prompt.strip()
    if not cid:
        title = prompt if len(prompt) <= 48 else prompt[:47].rstrip() + "…"
        cid = st.session_state.cid = db.create_conversation(title)

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status = st.empty()
        body = st.empty()
        status.markdown("_Thinking…_")
        buf, ok = "", True
        try:
            for ev in agent.stream_answer(cid, prompt):
                if ev["type"] == "status":
                    status.markdown(f"_{ev['text']}_")
                elif ev["type"] == "delta":
                    buf += ev["text"]
                    status.empty()
                    body.markdown(buf + "▌")
        except Exception as exc:  # surface a clear, actionable error
            ok = False
            status.empty()
            body.error(f"Something went wrong: {exc}\n\nYour question was saved. Send it again to retry.")
    if ok:
        st.rerun()  # reload from SQLite: renders sources, badges, timestamps, sidebar title
