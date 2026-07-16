"""Streamlit front-end for the voice code assistant — premium dark theme.

Design system: glassmorphism over layered radial gradients, Sora/Inter/
JetBrains Mono fonts, neon violet→cyan→pink accents, live GIF assets
(generated locally, served from ui/static via Streamlit static serving).

Run:
    .venv\\Scripts\\python.exe -m streamlit run ui/streamlit_app.py
"""
import hashlib
import json
import sys
from pathlib import Path

import requests
import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import BACKEND_URL, SIGNLANG_PYTHON, SIGNLANG_URL  # noqa: E402

st.set_page_config(page_title="Voice Code Assistant", page_icon="🎙️",
                   layout="wide", initial_sidebar_state="expanded")

TTS_VOICES = ["alloy", "ash", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer"]
LANG_HINTS = {"auto-detect": "", "English": "en", "Hindi": "hi", "Bengali": "bn",
              "Spanish": "es", "French": "fr", "German": "de", "Japanese": "ja",
              "Mandarin": "zh", "Arabic": "ar", "Portuguese": "pt", "Tamil": "ta"}
SUGGESTIONS = [
    "⚡ What happens when an order is placed, step by step?",
    "🧾 Where does the 2% legacy surcharge come from?",
    "🛰️ Which method talks to the mainframe, and what are its risks?",
]

WAVE = "app/static/wave.gif"
ORB = "app/static/orb.gif"
DOTS = "app/static/dots.gif"

# ══════════════════════════════════════════════════════════════════
#  THEME
# ══════════════════════════════════════════════════════════════════

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

:root{
  --bg:#0A0B14; --panel:rgba(255,255,255,.035); --line:rgba(255,255,255,.08);
  --violet:#7C6CFF; --cyan:#4CC9F0; --pink:#F72585; --green:#3DD68C;
  --amber:#FFB454; --text:#E8EAF6; --muted:#98A0B8;
}

html, body, .stApp, [class*="css"]{ font-family:'Inter',system-ui,sans-serif; color:var(--text); }
h1,h2,h3,h4{ font-family:'Sora',sans-serif !important; letter-spacing:-.02em; }
code, pre, kbd{ font-family:'JetBrains Mono',monospace !important; }

/* ── layered aurora background ── */
.stApp{
  background:
    radial-gradient(1100px 520px at 85% -10%, rgba(124,108,255,.16), transparent 60%),
    radial-gradient(900px 480px at -10% 25%, rgba(76,201,240,.10), transparent 55%),
    radial-gradient(760px 480px at 55% 115%, rgba(247,37,133,.09), transparent 60%),
    var(--bg) !important;
}
[data-testid="stHeader"]{ background:transparent; }
#MainMenu, footer, [data-testid="stAppDeployButton"], .stDeployButton{ visibility:hidden; }

/* ── sidebar: frosted glass ── */
[data-testid="stSidebar"]{
  background:rgba(13,14,24,.82); backdrop-filter:blur(20px);
  border-right:1px solid var(--line);
}
[data-testid="stSidebar"] hr{ border-color:var(--line); }

/* ── tabs → glowing pills ── */
.stTabs [data-baseweb="tab-list"]{
  gap:.35rem; background:rgba(255,255,255,.03); border:1px solid var(--line);
  padding:.4rem; border-radius:16px; width:fit-content;
}
.stTabs [data-baseweb="tab"]{
  border-radius:11px; padding:.5rem 1.05rem; background:transparent;
  color:var(--muted); font-weight:600; font-family:'Sora',sans-serif; border:none;
}
.stTabs [data-baseweb="tab"]:hover{ color:var(--text); background:rgba(255,255,255,.05); }
.stTabs [aria-selected="true"]{
  background:linear-gradient(135deg,var(--violet),var(--cyan)) !important;
  color:#fff !important; box-shadow:0 6px 24px rgba(124,108,255,.35);
}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"]{ display:none; }

/* ── chat bubbles: glass cards ── */
[data-testid="stChatMessage"]{
  background:var(--panel); border:1px solid var(--line); border-radius:20px;
  padding:1.05rem 1.25rem; backdrop-filter:blur(10px); margin-bottom:.35rem;
  box-shadow:0 8px 30px rgba(0,0,0,.25);
}

/* ── buttons ── */
.stButton>button, .stFormSubmitButton>button{
  border-radius:12px; border:1px solid var(--line); background:rgba(255,255,255,.045);
  color:var(--text); font-weight:600; font-family:'Sora',sans-serif;
  transition:all .18s ease;
}
.stButton>button:hover, .stFormSubmitButton>button:hover{
  border-color:rgba(124,108,255,.6); transform:translateY(-1px);
  box-shadow:0 8px 26px rgba(124,108,255,.28); color:#fff;
}
.stButton>button[kind="primary"], .stFormSubmitButton>button[kind="primaryFormSubmit"]{
  background:linear-gradient(135deg,var(--violet) 0%,var(--cyan) 120%);
  border:none; color:#fff; box-shadow:0 6px 26px rgba(124,108,255,.4);
}

/* ── inputs ── */
[data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea,
[data-testid="stNumberInput"] input{
  background:rgba(255,255,255,.045) !important; border-radius:12px !important;
  border:1px solid var(--line) !important; color:var(--text) !important;
}
[data-testid="stTextInput"] input:focus, [data-testid="stTextArea"] textarea:focus{
  border-color:var(--violet) !important; box-shadow:0 0 0 3px rgba(124,108,255,.22) !important;
}
[data-testid="stChatInput"]{
  background:rgba(255,255,255,.05); border:1px solid rgba(124,108,255,.35);
  border-radius:16px; box-shadow:0 10px 34px rgba(124,108,255,.18);
}
[data-baseweb="select"]>div{
  background:rgba(255,255,255,.045) !important; border-radius:12px !important;
  border-color:var(--line) !important;
}

/* ── expander → evidence card ── */
[data-testid="stExpander"]{
  background:rgba(124,108,255,.05); border:1px solid rgba(124,108,255,.22);
  border-radius:16px; overflow:hidden;
}
[data-testid="stExpander"] summary{ font-family:'Sora',sans-serif; font-weight:600; }

/* ── bordered containers → glass cards ── */
[data-testid="stVerticalBlockBorderWrapper"]{
  background:var(--panel); border:1px solid var(--line) !important;
  border-radius:18px !important; backdrop-filter:blur(10px);
}

/* code blocks */
.stCode, [data-testid="stCode"]{ border-radius:14px; }
pre{ background:#0D0F1C !important; border:1px solid var(--line); border-radius:14px; }

/* dataframe/table */
[data-testid="stTable"], [data-testid="stDataFrame"]{
  border:1px solid var(--line); border-radius:14px; overflow:hidden;
}

/* audio player */
audio{ width:100%; filter:invert(.9) hue-rotate(180deg); border-radius:12px; }

/* ── custom components ── */
@keyframes shimmer{ 0%{background-position:0% 50%} 100%{background-position:200% 50%} }
@keyframes floaty{ 0%,100%{transform:translateY(0)} 50%{transform:translateY(-7px)} }
@keyframes eq{ 0%,100%{height:5px} 50%{height:20px} }

.vca-title{
  font-family:'Sora',sans-serif; font-weight:800; letter-spacing:-.03em;
  background:linear-gradient(90deg,#E8EAF6,var(--violet),var(--cyan),var(--pink),#E8EAF6);
  background-size:200% auto; -webkit-background-clip:text; background-clip:text;
  -webkit-text-fill-color:transparent; animation:shimmer 6s linear infinite;
}
.vca-sub{ color:var(--muted); font-size:1.02rem; line-height:1.6; }

.vca-pill{
  display:inline-flex; align-items:center; gap:.4rem; padding:.28rem .8rem;
  border-radius:999px; font-size:.8rem; font-weight:600; letter-spacing:.01em;
  background:rgba(255,255,255,.05); border:1px solid var(--line); color:var(--muted);
  margin:.15rem .3rem .15rem 0;
}
.vca-pill.violet{ color:#CBC4FF; border-color:rgba(124,108,255,.45); background:rgba(124,108,255,.12); }
.vca-pill.cyan{ color:#A8E9FF; border-color:rgba(76,201,240,.45); background:rgba(76,201,240,.10); }
.vca-pill.pink{ color:#FFB3D2; border-color:rgba(247,37,133,.4); background:rgba(247,37,133,.10); }
.vca-pill.green{ color:#9CF0C8; border-color:rgba(61,214,140,.45); background:rgba(61,214,140,.10); }
.vca-pill.amber{ color:#FFD9A0; border-color:rgba(255,180,84,.45); background:rgba(255,180,84,.10); }

.vca-chip{
  display:flex; gap:.65rem; align-items:flex-start; padding:.8rem 1rem;
  background:var(--panel); border:1px solid var(--line); border-radius:14px;
  margin-bottom:.55rem;
}
.vca-chip .ic{ font-size:1.25rem; line-height:1.3; }
.vca-chip b{ font-family:'Sora',sans-serif; font-size:.9rem; }
.vca-chip span{ color:var(--muted); font-size:.8rem; display:block; }

.vca-cite{
  display:flex; align-items:center; gap:.7rem; padding:.55rem .8rem;
  background:rgba(255,255,255,.04); border:1px solid var(--line);
  border-radius:12px; margin-bottom:.45rem; font-size:.85rem;
}
.vca-cite .n{
  font-family:'JetBrains Mono',monospace; font-weight:600; color:var(--cyan);
  background:rgba(76,201,240,.12); border-radius:8px; padding:.1rem .45rem;
}
.vca-cite code{ color:#CBC4FF; background:none; font-size:.82rem; }
.vca-cite .lines{ color:var(--muted); font-size:.78rem; white-space:nowrap; }
.vca-cite .simbar{
  flex:1; min-width:70px; height:6px; background:rgba(255,255,255,.07);
  border-radius:99px; overflow:hidden;
}
.vca-cite .simbar i{
  display:block; height:100%; border-radius:99px;
  background:linear-gradient(90deg,var(--violet),var(--cyan));
}

.vca-eq{ display:inline-flex; gap:3px; align-items:flex-end; height:20px; margin-right:.55rem; }
.vca-eq i{ width:4px; border-radius:3px; background:linear-gradient(180deg,var(--cyan),var(--violet));
  animation:eq 1s ease-in-out infinite; }
.vca-eq i:nth-child(2){ animation-delay:.15s } .vca-eq i:nth-child(3){ animation-delay:.3s }
.vca-eq i:nth-child(4){ animation-delay:.45s } .vca-eq i:nth-child(5){ animation-delay:.6s }

.vca-user{
  display:flex; align-items:center; gap:.8rem; padding:.85rem 1rem;
  background:var(--panel); border:1px solid var(--line); border-radius:16px;
}
.vca-user .av{
  width:42px; height:42px; border-radius:14px; display:flex; align-items:center;
  justify-content:center; font-family:'Sora',sans-serif; font-weight:800; font-size:1.15rem;
  background:linear-gradient(135deg,var(--violet),var(--cyan)); color:#fff;
  box-shadow:0 6px 20px rgba(124,108,255,.4);
}
.vca-status{ display:flex; align-items:center; gap:.55rem; color:var(--muted); font-size:.85rem; }
.vca-float{ animation:floaty 5s ease-in-out infinite; }
.vca-hr{ border:none; height:1px; margin:.9rem 0;
  background:linear-gradient(90deg,transparent,rgba(124,108,255,.4),transparent); }
</style>
"""


def eq_bars() -> str:
    return "<span class='vca-eq'><i></i><i></i><i></i><i></i><i></i></span>"


def section_title(icon: str, text: str, sub: str = "") -> None:
    st.markdown(
        f"<h3 style='display:flex;align-items:center;margin-bottom:.15rem'>{eq_bars()}"
        f"{icon}&nbsp;{text}</h3>"
        + (f"<p class='vca-sub' style='margin-top:0'>{sub}</p>" if sub else ""),
        unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  BACKEND HELPERS
# ══════════════════════════════════════════════════════════════════

def api(method: str, path: str, **kwargs):
    headers = kwargs.pop("headers", {})
    if st.session_state.get("token"):
        headers["Authorization"] = f"Bearer {st.session_state.token}"
    try:
        resp = requests.request(method, f"{BACKEND_URL}{path}", headers=headers,
                                timeout=kwargs.pop("timeout", 300), **kwargs)
    except requests.ConnectionError:
        st.error(f"⚠️ Backend not reachable at {BACKEND_URL}. Start it with run_backend.ps1")
        st.stop()
    if resp.status_code == 401:
        st.session_state.pop("token", None)
        st.error("Session expired — please log in again.")
        st.stop()
    return resp


def ask_stream(question: str, repo: str, session_id: str):
    headers = {"Authorization": f"Bearer {st.session_state.token}"}
    with requests.post(f"{BACKEND_URL}/api/ask", headers=headers, stream=True,
                       json={"question": question, "repo": repo,
                             "session_id": session_id}, timeout=300) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if line and line.startswith("data: "):
                yield json.loads(line[6:])


# ══════════════════════════════════════════════════════════════════
#  LOGIN
# ══════════════════════════════════════════════════════════════════

def login_gate() -> dict:
    if "token" not in st.session_state:
        st.markdown("<div style='height:4vh'></div>", unsafe_allow_html=True)
        left, right = st.columns([1.25, 1], gap="large")
        with left:
            st.markdown(f"""
<div style='display:flex;align-items:center;gap:1rem'>
  <img src='{ORB}' width='72' class='vca-float' style='border-radius:24px'/>
  <div>
    <div class='vca-title' style='font-size:2.9rem;line-height:1.08'>Voice Code Assistant</div>
    <div class='vca-sub'>Talk to your <b>legacy codebase</b> — knowledge graph ◈ RAG ◈ speech</div>
  </div>
</div>""", unsafe_allow_html=True)
            st.markdown(f"<img src='{WAVE}' style='width:100%;border-radius:22px;"
                        f"border:1px solid rgba(124,108,255,.25);margin:1.1rem 0;"
                        f"box-shadow:0 18px 60px rgba(124,108,255,.22)'/>",
                        unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            chips = [
                ("🎙️", "Wake word + streaming voice", "answers begin speaking before they finish generating"),
                ("🕸️", "Code knowledge graph", "GitNexus analysis → Graphify graph → cited answers"),
                ("🤟", "Sign-language input", "MediaPipe ASL recognition as a full input path"),
                ("🛡️", "Role-aware access", "code you can't see never reaches the model"),
            ]
            for i, (ic, t, s) in enumerate(chips):
                with (c1 if i % 2 == 0 else c2):
                    st.markdown(f"<div class='vca-chip'><span class='ic'>{ic}</span>"
                                f"<span><b>{t}</b><span>{s}</span></span></div>",
                                unsafe_allow_html=True)
        with right:
            st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown("<h3 style='margin:.2rem 0'>✦ Sign in</h3>", unsafe_allow_html=True)
                with st.form("login", border=False):
                    username = st.text_input("Username", value="dev")
                    password = st.text_input("Password", type="password", value="dev123")
                    if st.form_submit_button("Enter the assistant  ⟶", use_container_width=True,
                                             type="primary"):
                        resp = api("POST", "/api/auth/token",
                                   data={"username": username, "password": password})
                        if resp.ok:
                            st.session_state.token = resp.json()["access_token"]
                            st.rerun()
                        else:
                            st.error("Invalid credentials")
                st.markdown(
                    "<span class='vca-pill violet'>👑 admin / admin123</span>"
                    "<span class='vca-pill cyan'>🛠️ dev / dev123</span>"
                    "<span class='vca-pill'>👀 viewer / viewer123</span>",
                    unsafe_allow_html=True)
                st.markdown("<p class='vca-sub' style='font-size:.8rem'>viewer sees docs only — "
                            "restricted source never reaches the LLM.</p>", unsafe_allow_html=True)
        st.stop()
    me = api("GET", "/api/me").json()
    st.session_state.me = me
    return me


# ══════════════════════════════════════════════════════════════════
#  SHARED RENDERERS
# ══════════════════════════════════════════════════════════════════

def render_message(msg: dict):
    with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "🎙️"):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and (msg.get("citations") or msg.get("confidence") is not None):
            conf = msg.get("confidence")
            render_evidence(msg.get("citations") or [],
                            {"score": conf, "band": ""} if isinstance(conf, float) else conf)


def render_evidence(citations: list, confidence: dict | None,
                    provider: str = "", stages: dict | None = None,
                    agents: list | None = None):
    if agents:
        st.markdown("<div style='margin:.2rem 0'>" + " <span style='color:var(--muted)'>→</span> ".join(
            f"<span class='vca-pill'>{a['agent']}</span>" for a in agents)
            + "</div>", unsafe_allow_html=True)
    pills = []
    if confidence and confidence.get("score") is not None:
        band = confidence.get("band") or ""
        color = {"high": "green", "medium": "amber", "low": "pink", "none": "pink"}.get(band, "cyan")
        pills.append(f"<span class='vca-pill {color}'>◉ confidence {confidence['score']}"
                     f"{' · ' + band if band else ''}</span>")
    if provider:
        pills.append(f"<span class='vca-pill violet'>⚡ {provider}</span>")
    if stages:
        lat = " · ".join(f"{k} {v/1000:.1f}s" if v >= 1000 else f"{k} {v:.0f}ms"
                         for k, v in stages.items())
        pills.append(f"<span class='vca-pill'>⏱ {lat}</span>")
    if pills:
        st.markdown("<div style='margin:.35rem 0'>" + "".join(pills) + "</div>",
                    unsafe_allow_html=True)
    if citations:
        with st.expander(f"🧾 Evidence · {len(citations)} citation(s) — code & graph"):
            for c in citations:
                sim = float(c.get("score") or 0)
                st.markdown(f"""
<div class='vca-cite'>
  <span class='n'>[{c['n']}]</span>
  <code>{c['path']}</code>
  <span class='lines'>L{c['start']}–{c['end']}{' · ' + c['symbol'] if c.get('symbol') else ''}</span>
  <span class='simbar'><i style='width:{min(100, sim * 100):.0f}%'></i></span>
  <span class='lines'>{sim:.2f}</span>
</div>""", unsafe_allow_html=True)


def transcribe_audio(audio_file) -> dict | None:
    digest = hashlib.md5(audio_file.getvalue()).hexdigest()
    if st.session_state.get("last_audio") == digest:
        return None
    st.session_state.last_audio = digest
    resp = api("POST", "/api/stt",
               files={"file": ("mic.wav", audio_file.getvalue(), "audio/wav")})
    if not resp.ok:
        st.error(f"STT failed: {resp.text}")
        return None
    return resp.json()


# ══════════════════════════════════════════════════════════════════
#  LIVE WAKE-WORD MODE (backend mic loop, controlled from the UI)
# ══════════════════════════════════════════════════════════════════

LIVE_STATES = {
    "starting": ("cyan", "⏳ starting…"),
    "waiting_wake": ("violet", "👂 say the wake word"),
    "capturing": ("pink", "🎤 listening — speak now"),
    "transcribing": ("amber", "📝 transcribing…"),
    "thinking": ("cyan", "🧠 thinking…"),
    "speaking": ("green", "🔊 speaking — say the wake word to interrupt"),
}


@st.fragment(run_every=1.5)
def live_status_panel():
    live = api("GET", "/api/live/status").json()
    if not live["running"]:
        st.markdown("<span class='vca-pill'>● stopped</span>", unsafe_allow_html=True)
        return
    color, label = LIVE_STATES.get(live["state"], ("", live["state"]))
    if live["state"] == "waiting_wake":
        label = f"👂 say “{live.get('wake_model', 'hey jarvis').replace('_', ' ')}”"
    st.markdown(f"<span class='vca-pill {color}'>{label}</span>", unsafe_allow_html=True)
    if live["state"] in ("waiting_wake", "capturing"):
        lvl = min(100, live.get("mic_level", 0) * 800)
        score = live.get("wake_score", 0)
        bar_color = "var(--green)" if lvl > 8 else "var(--pink)"
        st.markdown(f"""
<div style='display:flex;align-items:center;gap:.55rem;margin:.3rem 0'>
  <span class='vca-sub' style='font-size:.72rem;white-space:nowrap'>🎚 mic</span>
  <div style='flex:1;height:7px;background:rgba(255,255,255,.07);border-radius:99px;overflow:hidden'>
    <div style='width:{lvl:.0f}%;height:100%;border-radius:99px;background:{bar_color};transition:width .3s'></div>
  </div>
  <span class='vca-sub' style='font-size:.72rem;white-space:nowrap'>wake {score:.2f}</span>
</div>
<span class='vca-sub' style='font-size:.68rem'>🎤 {live.get("input_device", "?")}</span>""",
                    unsafe_allow_html=True)
    heard = live.get("heard") or {}
    if heard.get("text"):
        st.markdown(f"<span class='vca-sub' style='font-size:.78rem'>🗣️ "
                    f"“{heard['text'][:90]}”</span>", unsafe_allow_html=True)
    if live["state"] in ("thinking", "speaking") and live.get("partial_answer"):
        st.markdown(f"<span class='vca-sub' style='font-size:.75rem'>"
                    f"{live['partial_answer'][-160:]} ▌</span>", unsafe_allow_html=True)
    if not live.get("wake_available") and live["state"] != "starting":
        st.markdown("<span class='vca-pill amber'>wake model unavailable — "
                    "use 👂 Listen now</span>", unsafe_allow_html=True)
    if live.get("error"):
        st.markdown(f"<span class='vca-pill pink'>⚠ {live['error'][:90]}</span>",
                    unsafe_allow_html=True)


@st.fragment(run_every=1.0)
def live_chat_panel():
    """Live wake-word turns appear in the chat area as they happen: the
    moment the wake word fires, a user bubble opens; the answer then streams
    into an assistant bubble. When the turn completes it is committed to
    session history and the page refreshes."""
    live = api("GET", "/api/live/status").json()
    if not live["running"]:
        return
    # the chat follows the live conversation's session
    live_sid = live.get("session_id", "")
    if live_sid and live_sid != st.session_state.get("session_id"):
        st.session_state.session_id = live_sid
        st.rerun(scope="app")
    state = live["state"]
    heard = (live.get("heard") or {}).get("text", "")

    if state == "waiting_wake":
        st.markdown("<span class='vca-pill violet'>👂 hands-free active — say "
                    f"“{live.get('wake_model', 'hey jarvis').replace('_', ' ')}” "
                    "to ask</span>", unsafe_allow_html=True)
    elif state in ("capturing", "transcribing"):
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown("🎤 *listening — speak now…*" if state == "capturing"
                        else (heard or "📝 *transcribing…*"))
    elif state in ("thinking", "speaking"):
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(heard or "…")
        with st.chat_message("assistant", avatar="🎙️"):
            partial = live.get("partial_answer") or ""
            st.markdown(partial + " ▌" if partial else "🧠 *thinking…*")

    # spoken turn completed -> commit to chat history with citations
    if live.get("turns", 0) != st.session_state.get("live_turns", 0):
        st.session_state.live_turns = live.get("turns", 0)
        st.rerun(scope="app")


def voice_card(repo: str, repos: list) -> bool:
    """Voice panel: TTS toggle + wake-word live mode controls."""
    st.markdown("<h4>🔊 Voice</h4>", unsafe_allow_html=True)
    speak = st.toggle("Speak answers (streamed TTS)", value=True)
    live = api("GET", "/api/live/status").json()

    if not live["running"]:
        if st.button("🎙️ Start wake-word mode", type="primary",
                     use_container_width=True, disabled=not repos):
            resp = api("POST", "/api/live/start",
                       json={"repo": repo,
                             "session_id": st.session_state.get("session_id") or ""}).json()
            if resp.get("session_id"):
                st.session_state.session_id = resp["session_id"]
            st.session_state.live_turns = resp.get("turns", 0)
            st.rerun()
        if live.get("error"):
            st.markdown(f"<span class='vca-pill pink'>⚠ {live['error'][:90]}</span>",
                        unsafe_allow_html=True)
        st.markdown("<span class='vca-sub' style='font-size:.75rem'>hands-free: say "
                    "<b>“hey jarvis”</b> → beep → ask → spoken answer streams back. "
                    "Say it again to interrupt. Uses this machine's mic & speakers "
                    "(everything runs locally).</span>", unsafe_allow_html=True)
    else:
        c1, c2 = st.columns(2)
        if c1.button("⏹ Stop", use_container_width=True):
            api("POST", "/api/live/stop")
            st.rerun()
        if c2.button("👂 Listen now", use_container_width=True,
                     help="capture immediately without saying the wake word"):
            api("POST", "/api/live/trigger")
        live_status_panel()
    return speak


# ══════════════════════════════════════════════════════════════════
#  ASSISTANT TAB
# ══════════════════════════════════════════════════════════════════

def assistant_tab(me: dict):
    left, right = st.columns([3, 1.05], gap="medium")

    with right:
        with st.container(border=True):
            st.markdown("<h4>📦 Repository</h4>", unsafe_allow_html=True)
            repos = api("GET", "/api/repos").json()["repos"]
            last = st.session_state.get("last_repo")
            repo = st.selectbox("Indexed repos", repos or ["(none indexed)"],
                                index=repos.index(last) if last in repos else 0,
                                label_visibility="collapsed")
            st.session_state.last_repo = repo
            if "index_repo" in me["capabilities"]:
                with st.expander("＋ Index a new repo"):
                    mode = st.radio("Source", ["📁 Path", "📤 Upload", "🔗 URL"],
                                    horizontal=True, label_visibility="collapsed")
                    resp = None
                    if mode == "📁 Path":
                        path = st.text_input("Local folder",
                                             placeholder=r"D:\path\to\legacy\repo")
                        if st.button("🚀 Analyze & index", use_container_width=True) and path:
                            with st.spinner("GitNexus → Graphify → embeddings..."):
                                resp = api("POST", "/api/index",
                                           json={"path": path}, timeout=1800)
                    elif mode == "📤 Upload":
                        up_name = st.text_input("Repo name (optional)",
                                                placeholder="my-legacy-app")
                        uploads = st.file_uploader(
                            "ZIP / tar.gz archive, or individual source files",
                            accept_multiple_files=True,
                            type=["zip", "gz", "tgz", "tar", "py", "java", "js",
                                  "ts", "cs", "c", "cpp", "h", "php", "rb", "go",
                                  "sql", "md", "txt", "yml", "yaml", "json", "xml"])
                        if st.button("🚀 Import & index", use_container_width=True) and uploads:
                            with st.spinner("extracting → GitNexus → Graphify → embeddings..."):
                                resp = api("POST", "/api/index/upload",
                                           files=[("files", (u.name, u.getvalue()))
                                                  for u in uploads],
                                           data={"name": up_name}, timeout=1800)
                    else:
                        url = st.text_input("GitHub link or archive URL",
                                            placeholder="https://github.com/owner/repo")
                        url_name = st.text_input("Repo name (optional)", key="url_name")
                        if st.button("🚀 Fetch & index", use_container_width=True) and url:
                            with st.spinner("downloading → GitNexus → Graphify → embeddings..."):
                                resp = api("POST", "/api/index",
                                           json={"url": url, "name": url_name},
                                           timeout=1800)
                    if resp is not None:
                        if resp.ok:
                            s = resp.json()
                            st.success(f"Indexed **{s['repo']}** — {s['files']} files · "
                                       f"{s['graph_nodes']} nodes · {s['chunks_indexed']} chunks")
                            st.rerun()
                        else:
                            st.error(resp.text)

        with st.container(border=True):
            st.markdown("<h4>🧠 Session memory</h4>", unsafe_allow_html=True)
            sessions = api("GET", "/api/sessions").json()["sessions"]
            options = {"✨ new session": None}
            options.update({s["title"][:40]: s["session_id"] for s in sessions})
            choice = st.selectbox("Session", list(options.keys()), label_visibility="collapsed")
            if options[choice] and options[choice] != st.session_state.get("session_id"):
                st.session_state.session_id = options[choice]
                st.rerun()
            st.markdown("<span class='vca-sub' style='font-size:.78rem'>follow-ups work: "
                        "<i>“now explain that service in more detail”</i></span>",
                        unsafe_allow_html=True)

        with st.container(border=True):
            speak = voice_card(repo if repos else "", repos)

    with left:
        if not repos:
            st.info("Index a repository first (panel on the right) — try the bundled "
                    "`sample_legacy` demo codebase.")
            return
        if st.session_state.get("session_id") is None:
            st.session_state.session_id = api(
                "POST", "/api/sessions", json={"repo": repo}).json()["session_id"]
        sid = st.session_state.session_id

        history = api("GET", f"/api/sessions/{sid}/messages").json()["messages"]

        if history:   # clear the screen and start over
            _, cbtn = st.columns([4.2, 1])
            if cbtn.button("🧹 Clear chat", use_container_width=True,
                           help="start a fresh conversation (this one stays "
                                "in the Session memory list)"):
                live = api("GET", "/api/live/status").json()
                new_sid = api("POST", "/api/sessions",
                              json={"repo": repo}).json()["session_id"]
                if live.get("running"):   # move live voice onto the new session
                    api("POST", "/api/live/stop")
                    api("POST", "/api/live/start",
                        json={"repo": live.get("repo") or repo,
                              "session_id": new_sid})
                st.session_state.session_id = new_sid
                st.session_state.live_turns = 0
                for key in ("last_final", "pending_stt", "pending_question"):
                    st.session_state.pop(key, None)
                st.rerun()

        if not history:   # hero + suggestion chips on a fresh session
            st.markdown(f"""
<div style='display:flex;gap:1.1rem;align-items:center;background:var(--panel);
     border:1px solid var(--line);border-radius:20px;padding:1.1rem 1.3rem'>
  <img src='{ORB}' width='58' style='border-radius:18px' class='vca-float'/>
  <div>
    <div class='vca-title' style='font-size:1.5rem'>Ask me anything about
      <span style='color:var(--cyan);-webkit-text-fill-color:var(--cyan);
      font-family:JetBrains Mono,monospace'>{repo}</span></div>
    <div class='vca-sub' style='font-size:.9rem'>speak 🎙️ · type ⌨️ · sign 🤟 — answers stream
    with citations, confidence and graph evidence</div>
  </div>
</div>""", unsafe_allow_html=True)
            cols = st.columns(len(SUGGESTIONS))
            for col, s in zip(cols, SUGGESTIONS):
                if col.button(s, use_container_width=True, key=f"sugg_{s[:12]}"):
                    st.session_state.pending_question = s.split(" ", 1)[1]
                    st.rerun()

        for m in history:
            render_message(m)

        # live wake-word turns stream directly into the chat area
        live_chat_panel()

        # ── push-to-talk with ASR confirmation loop ──
        audio = st.audio_input("🎤 Push to talk", label_visibility="collapsed")
        if audio is not None:
            heard = transcribe_audio(audio)
            if heard:
                st.session_state.pending_stt = heard
        pending = st.session_state.get("pending_stt")
        if pending:
            st.markdown(f"<span class='vca-pill cyan'>🗣️ {pending['language']}</span>"
                        f"<span class='vca-pill {'amber' if pending['needs_confirmation'] else 'green'}'>"
                        f"◉ STT confidence {pending['confidence']:.2f}</span>",
                        unsafe_allow_html=True)
            corrected = st.text_input("Transcript — edit if wrong, then Ask",
                                      value=pending["text"], key="stt_correction")
            c1, c2 = st.columns([1, 5])
            if c1.button("Ask ⟶", type="primary"):
                if corrected.strip() and corrected.strip() != pending["text"].strip():
                    api("POST", "/api/asr_feedback",
                        json={"heard": pending["text"], "corrected": corrected,
                              "language": pending["language"]})
                st.session_state.pending_question = corrected.strip()
                st.session_state.pending_stt = None
                st.rerun()
            if c2.button("Discard ✕"):
                st.session_state.pending_stt = None
                st.rerun()

        typed = st.chat_input(f"Ask about {repo}…   (or use the mic above)")
        question = typed or st.session_state.pop("pending_question", None)

        if question:
            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(question)
            with st.chat_message("assistant", avatar="🎙️"):
                status = st.empty()
                body = st.empty()
                acc, final = "", None
                try:
                    for event in ask_stream(question, repo, sid):
                        if event["type"] == "status":
                            status.markdown(
                                f"<div class='vca-status'><img src='{DOTS}' width='52'/>"
                                f"{event['text']}</div>", unsafe_allow_html=True)
                        elif event["type"] == "token":
                            acc += event["text"]
                            body.markdown(acc + " ▌")
                        elif event["type"] == "final":
                            final = event
                        elif event["type"] == "error":
                            st.error(event["text"])
                except requests.RequestException as exc:
                    st.error(f"stream failed: {exc}")
                status.empty()
                if final:
                    body.markdown(final["answer"])
                    render_evidence(final["citations"], final["confidence"],
                                    final.get("provider", ""), final.get("stages"),
                                    final.get("agents"))
                    st.session_state.last_final = final
                    if speak and final["answer"]:
                        with st.spinner("🔊 synthesizing speech…"):
                            resp = api("POST", "/api/tts",
                                       json={"text": final["answer"][:3000]})
                        if resp.ok:
                            st.audio(resp.content, format="audio/wav", autoplay=True)


# ══════════════════════════════════════════════════════════════════
#  OTHER TABS
# ══════════════════════════════════════════════════════════════════

def navigator_tab(me: dict):
    section_title("🧭", "Code navigation",
                  "the exact files, methods and lines behind the last answer")
    final = st.session_state.get("last_final")
    if "view_code" not in me["capabilities"]:
        st.warning(f"Role **{me['role']}** is not authorized to view source code.")
        return
    if not final or not final.get("citations"):
        st.info("Ask a question first — cited files, methods and graph nodes appear "
                "here with the exact lines highlighted.")
        return
    repos = api("GET", "/api/repos").json()["repos"]
    repo = st.selectbox("Repo", repos, key="nav_repo_sel")
    labels = [f"[{c['n']}]  {c['path']} · L{c['start']}–{c['end']}"
              f"{'  ⌁ ' + c['symbol'] if c.get('symbol') else ''}"
              for c in final["citations"]]
    pick = st.radio("Citations", labels, label_visibility="collapsed")
    cite = final["citations"][labels.index(pick)]
    pad = st.slider("Context lines", 0, 60, 10)
    resp = api("GET", "/api/file", params={
        "repo": repo, "path": cite["path"],
        "start": max(1, cite["start"] - pad), "end": cite["end"] + pad})
    if resp.ok:
        data = resp.json()
        lang = Path(cite["path"]).suffix.lstrip(".") or "text"
        st.markdown(f"<span class='vca-pill violet'>📄 {data['path']}</span>"
                    f"<span class='vca-pill'>L{data['start']}–{data['end']} of "
                    f"{data['total_lines']}</span>"
                    f"<span class='vca-pill cyan'>cited L{cite['start']}–{cite['end']}</span>",
                    unsafe_allow_html=True)
        st.code(data["content"], language={"py": "python", "js": "javascript",
                "ts": "typescript", "cs": "csharp"}.get(lang, lang), line_numbers=True)
    else:
        st.error(resp.text)


def graph_tab(me: dict):
    section_title("🕸️", "Knowledge graph", "GitNexus analysis → Graphify graph — "
                  "🔴 red nodes were cited by the last answer")
    repos = api("GET", "/api/repos").json()["repos"]
    if not repos:
        st.info("Index a repository first.")
        return
    c1, c2 = st.columns([1, 2])
    repo = c1.selectbox("Repo", repos, key="graph_repo")
    term = c2.text_input("🔎 Find node (class / function / file)")
    if term:
        nodes = api("GET", "/api/graph/search", params={"repo": repo, "q": term}).json()["nodes"]
        st.markdown("".join(
            f"<span class='vca-pill violet'>◈ {n['kind']}</span>"
            f"<span class='vca-pill'><code>{n['id']}</code></span><br/>"
            for n in nodes[:6]) or "no matches", unsafe_allow_html=True)

    final = st.session_state.get("last_final") or {}
    highlight = ",".join(final.get("graph_nodes", [])[:20])
    resp = api("GET", "/api/graph/html", params={"repo": repo, "highlight": highlight})
    if resp.ok:
        components.html(resp.text, height=740, scrolling=False)
    else:
        st.error(resp.text)
    summary = api("GET", "/api/graph/summary", params={"repo": repo})
    if summary.ok:
        s = summary.json()
        st.markdown(f"<span class='vca-pill cyan'>◈ {s['nodes']} nodes</span>"
                    f"<span class='vca-pill violet'>⇄ {s['edges']} edges</span>"
                    + "".join(f"<span class='vca-pill'>🔥 {n['label']}</span>"
                              for n in s["important"][:6]),
                    unsafe_allow_html=True)


def sign_tab():
    section_title("🤟", "Sign language input",
                  "MediaPipe hand landmarks + RandomForest — 24 ASL letters and "
                  "HELLO / THANK YOU / PLEASE, ported from SignSpeak")
    try:
        health = requests.get(f"{SIGNLANG_URL}/api/health", timeout=3).json()
        st.markdown(f"<span class='vca-pill green'>● service online</span>"
                    f"<span class='vca-pill violet'>letters {'✓' if health['letter_model'] else '✕'}</span>"
                    f"<span class='vca-pill cyan'>two-hand words "
                    f"{'✓' if health['two_hand_model'] else '✕'}</span>", unsafe_allow_html=True)
        components.iframe(SIGNLANG_URL, height=820, scrolling=True)
    except requests.RequestException:
        st.markdown("<span class='vca-pill pink'>● service offline</span>", unsafe_allow_html=True)
        st.warning("Start the sign-language sidecar (needs the Python 3.10 venv):")
        st.code(f'& "{SIGNLANG_PYTHON}" services\\signlang\\app.py', language="powershell")

    sentence = st.text_input("Signed sentence → ask the assistant",
                             placeholder="e.g. HELLO EXPLAIN BILLING")
    if st.button("Send to assistant ⟶", type="primary") and sentence.strip():
        st.session_state.pending_question = sentence.strip()
        st.info("Question queued — open the **Assistant** tab to watch the answer stream.")


def profile_tab(me: dict):
    section_title("✨", "Personalization",
                  "language hints + custom vocabulary directly improve recognition "
                  "of accents and domain terms")
    p = me["profile"]
    with st.container(border=True):
        with st.form("profile", border=False):
            c1, c2, c3 = st.columns(3)
            lang_label = next((k for k, v in LANG_HINTS.items() if v == p.get("language", "")),
                              "auto-detect")
            language = c1.selectbox("🗣️ Speech language hint", list(LANG_HINTS.keys()),
                                    index=list(LANG_HINTS.keys()).index(lang_label))
            voice = c2.selectbox("🔊 TTS voice", TTS_VOICES,
                                 index=TTS_VOICES.index(p.get("tts_voice", "alloy"))
                                 if p.get("tts_voice") in TTS_VOICES else 0)
            rate = c3.slider("⏩ Speech rate", 0.5, 1.5, float(p.get("speech_rate", 1.0)), 0.05)
            styles = ["concise", "detailed", "step-by-step"]
            style = st.selectbox("📝 Answer style", styles,
                                 index=styles.index(p["answer_style"])
                                 if p.get("answer_style") in styles else 0)
            vocab = st.text_area("📚 Custom vocabulary (domain terms, class names, acronyms — "
                                 "primes the speech recognizer)",
                                 value=p.get("vocabulary", ""),
                                 placeholder="InvoiceReconciler, LedgerSvc, COBOL, mainframe")
            if st.form_submit_button("💾 Save profile", type="primary"):
                api("PUT", "/api/profile", json={
                    "language": LANG_HINTS[language], "tts_voice": voice,
                    "speech_rate": rate, "vocabulary": vocab, "answer_style": style})
                st.success("Saved — applies to your next voice interaction.")
    st.markdown(f"<div class='vca-user'><div class='av'>{me['username'][0].upper()}</div>"
                f"<div><b>{me['username']}</b> "
                f"<span class='vca-pill violet'>{me['role']}</span><br/>"
                f"<span class='vca-sub' style='font-size:.78rem'>"
                f"{' · '.join(me['capabilities'])}</span></div></div>",
                unsafe_allow_html=True)


def observability_tab():
    section_title("📊", "Observability",
                  "structured Loguru logs · latency metrics · retrieval diagnostics · prompt traces")
    resp = api("GET", "/api/metrics")
    if not resp.ok:
        st.warning("Your role does not include metrics access.")
        return
    m = resp.json()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<h4>⏱ Latency by request kind</h4>", unsafe_allow_html=True)
        st.table([{"kind": k, **v} for k, v in m["kinds"].items()])
    with c2:
        st.markdown("<h4>🧩 Latency by pipeline stage</h4>", unsafe_allow_html=True)
        st.table([{"stage": k, **v} for k, v in m["stages"].items()])
    st.markdown("<h4>🛰️ Recent traces</h4><p class='vca-sub' style='font-size:.8rem'>"
                "full prompt + retrieval diagnostics in <code>data/logs/traces/</code>, "
                "structured logs in <code>data/logs/app.jsonl</code></p>",
                unsafe_allow_html=True)
    st.dataframe([
        {"trace": t["trace_id"], "kind": t["kind"], "user": t["user"],
         "total_ms": t["total_ms"],
         "stages": " ".join(f"{k}:{v:.0f}" for k, v in t["stages"].items()),
         "meta": json.dumps(t.get("meta", {}))[:120]}
        for t in m["recent"]], use_container_width=True)
    bias = api("GET", "/api/asr_bias")
    if bias.ok:
        st.markdown("<h4>🛡️ Responsible AI — ASR corrections by language</h4>"
                    "<p class='vca-sub' style='font-size:.8rem'>rising counts flag "
                    "recognition bias for that language or accent</p>", unsafe_allow_html=True)
        st.table(bias.json()["corrections_by_language"] or
                 [{"language": "—", "corrections": 0}])


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    st.markdown(CSS, unsafe_allow_html=True)
    me = login_gate()

    with st.sidebar:
        st.markdown(f"""
<div style='display:flex;align-items:center;gap:.75rem;padding:.3rem 0 .6rem 0'>
  <img src='{ORB}' width='46' style='border-radius:15px' class='vca-float'/>
  <div>
    <div class='vca-title' style='font-size:1.25rem'>Voice Code<br/>Assistant</div>
  </div>
</div>""", unsafe_allow_html=True)
        st.markdown(f"<div class='vca-user'><div class='av'>{me['username'][0].upper()}</div>"
                    f"<div><b>{me['username']}</b><br/>"
                    f"<span class='vca-pill violet' style='margin:0'>{me['role']}</span></div></div>",
                    unsafe_allow_html=True)
        if st.button("⎋ Log out", use_container_width=True):
            st.session_state.clear()
            st.rerun()
        st.markdown("<hr class='vca-hr'/>", unsafe_allow_html=True)
        health = api("GET", "/api/health").json()
        st.markdown("<b style='font-family:Sora'>⚡ LLM chain</b><br/>" +
                    "".join(f"<span class='vca-pill cyan'>{p}</span>"
                            for p in health["llm_providers"]),
                    unsafe_allow_html=True)
        st.markdown("<b style='font-family:Sora'>📦 Indexed</b><br/>" +
                    ("".join(f"<span class='vca-pill violet'>{r}</span>"
                             for r in health["repos"]) or
                     "<span class='vca-pill'>none</span>"),
                    unsafe_allow_html=True)
        st.markdown("<hr class='vca-hr'/>", unsafe_allow_html=True)
        st.markdown(f"<img src='{WAVE}' style='width:100%;border-radius:14px;"
                    f"opacity:.85;border:1px solid var(--line)'/>", unsafe_allow_html=True)

    tabs = st.tabs(["💬  Assistant", "🧭  Code Navigator", "🕸️  Knowledge Graph",
                    "🤟  Sign Language", "✨  Personalization", "📊  Observability"])
    with tabs[0]:
        assistant_tab(me)
    with tabs[1]:
        navigator_tab(me)
    with tabs[2]:
        graph_tab(me)
    with tabs[3]:
        sign_tab()
    with tabs[4]:
        profile_tab(me)
    with tabs[5]:
        observability_tab()


main()
