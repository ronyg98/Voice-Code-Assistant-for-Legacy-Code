"""SQLite persistence (chosen over Redis: zero-install, free, Windows-friendly).

Holds users (auth), per-user personalization profiles, chat sessions with full
message history (session memory for follow-up questions), and ASR feedback
pairs (user-corrected transcripts) used to monitor recognition bias.
"""
import json
import sqlite3
import threading
import time
import uuid

from app.config import DB_PATH

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    username   TEXT PRIMARY KEY,
    pw_hash    BLOB NOT NULL,
    role       TEXT NOT NULL DEFAULT 'viewer',
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS profiles (          -- personalization
    username     TEXT PRIMARY KEY REFERENCES users(username),
    language     TEXT DEFAULT '',              -- preferred STT language hint ('' = auto)
    tts_voice    TEXT DEFAULT 'alloy',
    speech_rate  REAL DEFAULT 1.0,
    vocabulary   TEXT DEFAULT '',              -- domain terms fed to Whisper as prompt
    answer_style TEXT DEFAULT 'concise'
);
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    username   TEXT NOT NULL,
    repo       TEXT DEFAULT '',
    title      TEXT DEFAULT '',
    entities   TEXT DEFAULT '[]',              -- recently referenced graph nodes
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    role       TEXT NOT NULL,                  -- 'user' | 'assistant'
    content    TEXT NOT NULL,
    citations  TEXT DEFAULT '[]',
    confidence REAL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS asr_feedback (      -- responsible-AI: bias monitoring
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    username   TEXT NOT NULL,
    heard      TEXT NOT NULL,                  -- what the ASR transcribed
    corrected  TEXT NOT NULL,                  -- what the user says they said
    language   TEXT DEFAULT '',
    created_at REAL NOT NULL
);
"""


def get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.executescript(_SCHEMA)
        _conn.commit()
    return _conn


def _execute(sql: str, params: tuple = ()) -> sqlite3.Cursor:
    with _lock:
        cur = get_conn().execute(sql, params)
        get_conn().commit()
        return cur


def _query(sql: str, params: tuple = ()) -> list[sqlite3.Row]:
    with _lock:
        return get_conn().execute(sql, params).fetchall()


# ── users ─────────────────────────────────────────────────────────

def create_user(username: str, pw_hash: bytes, role: str) -> None:
    _execute("INSERT OR REPLACE INTO users VALUES (?,?,?,?)",
             (username, pw_hash, role, time.time()))
    _execute("INSERT OR IGNORE INTO profiles (username) VALUES (?)", (username,))


def get_user(username: str) -> dict | None:
    rows = _query("SELECT * FROM users WHERE username=?", (username,))
    return dict(rows[0]) if rows else None


def count_users() -> int:
    return _query("SELECT COUNT(*) n FROM users")[0]["n"]


# ── personalization profiles ──────────────────────────────────────

def get_profile(username: str) -> dict:
    rows = _query("SELECT * FROM profiles WHERE username=?", (username,))
    if not rows:
        _execute("INSERT OR IGNORE INTO profiles (username) VALUES (?)", (username,))
        rows = _query("SELECT * FROM profiles WHERE username=?", (username,))
    return dict(rows[0])


def update_profile(username: str, **fields) -> None:
    allowed = {"language", "tts_voice", "speech_rate", "vocabulary", "answer_style"}
    fields = {k: v for k, v in fields.items() if k in allowed}
    if not fields:
        return
    sets = ", ".join(f"{k}=?" for k in fields)
    _execute(f"UPDATE profiles SET {sets} WHERE username=?",
             (*fields.values(), username))


# ── sessions / memory ─────────────────────────────────────────────

def create_session(username: str, repo: str = "", title: str = "") -> str:
    sid = uuid.uuid4().hex[:16]
    now = time.time()
    _execute("INSERT INTO sessions VALUES (?,?,?,?,?,?,?)",
             (sid, username, repo, title or "New session", "[]", now, now))
    return sid


def get_session(session_id: str) -> dict | None:
    rows = _query("SELECT * FROM sessions WHERE session_id=?", (session_id,))
    return dict(rows[0]) if rows else None


def list_sessions(username: str) -> list[dict]:
    return [dict(r) for r in _query(
        "SELECT * FROM sessions WHERE username=? ORDER BY updated_at DESC LIMIT 50",
        (username,))]


def touch_session(session_id: str, repo: str | None = None,
                  title: str | None = None, entities: list | None = None) -> None:
    sess = get_session(session_id)
    if not sess:
        return
    _execute(
        "UPDATE sessions SET repo=?, title=?, entities=?, updated_at=? WHERE session_id=?",
        (repo if repo is not None else sess["repo"],
         title if title is not None else sess["title"],
         json.dumps(entities) if entities is not None else sess["entities"],
         time.time(), session_id))


def add_message(session_id: str, role: str, content: str,
                citations: list | None = None, confidence: float | None = None) -> None:
    _execute(
        "INSERT INTO messages (session_id, role, content, citations, confidence, created_at)"
        " VALUES (?,?,?,?,?,?)",
        (session_id, role, content, json.dumps(citations or []), confidence, time.time()))


def get_messages(session_id: str, limit: int = 50) -> list[dict]:
    rows = _query(
        "SELECT * FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
        (session_id, limit))
    out = []
    for r in reversed(rows):
        d = dict(r)
        d["citations"] = json.loads(d["citations"] or "[]")
        out.append(d)
    return out


# ── ASR feedback (bias monitoring) ────────────────────────────────

def add_asr_feedback(username: str, heard: str, corrected: str, language: str) -> None:
    _execute("INSERT INTO asr_feedback (username, heard, corrected, language, created_at)"
             " VALUES (?,?,?,?,?)", (username, heard, corrected, language, time.time()))


def asr_feedback_stats() -> list[dict]:
    return [dict(r) for r in _query(
        "SELECT language, COUNT(*) corrections FROM asr_feedback GROUP BY language")]
