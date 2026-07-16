"""Central configuration.

Everything is driven by .env at the project root plus sane defaults, so the
whole stack runs on a laptop with only API keys configured. Paths default to
the local filesystem (data/ subfolders) per the storage requirement.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# ── Storage (local filesystem) ────────────────────────────────────
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
CHROMA_DIR = DATA_DIR / "chroma"
GRAPH_DIR = DATA_DIR / "graphs"
LOG_DIR = DATA_DIR / "logs"
TRACE_DIR = LOG_DIR / "traces"
MODELS_DIR = DATA_DIR / "models"          # downloaded VAD / wake-word models
DB_PATH = DATA_DIR / "app.db"             # SQLite: users, sessions, memory

for _d in (DATA_DIR, CHROMA_DIR, GRAPH_DIR, LOG_DIR, TRACE_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── API keys (from .env) ──────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "").strip()
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "").strip()

# ── LLM provider chain (first available wins, falls through on error) ─
LLM_PROVIDERS = [
    {  # primary: fast + cheap + reliable
        "name": "openai",
        "base_url": "https://api.openai.com/v1",
        "api_key": OPENAI_API_KEY,
        "model": os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
    },
    {  # fallback 1: free tier, very fast
        "name": "groq",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key": GROQ_API_KEY,
        "model": os.getenv("GROQ_CHAT_MODEL", "llama-3.3-70b-versatile"),
    },
    {  # fallback 2
        "name": "mistral",
        "base_url": "https://api.mistral.ai/v1",
        "api_key": MISTRAL_API_KEY,
        "model": os.getenv("MISTRAL_CHAT_MODEL", "mistral-small-latest"),
    },
    {  # fallback 3 (OpenAI-compatible Gemini endpoint)
        "name": "google",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "api_key": GOOGLE_API_KEY,
        "model": os.getenv("GOOGLE_CHAT_MODEL", "gemini-2.0-flash"),
    },
]

# ── Speech ────────────────────────────────────────────────────────
STT_MODEL = os.getenv("STT_MODEL", "whisper-1")
TTS_MODEL = os.getenv("TTS_MODEL", "tts-1")
TTS_VOICE = os.getenv("TTS_VOICE", "alloy")
TTS_SAMPLE_RATE = 24_000                  # OpenAI TTS pcm output rate

# ── Embeddings / retrieval ────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "8"))
GRAPH_HOPS = int(os.getenv("GRAPH_HOPS", "1"))

# ── Auth ──────────────────────────────────────────────────────────
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-me")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "480"))

# Role -> capabilities + repo path scopes. "paths" are glob patterns relative
# to the indexed repo root; retrieval results outside a user's scope are
# dropped BEFORE they ever reach the LLM, so unauthorized code cannot leak
# into an answer.
ROLES = {
    "admin": {
        "capabilities": {"ask", "view_code", "view_graph", "index_repo", "metrics", "manage_users"},
        "paths": ["*"],
    },
    "developer": {
        "capabilities": {"ask", "view_code", "view_graph", "index_repo"},
        "paths": ["*"],
    },
    "viewer": {  # may ask questions but never sees raw source of restricted areas
        "capabilities": {"ask", "view_graph"},
        "paths": ["docs/*", "README*", "*.md"],
    },
}

# ── Voice front-end ───────────────────────────────────────────────
WAKEWORD_MODEL = os.getenv("WAKEWORD_MODEL", "hey_jarvis")   # openwakeword pretrained
WAKEWORD_THRESHOLD = float(os.getenv("WAKEWORD_THRESHOLD", "0.5"))
VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.5"))
MIC_SAMPLE_RATE = 16_000

# ── Services ──────────────────────────────────────────────────────
BACKEND_HOST = os.getenv("BACKEND_HOST", "127.0.0.1")
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))
BACKEND_URL = f"http://{BACKEND_HOST}:{BACKEND_PORT}"
SIGNLANG_PORT = int(os.getenv("SIGNLANG_PORT", "5055"))
SIGNLANG_URL = f"http://127.0.0.1:{SIGNLANG_PORT}"
SIGNLANG_PYTHON = os.getenv("SIGNLANG_PYTHON", r"D:\GenAI Prac\.venv\Scripts\python.exe")
