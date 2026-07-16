# 🎙 Voice Code Assistant for Legacy Codebases

A voice-enabled AI assistant that answers questions about legacy codebases by
combining a **code knowledge graph** with **repository understanding** and
retrieval-augmented generation — built to work in the real world: diverse
speech, noisy rooms, interruptions, sign-language input, and role-aware
access control.

```
                         ┌────────────────────────────────────────────┐
 wake word (openWakeWord)│  Streamlit UI (8501)     Sign service (5055)│
 VAD (Silero, onnx)      │  chat · voice · graph ·  MediaPipe + RF     │
 denoise (noisereduce)   │  navigator · metrics     ASL letters/words  │
        │                └───────────┬────────────────────────────────┘
        ▼                            │ JWT (OAuth2 password flow)
 ┌─────────────┐            ┌────────▼─────────────────────────────────┐
 │ live voice  │  SSE/HTTP  │ FastAPI backend (8000)                   │
 │ assistant   ├───────────►│  /ask /stt /tts /graph /file /metrics    │
 │ (desktop)   │            │  RBAC filter → RAG → LLM chain (stream)  │
 └─────────────┘            └───┬──────────┬───────────┬───────────────┘
                                │          │           │
                     Whisper STT│   ChromaDB vectors    │ OpenAI TTS
                     (OpenAI)   │   (text-embedding-    │ (streamed,
                                │    3-large)           │  sentence-
                          ┌─────▼──────────────────┐    │  pipelined)
                          │ GitNexus repo analysis │    │
                          │  → Graphify knowledge  │    │
                          │    graph (networkx)    │    │
                          └────────────────────────┘    ▼
                             MCP server exposes it all as tools
```

## Feature checklist

| Requirement | Where |
|---|---|
| Speech diversity (accents, multilingual) | Whisper auto language detect + per-user language hint & custom vocabulary priming (`app/stt.py`, Personalization tab) |
| Sign language on the UI | `services/signlang/` (ported from SignSpeak, `D:\GenAI Prac`) embedded as a Streamlit tab; signed sentences can be sent to the assistant |
| Real-world noise / interruptions | `voice/denoise.py` (noisereduce), Silero VAD, barge-in during TTS playback (`voice/live_assistant.py`) |
| Responsible AI / ASR bias | STT confidence + "did I get that right?" correction loop; corrections logged per language (`/api/asr_bias`, Observability tab) |
| Personalization | Per-user profile: language, TTS voice, speech rate, vocabulary, answer style (SQLite) |
| Multimodal fallback | Text chat, push-to-talk mic, sign language, visual citations/graph |
| Knowledge graph + repo understanding | `app/intel/gitnexus.py` (analysis) → `app/intel/graphify.py` (graph) → context → LLM |
| Wake word | openWakeWord `hey_jarvis` pretrained (set `WAKEWORD_MODEL` for a custom "Hey Assistant" model) |
| Streaming STT/TTS | SSE token stream; TTS starts speaking after the first sentence (`app/tts.py:stream_sentences`) |
| Session memory | SQLite sessions + recently-discussed entities → follow-ups like "explain that service in more detail" |
| Code navigation | Citations carry file + line ranges + graph node ids; Navigator tab shows the code, Graph tab highlights cited nodes in red |
| Role-aware access control | JWT + role capabilities + per-role path scopes; unauthorized code is filtered **before** the LLM sees it |
| Confidence + citations | Retrieval-similarity confidence bands; inline `[n]` citations resolved to code evidence |
| Observability | Loguru structured JSONL logs, per-stage latency metrics, retrieval diagnostics, full prompt traces (`data/logs/`) |
| Protocol layer (MCP) | `app/mcp_server.py`: `search_code`, `graph_find`, `graph_neighbors`, `ask_codebase`, `index_repo` |

## Quickstart

Prereqs: keys in `.env` (already present: OpenAI, Groq, Mistral, Google);
Python 3.13 venv in `.venv` (already installed via `requirements.txt`); the
sign-language sidecar needs the Python 3.10 venv from `D:\GenAI Prac`
(MediaPipe does not support 3.13).

```powershell
.\run_all.ps1          # backend (8000) + sign service (5055) + UI (8501)
# or individually:
.\run_backend.ps1
.\run_signlang.ps1
.\run_ui.ps1
.\run_live.ps1         # hands-free wake-word assistant in the terminal
```

Open http://127.0.0.1:8501 and log in:

| account | password | role |
|---|---|---|
| `admin` | `admin123` | everything incl. metrics & user management |
| `dev` | `dev123` | ask, view code, view graph, index repos |
| `viewer` | `viewer123` | ask + graph only; **source access limited to docs/** |

> Change these with `scripts\create_user.py <user> <password> <role>`.

A demo legacy codebase (`sample_legacy/` — Python + Java "OrderFlow" order
management system with a simulated COBOL mainframe bridge) is already
indexed. Index your own repo from the UI (right panel) or:

```powershell
.venv\Scripts\python.exe scripts\index_repo.py D:\path\to\legacy\repo
```

Try asking (typed or spoken): *"What happens when an order is placed?"* then
*"Now explain that billing service in more detail."* — the follow-up resolves
through session memory. Check the **Code Navigator** and **Knowledge Graph**
tabs afterwards: cited lines and graph nodes are highlighted.

### Hands-free mode

```powershell
.\run_live.ps1 -Repo sample_legacy
```

Say **"hey jarvis"** → beep → ask your question. The answer streams to TTS
sentence-by-sentence (it starts speaking before the LLM finishes). Say the
wake word again while it talks to interrupt (barge-in). If openWakeWord can't
load, it falls back to push-to-talk (Enter).

### MCP (use the assistant from Claude Code or any MCP client)

```powershell
claude mcp add voice-code-assistant -- "D:\Voice Assistant AI\.venv\Scripts\python.exe" -m app.mcp_server
```

## How answering works — multi-agent architecture (LangGraph + MCP)

Indexing: **GitNexus** (`app/intel/gitnexus.py`) parses the repo (Python via
AST, other languages via pragmatic patterns, git history when available) →
**Graphify** (`app/intel/graphify.py`) builds the knowledge graph →
symbol-aware chunks are embedded with **text-embedding-3-large** into
**ChromaDB**. Repos can be indexed from a **local path**, an **uploaded
ZIP/tar/source files**, or a **GitHub/archive URL** (Repository card, or
`/api/index` + `/api/index/upload`).

Answering runs a **LangGraph agent graph** (`app/agents/graph.py`):

```
START ─▶ 🧭 planner ──(codebase?)──▶ 📚 retriever ─▶ 🕸 graph analyst
              │                                            │
              └────(direct)────▶ ✍️ synthesizer ◀──────────┘
                                       │
                                   🔍 critic ─▶ END
```

- **planner** routes the question. A heuristic fast-path decides instantly
  (~3 ms) for greetings and self-contained questions - the planner LLM only
  runs for follow-ups whose references need session context to resolve
  ("what if *it* is called twice" →
  "risks if MainframeBridge.push_ledger_entry is called twice"),
  keeping voice-mode latency low
- **retriever** does RBAC-filtered semantic search (unauthorized code never
  reaches any agent)
- **graph analyst** expands hits through the knowledge graph
- **synthesizer** (LLM chain: OpenAI → Groq → Mistral → Gemini, automatic
  failover) streams the cited answer token-by-token
- **critic** verifies citation coverage (flags invented `[n]`s, caps
  confidence for uncited answers) and scores confidence

Agent activity streams live into the UI, and each answer shows its agent
trace. **MCP + Agents combination:** the agents and the MCP server share one
toolbox (`app/agents/tools.py`) — external MCP clients get `search_code`,
`graph_find`, `graph_neighbors`, `repo_overview`, `index_repo`, and
`ask_codebase` (which runs the full agent graph).

*Note: GitNexus and Graphify exist publicly as a browser-based TypeScript
tool and unrelated packages — neither ships a usable Python API, so this
project implements both roles as internal modules under `app/intel/` with the
exact pipeline you'd get from those tools: repo analysis → knowledge graph →
LLM context.*

## Design decisions (free-tier, Windows-friendly)

- **SQLite over Redis** for session memory: zero-install, durable, free.
  Swap by reimplementing `app/db.py`.
- **noisereduce over RNNoise**: same role (noise suppression), pure Python,
  no native toolchain on Windows.
- **Silero VAD via onnxruntime** (no PyTorch): saves a ~2 GB install; the
  model auto-downloads to `data/models/`.
- **JWT/OAuth2 over Entra ID**: free and local; the interface in
  `app/auth.py` is the single place to swap in Entra-issued tokens.
- **Wake word**: openWakeWord's pretrained `hey_jarvis` (no free
  "Hey Assistant" model exists; openWakeWord supports training a custom one —
  drop it in and set `WAKEWORD_MODEL`).
- **Sign language runs as a sidecar** on Python 3.10 because MediaPipe
  doesn't support the main venv's 3.13. Models/code ported from
  `D:\GenAI Prac` into `services/signlang/`.

## Observability

- `data/logs/app.jsonl` — structured Loguru logs (rotated)
- `data/logs/traces/<trace_id>.json` — full prompt + retrieval diagnostics
  per request
- `/api/metrics` — p50/p95 latency per request kind and per pipeline stage
  (retrieval, llm, stt, tts, denoise), surfaced in the Observability tab
- `/api/asr_bias` — ASR correction counts by language (bias monitoring)

## Project structure

```
app/                 FastAPI backend
  intel/             gitnexus (analysis) · graphify (graph) · chunker · embeddings
  main.py            API endpoints (SSE /ask, stt, tts, graph, file, metrics)
  pipeline.py        memory → retrieval → streaming LLM orchestration
  rag.py             RBAC-filtered retrieval, confidence, citations
  auth.py db.py      JWT + roles · SQLite (users, profiles, sessions, feedback)
  stt.py tts.py      Whisper · OpenAI TTS (sentence-pipelined streaming)
  llm.py             provider fallback chain (OpenAI/Groq/Mistral/Gemini)
  mcp_server.py      MCP tools
  observability.py   traces, latency metrics
voice/               wake word · VAD · denoise · live assistant (barge-in)
ui/streamlit_app.py  web UI (login, chat, navigator, graph, sign, profile, metrics)
services/signlang/   sign-language sidecar (Python 3.10 + MediaPipe)
sample_legacy/       demo legacy codebase (OrderFlow)
scripts/             index_repo.py · create_user.py
data/                chroma / graphs / logs / models / app.db  (local storage)
```

## Verified end-to-end

`scratchpad` test suites exercised: indexing (11 files → 46-node graph →
26 chunks), SSE ask with citations + confidence, session-memory follow-up,
RBAC (viewer blocked from source, viewer answers cite docs only), graph
search/summary/HTML export, TTS→WAV, STT round-trip of that WAV (transcript
correct, confidence 0.73), Silero VAD on real speech (p=1.0), openWakeWord
load, denoiser, MCP tool registry, and headless Streamlit renders of every
tab against the live backend.
