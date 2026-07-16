"""FastAPI backend for the voice code assistant.

Auth is OAuth2+JWT; every content endpoint is role-gated (see app/auth.py).
/api/ask streams Server-Sent Events so the UI renders tokens as they arrive
and TTS can begin before the answer is complete.
"""
import json
import tarfile
import zipfile
from pathlib import Path

import requests
from fastapi import Depends, FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.security import OAuth2PasswordRequestForm
from loguru import logger
from pydantic import BaseModel

from app import auth, db, llm, pipeline, rag, stt, tts
from app.auth import current_user, require
from app.config import BACKEND_HOST, BACKEND_PORT
from app.indexer import index_repository
from app.intel import embeddings, graphify
from app.logging_setup import setup_logging
from app.observability import Trace, metrics_snapshot

setup_logging()
app = FastAPI(title="Voice Code Assistant", version="1.0")


@app.on_event("startup")
def _startup() -> None:
    db.get_conn()
    auth.ensure_default_users()
    logger.info("backend ready; llm providers: {}", llm.available_providers())


# ── auth ──────────────────────────────────────────────────────────

@app.post("/api/auth/token")
def token(form: OAuth2PasswordRequestForm = Depends()):
    user = auth.authenticate(form.username, form.password)
    if not user:
        raise HTTPException(401, "bad username or password")
    return {"access_token": auth.create_token(user["username"], user["role"]),
            "token_type": "bearer", "role": user["role"]}


@app.get("/api/me")
def me(user: dict = Depends(current_user)):
    return {"username": user["username"], "role": user["role"],
            "capabilities": sorted(user["capabilities"]),
            "profile": db.get_profile(user["username"])}


class ProfileUpdate(BaseModel):
    language: str | None = None
    tts_voice: str | None = None
    speech_rate: float | None = None
    vocabulary: str | None = None
    answer_style: str | None = None


@app.put("/api/profile")
def update_profile(body: ProfileUpdate, user: dict = Depends(current_user)):
    db.update_profile(user["username"],
                      **{k: v for k, v in body.model_dump().items() if v is not None})
    return db.get_profile(user["username"])


# ── repositories / indexing ───────────────────────────────────────

class IndexRequest(BaseModel):
    path: str = ""
    url: str = ""
    name: str = ""


@app.post("/api/index")
def index_repo(body: IndexRequest, user: dict = Depends(require("index_repo"))):
    try:
        path = body.path
        if body.url:
            from app import importer
            path = importer.import_url(body.url, name=body.name)
        if not path:
            raise HTTPException(400, "provide a local path or a url")
        return index_repository(path, user=user["username"])
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except requests.RequestException as exc:
        raise HTTPException(502, f"fetch failed: {exc}")


@app.post("/api/index/upload")
async def index_upload(files: list[UploadFile], name: str = Form(""),
                       user: dict = Depends(require("index_repo"))):
    """Import uploaded archive(s)/source files and index them as a repo."""
    from app import importer
    payload = [(f.filename or "file", await f.read()) for f in files]
    if not payload:
        raise HTTPException(400, "no files uploaded")
    try:
        path = importer.import_files(name, payload)
        return index_repository(path, user=user["username"])
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except (zipfile.BadZipFile, tarfile.TarError) as exc:
        raise HTTPException(400, f"could not read archive: {exc}")


@app.get("/api/repos")
def repos(user: dict = Depends(current_user)):
    return {"repos": sorted(set(embeddings.indexed_repos()) | set(graphify.list_graphs()))}


# ── sessions (memory) ─────────────────────────────────────────────

class SessionCreate(BaseModel):
    repo: str = ""


@app.post("/api/sessions")
def new_session(body: SessionCreate, user: dict = Depends(current_user)):
    return {"session_id": db.create_session(user["username"], repo=body.repo)}


@app.get("/api/sessions")
def sessions(user: dict = Depends(current_user)):
    return {"sessions": db.list_sessions(user["username"])}


@app.get("/api/sessions/{session_id}/messages")
def session_messages(session_id: str, user: dict = Depends(current_user)):
    sess = db.get_session(session_id)
    if not sess or sess["username"] != user["username"]:
        raise HTTPException(404, "session not found")
    return {"messages": db.get_messages(session_id)}


# ── ask (SSE streaming) ───────────────────────────────────────────

class AskRequest(BaseModel):
    question: str
    session_id: str
    repo: str


@app.post("/api/ask")
def ask(body: AskRequest, user: dict = Depends(require("ask"))):
    trace = Trace(kind="ask", user=user["username"])

    def sse():
        try:
            for event in pipeline.answer_stream(
                    body.question, body.session_id, user, body.repo, trace):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:   # keep the stream well-formed on any failure
            logger.exception("ask stream failed")
            yield f"data: {json.dumps({'type': 'error', 'text': str(exc)})}\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


# ── speech ────────────────────────────────────────────────────────

@app.post("/api/stt")
async def speech_to_text(file: UploadFile, user: dict = Depends(current_user)):
    profile = db.get_profile(user["username"])
    audio = await file.read()
    if not audio:
        raise HTTPException(400, "empty audio")
    trace = Trace(kind="stt", user=user["username"])
    with trace.stage("stt"):
        try:
            result = stt.transcribe(audio, filename=file.filename or "audio.wav",
                                    language=profile.get("language", ""),
                                    vocabulary=profile.get("vocabulary", ""))
        except Exception as exc:
            trace.finish()
            raise HTTPException(502, f"STT failed: {exc}")
    trace.meta.update({"language": result["language"],
                       "confidence": result["confidence"]})
    trace.finish()
    result.pop("segments", None)
    return result


class FeedbackRequest(BaseModel):
    heard: str
    corrected: str
    language: str = ""


@app.post("/api/asr_feedback")
def asr_feedback(body: FeedbackRequest, user: dict = Depends(current_user)):
    db.add_asr_feedback(user["username"], body.heard, body.corrected, body.language)
    return {"ok": True}


class TTSRequest(BaseModel):
    text: str
    voice: str = ""
    speed: float = 0.0


@app.post("/api/tts")
def text_to_speech(body: TTSRequest, user: dict = Depends(current_user)):
    profile = db.get_profile(user["username"])
    trace = Trace(kind="tts", user=user["username"])
    with trace.stage("tts"):
        try:
            wav = tts.synthesize_wav(
                body.text, voice=body.voice or profile.get("tts_voice", ""),
                speed=body.speed or float(profile.get("speech_rate", 1.0)))
        except Exception as exc:
            trace.finish()
            raise HTTPException(502, f"TTS failed: {exc}")
    trace.finish()
    return Response(content=wav, media_type="audio/wav")


# ── live voice mode (wake word in the UI) ─────────────────────────

class LiveStartRequest(BaseModel):
    repo: str
    session_id: str = ""


@app.post("/api/live/start")
def live_start(body: LiveStartRequest, user: dict = Depends(require("ask"))):
    from voice.live_service import live_service
    session_id = body.session_id or db.create_session(user["username"], repo=body.repo)
    ok, msg = live_service.start(user, body.repo, session_id)
    return {"ok": ok, "message": msg, "session_id": session_id,
            **live_service.status()}


@app.post("/api/live/stop")
def live_stop(user: dict = Depends(current_user)):
    from voice.live_service import live_service
    live_service.stop()
    return live_service.status()


@app.post("/api/live/trigger")
def live_trigger(user: dict = Depends(require("ask"))):
    from voice.live_service import live_service
    live_service.trigger_listen()
    return live_service.status()


@app.get("/api/live/status")
def live_status(user: dict = Depends(current_user)):
    from voice.live_service import live_service
    return live_service.status()


# ── code navigation & graph ───────────────────────────────────────

@app.get("/api/file")
def get_file(repo: str, path: str, start: int = 1, end: int = 0,
             user: dict = Depends(require("view_code"))):
    if not auth.path_allowed(user, path):
        raise HTTPException(403, f"role '{user['role']}' may not view {path}")
    g = rag.get_graph(repo)
    if g is None:
        raise HTTPException(404, f"repo '{repo}' is not indexed")
    root = Path(g.graph.get("root", ""))
    target = (root / path).resolve()
    if root not in target.parents and target != root:
        raise HTTPException(400, "path escapes repository root")
    if not target.is_file():
        raise HTTPException(404, f"file not found: {path}")
    lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    end = end or len(lines)
    start = max(1, start)
    return {"path": path, "start": start, "end": min(end, len(lines)),
            "total_lines": len(lines),
            "content": "\n".join(lines[start - 1:end])}


@app.get("/api/graph/search")
def graph_search(repo: str, q: str, user: dict = Depends(require("view_graph"))):
    g = rag.get_graph(repo)
    if g is None:
        raise HTTPException(404, f"repo '{repo}' is not indexed")
    nodes = [n for n in graphify.find_nodes(g, q)
             if auth.path_allowed(user, n.get("path", ""))]
    return {"nodes": nodes}


@app.get("/api/graph/summary")
def graph_summary(repo: str, user: dict = Depends(require("view_graph"))):
    g = rag.get_graph(repo)
    if g is None:
        raise HTTPException(404, f"repo '{repo}' is not indexed")
    return {"nodes": g.number_of_nodes(), "edges": g.number_of_edges(),
            "important": graphify.important_nodes(g)}


@app.get("/api/graph/html", response_class=HTMLResponse)
def graph_html(repo: str, highlight: str = "",
               user: dict = Depends(require("view_graph"))):
    g = rag.get_graph(repo)
    if g is None:
        raise HTTPException(404, f"repo '{repo}' is not indexed")
    ids = [h for h in highlight.split(",") if h]
    path = graphify.export_html(g, highlight=ids)
    return path.read_text(encoding="utf-8")


# ── observability ─────────────────────────────────────────────────

@app.get("/api/metrics")
def metrics(user: dict = Depends(require("metrics"))):
    return metrics_snapshot()


@app.get("/api/asr_bias")
def asr_bias(user: dict = Depends(require("metrics"))):
    return {"corrections_by_language": db.asr_feedback_stats()}


@app.get("/api/health")
def health():
    return {"status": "ok", "llm_providers": llm.available_providers(),
            "repos": embeddings.indexed_repos()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=BACKEND_HOST, port=BACKEND_PORT, reload=False)
