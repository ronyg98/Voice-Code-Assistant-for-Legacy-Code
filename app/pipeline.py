"""Ask-pipeline orchestration - now a LangGraph multi-agent system.

`answer_stream` keeps its original event contract (consumed by the FastAPI
SSE endpoint, the Streamlit UI, and the live voice loop):

  {"type": "status", "text": ..., "agent": ...}   agent activity, live
  {"type": "token",  "text": ...}                 streamed answer tokens
  {"type": "final",  "answer", "citations", "confidence", "graph_nodes",
                     "provider", "trace_id", "stages", "agents"}

Internally the work is done by the agent graph in app/agents/graph.py
(planner → retriever → graph analyst → synthesizer → critic). The graph runs
in a worker thread; agents push events through an emit-queue so tokens reach
TTS/UI while later agents are still working.
"""
import json
import queue
import threading
from collections.abc import Iterator

from loguru import logger

from app import db
from app.agents.graph import agent_graph
from app.observability import Trace

MEMORY_TURNS = 8
_DONE = object()


def answer_stream(question: str, session_id: str, user: dict,
                  repo: str, trace: Trace) -> Iterator[dict]:
    trace.meta.update({"repo": repo, "question": question[:200]})
    profile = db.get_profile(user["username"])
    session = db.get_session(session_id)
    entities = json.loads(session["entities"]) if session else []
    history = [{"role": m["role"], "content": m["content"]}
               for m in (db.get_messages(session_id, limit=MEMORY_TURNS * 2)
                         if session else [])]

    events: queue.Queue = queue.Queue()
    final_state: dict = {}

    def run_graph():
        try:
            state = {
                "question": question, "repo": repo, "user": user,
                "style": profile.get("answer_style", "concise"),
                "entities": entities, "history": history,
                "emit": events.put, "trace": trace,
            }
            final_state.update(agent_graph.invoke(state))
        except Exception as exc:
            logger.bind(trace_id=trace.trace_id).exception("agent graph failed")
            events.put({"type": "error", "text": str(exc)})
        finally:
            events.put(_DONE)

    threading.Thread(target=run_graph, daemon=True,
                     name=f"agents-{trace.trace_id}").start()

    failed = False
    while (event := events.get()) is not _DONE:
        if event.get("type") == "error":
            failed = True
        yield event

    if failed or not final_state:
        yield {"type": "final",
               "answer": "Sorry - the agent pipeline failed. Check the logs.",
               "citations": [], "confidence": {"score": 0, "band": "none",
                                               "reason": "agent error"},
               "graph_nodes": [], "provider": "none", "agents": [],
               "trace_id": trace.trace_id, "stages": trace.stages}
        trace.finish()
        return

    answer = final_state.get("answer", "")
    citations = final_state.get("citations", [])
    confidence = final_state.get("confidence", {})
    trace.answer = answer
    trace.meta.update({"provider": final_state.get("provider"),
                       "confidence": confidence.get("score"),
                       "route": final_state.get("route"),
                       "n_citations": len(citations)})

    # session memory: store the turn + entities for follow-ups
    if session:
        db.add_message(session_id, "user", question)
        db.add_message(session_id, "assistant", answer, citations=citations,
                       confidence=confidence.get("score"))
        new_entities = entities + [c["symbol"] or c["path"] for c in citations]
        title = session["title"] if session["title"] != "New session" else question[:60]
        db.touch_session(session_id, repo=repo, title=title,
                         entities=list(dict.fromkeys(new_entities))[-20:])

    summary = trace.finish()
    yield {"type": "final", "answer": answer, "citations": citations,
           "confidence": confidence,
           "graph_nodes": final_state.get("graph_nodes", []),
           "provider": final_state.get("provider", "?"),
           "agents": final_state.get("agent_steps", []),
           "trace_id": trace.trace_id, "stages": summary["stages"]}
