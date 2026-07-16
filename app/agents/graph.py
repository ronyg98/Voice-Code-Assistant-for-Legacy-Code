"""Multi-agent answer pipeline built on LangGraph.

    START ─▶ 🧭 planner ──(needs codebase?)──▶ 📚 retriever ─▶ 🕸 graph_analyst
                   │                                                  │
                   └──(direct/meta)──▶ ✍️ synthesizer ◀───────────────┘
                                            │
                                        🔍 critic ─▶ END

Agents:
  planner        LLM: classifies the question, resolves follow-up references
                 ("that service") against session entities, and rewrites the
                 question into a standalone search query.
  retriever      toolbox: RBAC-filtered semantic search over ChromaDB.
  graph_analyst  toolbox: expands retrieval seeds through the knowledge
                 graph, adds structural relations and hub context.
  synthesizer    LLM: streams the final cited answer token by token.
  critic         verifies citation coverage and scores confidence.

State carries an `emit(event)` callback: nodes push {"type": "status"|"token"}
events into a queue the SSE/voice layer consumes, so agent activity and the
answer stream to the UI in real time. Tools are the same functions the MCP
server exposes (app/agents/tools.py).
"""
import json
import re
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from loguru import logger

from app import llm
from app.agents import tools
from app.config import GRAPH_HOPS, RETRIEVAL_TOP_K


class AgentState(TypedDict, total=False):
    # inputs
    question: str
    repo: str
    user: dict
    style: str
    entities: list
    history: list          # prior chat turns [{role, content}]
    emit: Any              # callable(event: dict) -> None
    trace: Any             # observability Trace
    # planner outputs
    route: str             # "codebase" | "direct"
    search_query: str
    # retrieval outputs
    hits: list
    graph_context: list
    graph_nodes: list
    rbac_dropped: int
    # synthesis outputs
    answer: str
    provider: str
    citations: list
    confidence: dict
    agent_steps: list


PLANNER_PROMPT = """You are the planner agent of a code assistant answering questions about \
an indexed software repository. Given a user question, the repository's key components, \
recent conversation, and recently-discussed code entities, output STRICT JSON:
{"route": "codebase" or "direct", "search_query": "..."}

- route "codebase": anything that could be answered from the repository - its code, \
architecture, behavior, history, or DOMAIN. If the question mentions concepts related to \
the repository's components (listed in context), it is about the codebase even if it \
sounds like a general question. When in doubt, choose "codebase".
- route "direct": ONLY greetings, small talk, or questions about you, the assistant.
- search_query: the question rewritten as a standalone search query. Resolve references \
like "that service" or "it" using the entities/conversation. Keep code identifiers intact."""


def _step(state: AgentState, agent: str, text: str) -> None:
    state["emit"]({"type": "status", "agent": agent, "text": f"{agent}: {text}"})
    state.setdefault("agent_steps", []).append({"agent": agent, "text": text})
    logger.bind(trace_id=state["trace"].trace_id).info("[{}] {}", agent, text)


# Heuristic fast-path: most questions don't need an LLM call to route.
_GREETING_RE = re.compile(
    r"^(hi|hello|hey|yo|thanks?|thank you|good (?:morning|afternoon|evening))\b[\s!,.]*",
    re.I)
_ABOUT_ASSISTANT_RE = re.compile(
    r"\b(what can you do|who are you|your name|how do (?:i|you) use)\b", re.I)
_COREF_RE = re.compile(
    r"\b(it|its|that|this|these|those|they|them|same|again|previous|earlier"
    r"|above|more detail)\b", re.I)


def _fast_plan(state: AgentState) -> tuple[str, str, str] | None:
    """Decide route/query without the LLM when it's safe to. Returns
    (route, search_query, reason) or None to fall through to the LLM."""
    q = state["question"].strip()
    m = _GREETING_RE.match(q)
    rest = q[m.end():].strip() if m else q
    if (_ABOUT_ASSISTANT_RE.search(q) and len(q) < 80) or \
            (m and len(rest.split()) < 2):
        return "direct", q, "small talk"
    # follow-up references only need LLM resolution when there IS a past
    # conversation to resolve them against
    has_context = bool(state.get("history") or state.get("entities"))
    if not has_context or not _COREF_RE.search(q):
        return "codebase", rest or q, "self-contained"
    return None


# ── nodes ─────────────────────────────────────────────────────────

def planner(state: AgentState) -> dict:
    with state["trace"].stage("agent_planner"):
        fast = _fast_plan(state)
        if fast:   # no LLM call - instant routing for low-latency voice mode
            route, search_query, reason = fast
            _step(state, "🧭 planner",
                  f"fast-path ({reason}) · route={route} · query “{search_query[:60]}”")
            return {"route": route, "search_query": search_query,
                    "agent_steps": state["agent_steps"]}
        question = state["question"]
        context_bits = []
        overview = tools.repo_overview(state["repo"])
        if overview.get("hubs"):   # ground routing in what the repo contains
            context_bits.append(
                f"Repository '{state['repo']}' key components: "
                + ", ".join(overview["hubs"]))
        if state.get("entities"):
            context_bits.append("Recently discussed: " + ", ".join(state["entities"][-8:]))
        for m in state.get("history", [])[-4:]:
            context_bits.append(f"{m['role']}: {m['content'][:150]}")
        route, search_query, provider = "codebase", question, "fallback"
        try:
            # groq pinned: llama-3.3 turns this tiny prompt around ~3x faster
            raw, provider = llm.chat(
                [{"role": "system", "content": PLANNER_PROMPT},
                 {"role": "user", "content": "\n".join(context_bits + [f"Question: {question}"])}],
                temperature=0.0, max_tokens=150, prefer="groq")
            plan = json.loads(re.search(r"\{.*\}", raw, re.S).group())
            route = plan.get("route", "codebase")
            search_query = (plan.get("search_query") or question).strip()
        except Exception as exc:
            logger.warning("planner fallback (heuristic): {}", exc)
    _step(state, "🧭 planner",
          f"route={route} · query “{search_query[:70]}” ({provider})")
    return {"route": route, "search_query": search_query,
            "agent_steps": state.get("agent_steps", [])}


def route_after_planner(state: AgentState) -> str:
    return "retriever" if state["route"] == "codebase" else "synthesizer"


def retriever(state: AgentState) -> dict:
    with state["trace"].stage("agent_retriever"):
        hits = tools.search_code(state["repo"], state["search_query"],
                                 top_k=RETRIEVAL_TOP_K, user=state["user"])
        state["trace"].log_retrieval(state["search_query"], hits)
    _step(state, "📚 retriever", f"{len(hits)} snippets from vector index")
    return {"hits": hits, "agent_steps": state["agent_steps"]}


def graph_analyst(state: AgentState) -> dict:
    with state["trace"].stage("agent_graph"):
        seeds = [h["node_id"] for h in state.get("hits", []) if h.get("node_id")]
        for term in _keywords(state["search_query"]):
            seeds.extend(n["id"] for n in
                         tools.graph_find(state["repo"], term, limit=2, user=state["user"]))
        seeds = list(dict.fromkeys(seeds))[:12]
        hood = tools.graph_neighbors(state["repo"], seeds, hops=GRAPH_HOPS,
                                     user=state["user"])
        context = tools.describe_nodes(state["repo"], seeds, user=state["user"])
    _step(state, "🕸 graph analyst",
          f"{len(seeds)} seeds → {len(hood['nodes'])} related nodes")
    return {"graph_context": context[:12],
            "graph_nodes": [n["id"] for n in hood["nodes"]],
            "agent_steps": state["agent_steps"]}


SYNTH_PROMPT = """You are a voice-first AI code assistant for legacy codebases. You answer \
questions about the indexed repository using ONLY the numbered code snippets and knowledge-graph \
relations provided as context.

Rules:
- Cite evidence inline with bracketed numbers like [1] or [2][3] after every claim that comes \
from the code. Never invent citations.
- If the context does not contain the answer, say so plainly. Do not guess.
- Answers are often SPOKEN ALOUD: prefer clear short sentences, no markdown tables, minimal \
code excerpts - name files and methods instead, unless code is asked for.
- Resolve follow-up references using the conversation history and recently-discussed entities.
- Answer style requested by this user: {style}."""


def synthesizer(state: AgentState) -> dict:
    _step(state, "✍️ synthesizer", "composing cited answer")
    citations, context_block = [], "(no code context - answer directly)"
    if state.get("hits"):
        parts = []
        for i, h in enumerate(state["hits"], start=1):
            parts.append(f"[{i}] {h['path']} (lines {h['start']}-{h['end']}"
                         f"{', ' + h['symbol'] if h.get('symbol') else ''})\n{h['text']}")
            citations.append({"n": i, "path": h["path"], "start": h["start"],
                              "end": h["end"], "symbol": h.get("symbol", ""),
                              "score": h.get("score"), "node_id": h.get("node_id")})
        context_block = "\n\n".join(parts)
        if state.get("graph_context"):
            context_block += ("\n\nKNOWLEDGE GRAPH RELATIONS:\n"
                              + "\n".join("- " + c for c in state["graph_context"]))

    messages = [{"role": "system",
                 "content": SYNTH_PROMPT.format(style=state.get("style", "concise"))}]
    for m in state.get("history", []):
        messages.append({"role": m["role"], "content": m["content"][:2000]})
    entity_line = (f"RECENTLY DISCUSSED ENTITIES: {', '.join(state['entities'][-8:])}\n\n"
                   if state.get("entities") else "")
    messages.append({"role": "user", "content":
                     f"REPOSITORY: {state['repo']}\n\nCONTEXT SNIPPETS:\n{context_block}\n\n"
                     f"{entity_line}QUESTION: {state['question']}"})
    state["trace"].prompt = json.dumps(messages, ensure_ascii=False)[:40000]

    parts, provider = [], "?"
    with state["trace"].stage("agent_synthesizer_llm"):
        for delta, provider in llm.chat_stream(messages):
            parts.append(delta)
            state["emit"]({"type": "token", "text": delta})
    return {"answer": "".join(parts), "provider": provider,
            "citations": citations, "agent_steps": state["agent_steps"]}


def critic(state: AgentState) -> dict:
    """Citation-coverage check + confidence scoring (no extra LLM latency)."""
    with state["trace"].stage("agent_critic"):
        answer = state.get("answer", "")
        hits = state.get("hits", [])
        cited_ns = {int(n) for n in re.findall(r"\[(\d{1,2})\]", answer)}
        valid = {c["n"] for c in state.get("citations", [])}
        bogus = cited_ns - valid
        cited = [c for c in state.get("citations", []) if c["n"] in cited_ns]
        if not cited:
            cited = state.get("citations", [])[:3]

        if not hits:
            confidence = {"score": 0.15 if state["route"] == "direct" else 0.0,
                          "band": "none", "reason": "no code evidence used"}
        else:
            top = [h["score"] for h in hits[:3]]
            score = sum(top) / len(top)
            if state.get("graph_context"):
                score = min(1.0, score + 0.05)
            if answer and not cited_ns:          # uncited answer -> cap it
                score = min(score, 0.35)
            if bogus:                            # invented citation numbers
                score = min(score, 0.3)
            band = "high" if score >= 0.55 else ("medium" if score >= 0.40 else "low")
            confidence = {"score": round(score, 3), "band": band,
                          "reason": f"retrieval avg {sum(top)/len(top):.2f}"
                                    f"{' + graph support' if state.get('graph_context') else ''}"
                                    f"{' · uncited answer' if not cited_ns else ''}"
                                    f"{' · invalid citations flagged' if bogus else ''}"}
    verdict = f"confidence {confidence['band']}"
    if bogus:
        verdict += f" · flagged invalid citations {sorted(bogus)}"
    _step(state, "🔍 critic", verdict)
    return {"citations": cited, "confidence": confidence,
            "graph_nodes": [c["node_id"] for c in cited if c.get("node_id")]
                           or state.get("graph_nodes", []),
            "agent_steps": state["agent_steps"]}


def _keywords(question: str) -> list[str]:
    stop = {"what", "when", "where", "which", "does", "this", "that", "with",
            "from", "have", "how", "why", "who", "the", "and", "for", "are",
            "can", "you", "please", "explain", "show", "tell", "about", "more",
            "detail", "code", "file", "method", "class", "function", "service"}
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", question)
    return [w for w in words if w.lower() not in stop][:8]


# ── graph assembly ────────────────────────────────────────────────

def build_agent_graph():
    g = StateGraph(AgentState)
    g.add_node("planner", planner)
    g.add_node("retriever", retriever)
    g.add_node("graph_analyst", graph_analyst)
    g.add_node("synthesizer", synthesizer)
    g.add_node("critic", critic)
    g.add_edge(START, "planner")
    g.add_conditional_edges("planner", route_after_planner,
                            {"retriever": "retriever", "synthesizer": "synthesizer"})
    g.add_edge("retriever", "graph_analyst")
    g.add_edge("graph_analyst", "synthesizer")
    g.add_edge("synthesizer", "critic")
    g.add_edge("critic", END)
    return g.compile()


agent_graph = build_agent_graph()
