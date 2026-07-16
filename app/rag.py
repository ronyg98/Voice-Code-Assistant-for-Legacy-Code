"""Retrieval: vector search + knowledge-graph expansion + RBAC filtering.

Produces the context block for the LLM plus citation records. Every context
snippet is numbered [1]..[n]; the model is instructed to cite those numbers,
so every answer links back to real code and graph evidence.

Confidence = f(top similarity scores, agreement, graph support). Surfaced to
the UI as a score + band so users can calibrate trust (responsible AI).
"""
from loguru import logger

from app import auth
from app.config import GRAPH_HOPS, RETRIEVAL_TOP_K
from app.intel import embeddings, graphify
from app.observability import Trace

_graph_cache: dict = {}


def get_graph(repo: str):
    if repo not in _graph_cache:
        _graph_cache[repo] = graphify.load_graph(repo)
    return _graph_cache[repo]


def invalidate_graph(repo: str) -> None:
    _graph_cache.pop(repo, None)


def retrieve(repo: str, question: str, user: dict, trace: Trace) -> dict:
    with trace.stage("retrieval_vector"):
        hits = embeddings.query(repo, question, top_k=RETRIEVAL_TOP_K)

    hits, dropped = auth.filter_hits(user, hits)
    if dropped:
        logger.bind(trace_id=trace.trace_id).info(
            "RBAC dropped {} retrieval hits for role {}", dropped, user["role"])
    trace.log_retrieval(question, hits)
    trace.meta["rbac_dropped"] = dropped

    graph_context, graph_nodes = [], []
    g = get_graph(repo)
    if g is not None:
        with trace.stage("retrieval_graph"):
            seed_ids = [h["node_id"] for h in hits if h.get("node_id")]
            # also match entities named directly in the question
            for term in _keywords(question):
                for node in graphify.find_nodes(g, term, limit=2):
                    if auth.path_allowed(user, node.get("path", "")):
                        seed_ids.append(node["id"])
            seed_ids = list(dict.fromkeys(seed_ids))[:12]
            hood = graphify.neighborhood(g, seed_ids, hops=GRAPH_HOPS)
            graph_nodes = [n["id"] for n in hood["nodes"]
                           if auth.path_allowed(user, n.get("path", ""))]
            for nid in seed_ids:
                if g.has_node(nid) and auth.path_allowed(user, g.nodes[nid].get("path", "")):
                    graph_context.append(graphify.describe_node(g, nid))

    confidence = _confidence(hits, bool(graph_context))
    return {
        "hits": hits,
        "graph_context": graph_context[:12],
        "graph_nodes": graph_nodes,
        "confidence": confidence,
        "rbac_dropped": dropped,
    }


def build_context_block(retrieval: dict) -> tuple[str, list[dict]]:
    """Numbered snippets for the prompt + citation records for the UI."""
    parts, citations = [], []
    for i, h in enumerate(retrieval["hits"], start=1):
        parts.append(f"[{i}] {h['path']} (lines {h['start']}-{h['end']}"
                     f"{', ' + h['symbol'] if h['symbol'] else ''})\n{h['text']}")
        citations.append({
            "n": i, "path": h["path"], "start": h["start"], "end": h["end"],
            "symbol": h["symbol"], "score": h["score"], "node_id": h["node_id"],
        })
    graph_part = ""
    if retrieval["graph_context"]:
        graph_part = ("\n\nKNOWLEDGE GRAPH RELATIONS (structure of the codebase):\n"
                      + "\n".join("- " + c for c in retrieval["graph_context"]))
    return "\n\n".join(parts) + graph_part, citations


def _confidence(hits: list[dict], has_graph: bool) -> dict:
    if not hits:
        return {"score": 0.0, "band": "none",
                "reason": "no relevant code found in the index"}
    top = [h["score"] for h in hits[:3]]
    score = sum(top) / len(top)
    if has_graph:
        score = min(1.0, score + 0.05)   # structural corroboration
    band = "high" if score >= 0.55 else ("medium" if score >= 0.40 else "low")
    return {"score": round(score, 3), "band": band,
            "reason": f"top-{len(top)} retrieval similarity avg {score:.2f}"
                      f"{' + graph support' if has_graph else ''}"}


def _keywords(question: str) -> list[str]:
    import re
    stop = {"what", "when", "where", "which", "does", "this", "that", "with",
            "from", "have", "how", "why", "who", "the", "and", "for", "are",
            "can", "you", "please", "explain", "show", "tell", "about", "more",
            "detail", "code", "file", "method", "class", "function", "service"}
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", question)
    return [w for w in words if w.lower() not in stop][:8]
