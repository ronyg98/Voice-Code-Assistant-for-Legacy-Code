"""MCP protocol layer - exposes the SAME toolbox the LangGraph agents use.

app/agents/tools.py is the single source of truth for code-intelligence
tools; the agents call it in-process, external MCP clients (Claude Code,
IDEs, other agents) call it over MCP. `ask_codebase` runs the full
multi-agent graph (planner → retriever → graph analyst → synthesizer →
critic) and returns the cited, confidence-scored result.

Run (stdio transport):
    .venv\\Scripts\\python.exe -m app.mcp_server

Claude Code registration:
    claude mcp add voice-code-assistant -- "D:\\Voice Assistant AI\\.venv\\Scripts\\python.exe" -m app.mcp_server
"""
import json

from mcp.server.fastmcp import FastMCP

from app.agents import tools
from app.config import ROLES
from app.indexer import index_repository
from app.logging_setup import setup_logging
from app.observability import Trace

setup_logging()
mcp = FastMCP("voice-code-assistant")

# MCP callers act with developer-level access (local trusted transport).
_MCP_USER = {"username": "mcp", "role": "developer", **ROLES["developer"]}


@mcp.tool()
def list_repos() -> str:
    """List repositories that have been indexed (vector index + knowledge graph)."""
    return json.dumps(tools.list_repos())


@mcp.tool()
def index_repo(path: str) -> str:
    """Analyze and index a legacy repository at a local path: GitNexus
    analysis -> Graphify knowledge graph -> embeddings into ChromaDB."""
    return json.dumps(index_repository(path, user="mcp"))


@mcp.tool()
def search_code(repo: str, query: str, top_k: int = 6) -> str:
    """Semantic search over an indexed repo. Returns snippets with paths,
    line ranges, similarity scores, and knowledge-graph node ids."""
    hits = tools.search_code(repo, query, top_k=top_k, user=_MCP_USER)
    for h in hits:
        h["text"] = h["text"][:1200]
    return json.dumps(hits, ensure_ascii=False)


@mcp.tool()
def graph_find(repo: str, term: str) -> str:
    """Find knowledge-graph nodes (files/classes/functions) matching a name."""
    return json.dumps(tools.graph_find(repo, term, user=_MCP_USER),
                      ensure_ascii=False)


@mcp.tool()
def graph_neighbors(repo: str, node_id: str, hops: int = 1) -> str:
    """Neighborhood (callers, callees, imports, containment) of a graph node."""
    return json.dumps(tools.graph_neighbors(repo, [node_id], hops=hops,
                                            user=_MCP_USER), ensure_ascii=False)


@mcp.tool()
def repo_overview(repo: str) -> str:
    """High-level shape of an indexed repo: size and hub files/classes."""
    return json.dumps(tools.repo_overview(repo), ensure_ascii=False)


@mcp.tool()
def ask_codebase(repo: str, question: str) -> str:
    """Ask a natural-language question about an indexed repo. Runs the full
    multi-agent pipeline; returns the answer with citations (file + line
    ranges), confidence score, and the agent steps that produced it."""
    from app import db, pipeline
    session_id = db.create_session("mcp", repo=repo)
    trace = Trace(kind="ask", user="mcp")
    final = {}
    for event in pipeline.answer_stream(question, session_id, _MCP_USER, repo, trace):
        if event["type"] == "final":
            final = event
    return json.dumps({"answer": final.get("answer", ""),
                       "citations": final.get("citations", []),
                       "confidence": final.get("confidence", {}),
                       "agents": final.get("agents", [])}, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run()
