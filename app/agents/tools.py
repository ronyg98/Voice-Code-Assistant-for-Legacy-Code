"""Shared toolbox: ONE set of code-intelligence tools, exposed two ways.

    ┌─────────────────────┐        ┌──────────────────────────┐
    │  LangGraph agents   │──────▶ │                          │
    │  (app/agents/graph) │        │   app/agents/tools.py    │
    └─────────────────────┘        │   (this module)          │
    ┌─────────────────────┐        │                          │
    │  MCP server         │──────▶ │  search · graph · git ·  │
    │  (app/mcp_server)   │        │  files · indexing        │
    └─────────────────────┘        └──────────────────────────┘

The agents call these functions in-process; external MCP clients (Claude
Code, IDEs) call the same functions over the MCP protocol. One toolbox,
two transports - that's the MCP + Agents combination.
"""
from loguru import logger

from app import auth, rag
from app.intel import embeddings, graphify


def list_repos() -> dict:
    """Repositories with a vector index and/or knowledge graph."""
    return {"indexed": embeddings.indexed_repos(), "graphs": graphify.list_graphs()}


def search_code(repo: str, query: str, top_k: int = 8,
                user: dict | None = None) -> list[dict]:
    """Semantic code search. If a user is given, results are RBAC-filtered
    BEFORE they are returned (unauthorized code never reaches an agent)."""
    hits = embeddings.query(repo, query, top_k=top_k)
    if user is not None:
        hits, dropped = auth.filter_hits(user, hits)
        if dropped:
            logger.info("toolbox RBAC dropped {} hits for role {}",
                        dropped, user.get("role"))
    return hits


def graph_find(repo: str, term: str, limit: int = 10,
               user: dict | None = None) -> list[dict]:
    """Find knowledge-graph nodes (files/classes/functions) by name."""
    g = rag.get_graph(repo)
    if g is None:
        return []
    nodes = graphify.find_nodes(g, term, limit=limit)
    if user is not None:
        nodes = [n for n in nodes if auth.path_allowed(user, n.get("path", ""))]
    return nodes


def graph_neighbors(repo: str, node_ids: list[str], hops: int = 1,
                    user: dict | None = None) -> dict:
    """Structural neighborhood (contains/imports/calls/inherits) of nodes."""
    g = rag.get_graph(repo)
    if g is None:
        return {"nodes": [], "edges": []}
    hood = graphify.neighborhood(g, node_ids, hops=hops)
    if user is not None:
        allowed = {n["id"] for n in hood["nodes"]
                   if auth.path_allowed(user, n.get("path", ""))}
        hood["nodes"] = [n for n in hood["nodes"] if n["id"] in allowed]
        hood["edges"] = [e for e in hood["edges"]
                         if e["source"] in allowed and e["target"] in allowed]
    return hood


def describe_nodes(repo: str, node_ids: list[str],
                   user: dict | None = None) -> list[str]:
    """One-line LLM-ready descriptions of graph nodes and their relations."""
    g = rag.get_graph(repo)
    if g is None:
        return []
    out = []
    for nid in node_ids:
        if g.has_node(nid) and (user is None or
                                auth.path_allowed(user, g.nodes[nid].get("path", ""))):
            out.append(graphify.describe_node(g, nid))
    return out


def repo_overview(repo: str) -> dict:
    """High-level shape of a repo: size, hub files/classes by centrality."""
    g = rag.get_graph(repo)
    if g is None:
        return {"error": f"repo '{repo}' is not indexed"}
    return {"repo": repo, "nodes": g.number_of_nodes(),
            "edges": g.number_of_edges(),
            "hubs": [n["label"] for n in graphify.important_nodes(g, limit=10)]}
