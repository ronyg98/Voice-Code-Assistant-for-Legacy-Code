"""Graphify: builds and queries the code knowledge graph.

Consumes the GitNexus analysis and produces a networkx directed graph:

    repo ──contains──> file ──contains──> class ──contains──> method
    file ──imports──> file            (resolved within the repo)
    symbol/file ──calls──> symbol     (name-resolved, best effort)
    class ──inherits──> class

Graphs persist as node-link JSON on the local filesystem (data/graphs/) and
export to interactive pyvis HTML for the Streamlit graph view.
"""
import json
import re
from pathlib import Path

import networkx as nx
from loguru import logger

from app.config import GRAPH_DIR

EDGE_COLORS = {"contains": "#8a8f98", "imports": "#4c9be8",
               "calls": "#e8734c", "inherits": "#9b59b6"}
NODE_COLORS = {"repo": "#f1c40f", "file": "#4c9be8", "class": "#9b59b6",
               "function": "#2ecc71", "method": "#27ae60"}


def build_graph(analysis: dict) -> nx.DiGraph:
    g = nx.DiGraph(name=analysis["name"], root=analysis["root"])
    repo_id = f"repo:{analysis['name']}"
    g.add_node(repo_id, kind="repo", label=analysis["name"], path="")

    symbol_index: dict[str, list[str]] = {}   # simple name -> node ids
    module_index: dict[str, str] = {}         # import-able module name -> file node

    for f in analysis["files"]:
        fid = f"file:{f['path']}"
        g.add_node(fid, kind="file", label=f["path"], path=f["path"],
                   language=f["language"], loc=f["loc"],
                   commits=f.get("git", {}).get("commits", 0))
        g.add_edge(repo_id, fid, kind="contains")
        module_index[_module_name(f["path"])] = fid

        for s in f["symbols"]:
            sid = f"sym:{f['path']}::{s['qualname']}"
            g.add_node(sid, kind=s["kind"], label=s["qualname"], path=f["path"],
                       start=s["start"], end=s["end"], doc=s.get("doc", ""))
            parent = (f"sym:{f['path']}::{s['parent']}" if s["parent"] else fid)
            g.add_edge(parent if g.has_node(parent) else fid, sid, kind="contains")
            symbol_index.setdefault(s["name"], []).append(sid)

    # imports (resolved to repo files only), inheritance, calls
    for f in analysis["files"]:
        fid = f"file:{f['path']}"
        for imp in f["imports"]:
            target = _resolve_import(imp, module_index)
            if target and target != fid:
                g.add_edge(fid, target, kind="imports")
        for s in f["symbols"]:
            sid = f"sym:{f['path']}::{s['qualname']}"
            for base in s.get("bases", []):
                base = base.split(".")[-1]
                for tid in symbol_index.get(base, []):
                    if g.nodes[tid]["kind"] == "class":
                        g.add_edge(sid, tid, kind="inherits")
        local_names = {s["name"] for s in f["symbols"]}
        for called in f["calls"]:
            for tid in symbol_index.get(called, []):
                if g.nodes[tid]["path"] != f["path"] or called in local_names:
                    g.add_edge(fid, tid, kind="calls")

    logger.info("graphify built '{}': {} nodes, {} edges",
                analysis["name"], g.number_of_nodes(), g.number_of_edges())
    return g


def _module_name(path: str) -> str:
    p = Path(path)
    return ".".join(p.with_suffix("").parts).lower()


def _resolve_import(imp: str, module_index: dict[str, str]) -> str | None:
    key = imp.replace("/", ".").replace("\\", ".").lower().strip(". ")
    for cand in (key, key.split(".")[-1]):
        for mod, fid in module_index.items():
            if mod == cand or mod.endswith("." + cand):
                return fid
    return None


# ── persistence (local filesystem) ────────────────────────────────

def graph_path(repo_name: str) -> Path:
    return GRAPH_DIR / f"{_safe(repo_name)}.json"


def save_graph(g: nx.DiGraph) -> Path:
    path = graph_path(g.graph["name"])
    data = nx.node_link_data(g, edges="edges")
    data["graph"] = dict(g.graph)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def load_graph(repo_name: str) -> nx.DiGraph | None:
    path = graph_path(repo_name)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    g = nx.node_link_graph(data, directed=True, edges="edges")
    g.graph.update(data.get("graph", {}))
    return g


def list_graphs() -> list[str]:
    return sorted(p.stem for p in GRAPH_DIR.glob("*.json"))


# ── queries ───────────────────────────────────────────────────────

def find_nodes(g: nx.DiGraph, term: str, limit: int = 25) -> list[dict]:
    term_l = term.lower()
    scored = []
    for nid, attrs in g.nodes(data=True):
        label = attrs.get("label", "").lower()
        if term_l in label or term_l in nid.lower():
            exact = label == term_l or label.split(".")[-1] == term_l
            scored.append((0 if exact else 1, len(label), _node_dict(g, nid)))
    scored.sort(key=lambda t: (t[0], t[1]))
    return [d for _, _, d in scored[:limit]]


def neighborhood(g: nx.DiGraph, node_ids: list[str], hops: int = 1) -> dict:
    """Subgraph (nodes+edges as dicts) within `hops` of the seed nodes."""
    seeds = [n for n in node_ids if g.has_node(n)]
    keep = set(seeds)
    frontier = set(seeds)
    for _ in range(hops):
        nxt = set()
        for n in frontier:
            nxt.update(g.successors(n))
            nxt.update(g.predecessors(n))
        keep |= nxt
        frontier = nxt
    sub = g.subgraph(keep)
    return {
        "nodes": [_node_dict(g, n) for n in sub.nodes],
        "edges": [{"source": u, "target": v, "kind": d.get("kind", "")}
                  for u, v, d in sub.edges(data=True)],
    }


def important_nodes(g: nx.DiGraph, limit: int = 10) -> list[dict]:
    """Hub files/classes by degree centrality + git hotspot count."""
    deg = nx.degree_centrality(g)
    scored = sorted(
        (nid for nid in g.nodes if g.nodes[nid]["kind"] in ("file", "class")),
        key=lambda n: deg[n] + 0.01 * g.nodes[n].get("commits", 0), reverse=True)
    return [_node_dict(g, n) for n in scored[:limit]]


def describe_node(g: nx.DiGraph, nid: str) -> str:
    """One-line textual context for the LLM prompt."""
    a = g.nodes[nid]
    rels = []
    for u, v, d in list(g.in_edges(nid, data=True))[:6]:
        rels.append(f"{d['kind']}<-{g.nodes[u].get('label', u)}")
    for u, v, d in list(g.out_edges(nid, data=True))[:6]:
        rels.append(f"{d['kind']}->{g.nodes[v].get('label', v)}")
    loc = f" (lines {a['start']}-{a['end']})" if "start" in a else ""
    return f"[{a['kind']}] {a.get('label', nid)} in {a.get('path', '?')}{loc}; " \
           f"relations: {', '.join(rels) if rels else 'none'}"


def _node_dict(g: nx.DiGraph, nid: str) -> dict:
    return {"id": nid, **{k: v for k, v in g.nodes[nid].items()}}


def _safe(name: str) -> str:
    return re.sub(r"[^\w.-]", "_", name)


# ── pyvis export for the UI ───────────────────────────────────────

def export_html(g: nx.DiGraph, highlight: list[str] | None = None,
                max_nodes: int = 400) -> Path:
    from pyvis.network import Network

    highlight = set(highlight or [])
    net = Network(height="720px", width="100%", directed=True,
                  bgcolor="#111418", font_color="#e8e8e8", cdn_resources="in_line")
    net.barnes_hut(gravity=-12000, spring_length=140)

    nodes = list(g.nodes)[:max_nodes]
    keep = set(nodes) | highlight
    for nid in keep:
        if not g.has_node(nid):
            continue
        a = g.nodes[nid]
        color = "#ff5252" if nid in highlight else NODE_COLORS.get(a["kind"], "#7f8c8d")
        size = 28 if a["kind"] == "repo" else (18 if a["kind"] == "file" else 10)
        net.add_node(nid, label=a.get("label", nid).split("/")[-1],
                     title=f"{a['kind']}: {a.get('label', '')}\n{a.get('path', '')}",
                     color=color, size=size + (8 if nid in highlight else 0))
    for u, v, d in g.edges(data=True):
        if u in keep and v in keep:
            net.add_edge(u, v, color=EDGE_COLORS.get(d.get("kind"), "#555"),
                         title=d.get("kind", ""))

    out = GRAPH_DIR / f"{_safe(g.graph['name'])}.html"
    # pyvis' write_html uses the platform default encoding (cp1252 on
    # Windows), which fails on the inlined vis.js - write utf-8 ourselves.
    html = net.generate_html(str(out), notebook=False)
    out.write_text(html, encoding="utf-8")
    return out
