"""End-to-end indexing pipeline for a legacy repository.

    GitNexus (analyze) -> Graphify (build + persist graph)
                       -> chunker + embeddings (vector index)

Used by POST /api/index and scripts/index_repo.py.
"""
from loguru import logger

from app import rag
from app.intel import chunker, embeddings, gitnexus, graphify
from app.observability import Trace


def index_repository(repo_path: str, user: str = "-") -> dict:
    trace = Trace(kind="index", user=user)
    with trace.stage("gitnexus_analyze"):
        analysis = gitnexus.analyze_repo(repo_path)
    with trace.stage("graphify_build"):
        graph = graphify.build_graph(analysis)
        graphify.save_graph(graph)
    with trace.stage("chunking"):
        chunks = chunker.chunk_analysis(analysis)
    with trace.stage("embedding"):
        n = embeddings.index_chunks(analysis["name"], chunks)

    rag.invalidate_graph(analysis["name"])
    summary = {
        "repo": analysis["name"],
        "root": analysis["root"],
        "files": analysis["n_files"],
        "loc": analysis["total_loc"],
        "languages": analysis["languages"],
        "graph_nodes": graph.number_of_nodes(),
        "graph_edges": graph.number_of_edges(),
        "chunks_indexed": n,
        "git_commits": analysis["git"].get("n_commits", 0),
    }
    trace.meta.update(summary)
    trace.finish()
    logger.info("indexing complete: {}", summary)
    return summary
