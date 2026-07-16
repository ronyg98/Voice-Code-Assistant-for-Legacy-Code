"""Vector index: OpenAI text-embedding-3-large -> ChromaDB (local, persistent).

One Chroma collection per indexed repo. Query returns hits with similarity
scores (used for confidence) and metadata linking back to file lines and
graph nodes (used for citations and code navigation).
"""
import re

import chromadb
from loguru import logger
from openai import OpenAI

from app.config import CHROMA_DIR, EMBEDDING_MODEL, OPENAI_API_KEY

_chroma = None          # chromadb PersistentClient (created lazily)
_openai: OpenAI | None = None
BATCH = 64


def _client():
    global _chroma
    if _chroma is None:
        _chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _chroma


def _oa() -> OpenAI:
    global _openai
    if _openai is None:
        _openai = OpenAI(api_key=OPENAI_API_KEY, timeout=120)
    return _openai


def collection_name(repo: str) -> str:
    return "repo_" + re.sub(r"[^a-zA-Z0-9._-]", "_", repo)[:50]


def _embed(texts: list[str]) -> list[list[float]]:
    resp = _oa().embeddings.create(model=EMBEDDING_MODEL,
                                   input=[t[:8000] for t in texts])
    return [d.embedding for d in resp.data]


def index_chunks(repo: str, chunks: list[dict]) -> int:
    """(Re)index a repo's chunks. Returns number of chunks embedded."""
    client = _client()
    name = collection_name(repo)
    try:
        client.delete_collection(name)   # full reindex keeps things consistent
    except Exception:
        pass
    col = client.create_collection(name, metadata={"hnsw:space": "cosine"})

    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i + BATCH]
        col.add(
            ids=[c["id"] for c in batch],
            embeddings=_embed([c["text"] for c in batch]),
            documents=[c["text"] for c in batch],
            metadatas=[{"path": c["path"], "start": c["start"], "end": c["end"],
                        "symbol": c["symbol"], "language": c["language"],
                        "node_id": c["node_id"]} for c in batch],
        )
        logger.info("indexed {}/{} chunks for '{}'",
                    min(i + BATCH, len(chunks)), len(chunks), repo)
    return len(chunks)


def query(repo: str, question: str, top_k: int = 8) -> list[dict]:
    """Similarity search. Returns hits with score in [0,1] (1 = identical)."""
    try:
        col = _client().get_collection(collection_name(repo))
    except Exception:
        return []
    res = col.query(query_embeddings=_embed([question]), n_results=top_k,
                    include=["documents", "metadatas", "distances"])
    hits = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0],
                               res["distances"][0]):
        hits.append({
            "id": f"{meta['path']}:{meta['start']}-{meta['end']}",
            "score": round(max(0.0, 1.0 - dist), 4),   # cosine distance -> similarity
            "text": doc,
            **meta,
        })
    return hits


def indexed_repos() -> list[str]:
    return sorted(c.name.removeprefix("repo_") for c in _client().list_collections()
                  if c.name.startswith("repo_"))
