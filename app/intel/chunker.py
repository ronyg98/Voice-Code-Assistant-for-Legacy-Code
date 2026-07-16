"""Symbol-aware code chunking for embeddings.

Chunks follow symbol boundaries when GitNexus found symbols (a function or
class per chunk, split if huge); otherwise fixed-size line windows with
overlap. Every chunk carries path + line range + graph node id so answers can
cite exact code and light up graph nodes.
"""

MAX_CHUNK_LINES = 80
OVERLAP_LINES = 10


def chunk_analysis(analysis: dict) -> list[dict]:
    chunks = []
    for f in analysis["files"]:
        chunks.extend(_chunk_file(f))
    return chunks


def _chunk_file(f: dict) -> list[dict]:
    lines = f["text"].splitlines()
    chunks = []
    # top-level symbols only (methods live inside their class chunk unless big)
    symbols = [s for s in f["symbols"] if not s["parent"]]
    covered = set()

    for s in symbols:
        start, end = s["start"], min(s["end"], len(lines))
        for w_start in range(start, end + 1, MAX_CHUNK_LINES):
            w_end = min(w_start + MAX_CHUNK_LINES - 1, end)
            chunks.append(_make(f, lines, w_start, w_end,
                                symbol=s["qualname"], kind=s["kind"]))
            covered.update(range(w_start, w_end + 1))
            if w_end >= end:
                break

    # any uncovered regions (module-level code, config files, docs)
    i = 1
    n = len(lines)
    while i <= n:
        if i in covered:
            i += 1
            continue
        j = i
        while j <= n and j not in covered and j - i < MAX_CHUNK_LINES:
            j += 1
        if any(line.strip() for line in lines[i - 1:j - 1]):
            chunks.append(_make(f, lines, max(1, i - OVERLAP_LINES if i > 1 else i),
                                j - 1, symbol="", kind="segment"))
        i = j
    return chunks


def _make(f: dict, lines: list[str], start: int, end: int,
          symbol: str, kind: str) -> dict:
    text = "\n".join(lines[start - 1:end])
    node_id = f"sym:{f['path']}::{symbol}" if symbol else f"file:{f['path']}"
    return {
        "id": f"{f['path']}:{start}-{end}",
        "path": f["path"],
        "language": f["language"],
        "start": start,
        "end": end,
        "symbol": symbol,
        "kind": kind,
        "node_id": node_id,
        # header helps the embedding model situate the code
        "text": f"// {f['path']} lines {start}-{end}"
                f"{' | ' + symbol if symbol else ''}\n{text}",
    }
