"""Request tracing: per-stage latency, retrieval diagnostics, prompt traces.

A `Trace` is created per user request (one voice turn / one question). Stages
are timed with a context manager; the finished trace is written to
data/logs/traces/<trace_id>.json and a compact summary is kept in memory for
the /api/metrics endpoint and the Streamlit observability dashboard.
"""
import json
import statistics
import threading
import time
import uuid
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from app.config import TRACE_DIR

_recent_traces: deque = deque(maxlen=200)   # summaries for /api/metrics
_lock = threading.Lock()


@dataclass
class Trace:
    kind: str                                # "ask" | "stt" | "tts" | "index"
    user: str = "-"
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    started: float = field(default_factory=time.time)
    stages: dict[str, float] = field(default_factory=dict)      # name -> ms
    meta: dict[str, Any] = field(default_factory=dict)          # free-form
    retrieval: list[dict] = field(default_factory=list)          # diagnostics
    prompt: str = ""                                             # full prompt trace
    answer: str = ""

    @contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            ms = (time.perf_counter() - t0) * 1000
            self.stages[name] = round(self.stages.get(name, 0.0) + ms, 1)
            logger.bind(trace_id=self.trace_id, stage=name).debug(
                "stage {} finished in {:.0f} ms", name, ms)

    def log_retrieval(self, query: str, hits: list[dict]) -> None:
        self.retrieval.append({
            "query": query,
            "hits": [
                {"id": h.get("id"), "score": h.get("score"), "path": h.get("path")}
                for h in hits
            ],
        })

    def finish(self) -> dict:
        total_ms = round((time.time() - self.started) * 1000, 1)
        summary = {
            "trace_id": self.trace_id,
            "kind": self.kind,
            "user": self.user,
            "ts": self.started,
            "total_ms": total_ms,
            "stages": self.stages,
            "meta": self.meta,
        }
        with _lock:
            _recent_traces.append(summary)
        # Full trace (prompt + retrieval diagnostics) to local filesystem.
        full = {**summary, "retrieval": self.retrieval,
                "prompt": self.prompt, "answer": self.answer}
        try:
            (TRACE_DIR / f"{self.trace_id}.json").write_text(
                json.dumps(full, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8")
        except OSError as exc:
            logger.warning("could not persist trace {}: {}", self.trace_id, exc)
        logger.bind(trace_id=self.trace_id).info(
            "{} done in {} ms | stages={}", self.kind, total_ms, self.stages)
        return summary


def metrics_snapshot() -> dict:
    """Aggregate latency stats over recent traces, per kind and per stage."""
    with _lock:
        traces = list(_recent_traces)
    by_kind: dict[str, list[float]] = {}
    stage_ms: dict[str, list[float]] = {}
    for t in traces:
        by_kind.setdefault(t["kind"], []).append(t["total_ms"])
        for s, ms in t["stages"].items():
            stage_ms.setdefault(s, []).append(ms)

    def stats(vals: list[float]) -> dict:
        vals = sorted(vals)
        return {
            "count": len(vals),
            "p50_ms": round(statistics.median(vals), 1),
            "p95_ms": round(vals[max(0, int(len(vals) * 0.95) - 1)], 1),
            "max_ms": round(vals[-1], 1),
        }

    return {
        "kinds": {k: stats(v) for k, v in by_kind.items()},
        "stages": {k: stats(v) for k, v in stage_ms.items()},
        "recent": traces[-25:][::-1],
    }
