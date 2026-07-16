"""Structured logging with Loguru.

Two sinks:
  - stderr: human-readable, colorized
  - data/logs/app.jsonl: one JSON object per line (machine-readable, used by
    the Observability tab), rotated at 10 MB.

Use `logger.bind(trace_id=..., stage=...)` anywhere; bound fields land in the
JSON `record.extra` so latency metrics and prompt traces can be correlated.
"""
import sys

from loguru import logger

from app.config import LOG_DIR

_configured = False


def setup_logging() -> None:
    global _configured
    if _configured:
        return
    _configured = True

    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <7}</level> | "
               "<cyan>{extra[trace_id]}</cyan> | {message}",
        filter=_ensure_trace_id,
    )
    logger.add(
        LOG_DIR / "app.jsonl",
        level="DEBUG",
        serialize=True,          # full structured JSON per line
        rotation="10 MB",
        retention=10,
        enqueue=True,            # safe across threads
        filter=_ensure_trace_id,
    )
    logger.info("logging initialised -> {}", LOG_DIR / "app.jsonl")


def _ensure_trace_id(record) -> bool:
    record["extra"].setdefault("trace_id", "-")
    return True
