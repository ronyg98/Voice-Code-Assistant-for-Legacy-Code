"""LLM access with a provider fallback chain.

All four providers in .env (OpenAI, Groq, Mistral, Google/Gemini) expose an
OpenAI-compatible chat-completions endpoint, so one client class covers them.
The chain is sticky: the last provider that worked is tried first. Streaming
is a generator of text deltas so TTS can start speaking before the full
answer exists.
"""
import threading
from collections.abc import Iterator

from loguru import logger
from openai import OpenAI

from app.config import LLM_PROVIDERS

_clients: dict[str, OpenAI] = {}
_preferred = {"name": None}
_lock = threading.Lock()


def _client(provider: dict) -> OpenAI:
    with _lock:
        if provider["name"] not in _clients:
            _clients[provider["name"]] = OpenAI(
                api_key=provider["api_key"], base_url=provider["base_url"],
                timeout=60, max_retries=1)
        return _clients[provider["name"]]


def _ordered_providers(prefer: str | None = None) -> list[dict]:
    providers = [p for p in LLM_PROVIDERS if p["api_key"]]
    pick = prefer or _preferred["name"]
    if pick:
        providers.sort(key=lambda p: p["name"] != pick)
    return providers


def available_providers() -> list[str]:
    return [p["name"] for p in LLM_PROVIDERS if p["api_key"]]


def chat(messages: list[dict], temperature: float = 0.3,
         max_tokens: int = 1200, prefer: str | None = None) -> tuple[str, str]:
    """Non-streaming completion. Returns (text, provider_name).

    `prefer` pins a provider to the front of the chain for THIS call (e.g.
    the planner runs on groq for latency) without changing the sticky
    preference used by other callers; the rest of the chain still backs
    it up on failure.
    """
    errors = []
    for provider in _ordered_providers(prefer):
        try:
            resp = _client(provider).chat.completions.create(
                model=provider["model"], messages=messages,
                temperature=temperature, max_tokens=max_tokens)
            if prefer is None:
                _preferred["name"] = provider["name"]
            return resp.choices[0].message.content or "", provider["name"]
        except Exception as exc:
            errors.append(f"{provider['name']}: {exc}")
            logger.warning("LLM provider {} failed, falling through: {}",
                           provider["name"], exc)
    raise RuntimeError("all LLM providers failed: " + " | ".join(errors))


def chat_stream(messages: list[dict], temperature: float = 0.3,
                max_tokens: int = 1200) -> Iterator[tuple[str, str]]:
    """Streaming completion. Yields (delta_text, provider_name).

    Falls through to the next provider only if the failure happens before the
    first token; mid-stream errors surface to the caller.
    """
    errors = []
    for provider in _ordered_providers():
        try:
            stream = _client(provider).chat.completions.create(
                model=provider["model"], messages=messages,
                temperature=temperature, max_tokens=max_tokens, stream=True)
        except Exception as exc:
            errors.append(f"{provider['name']}: {exc}")
            logger.warning("LLM provider {} failed, falling through: {}",
                           provider["name"], exc)
            continue
        _preferred["name"] = provider["name"]
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content, provider["name"]
        return
    raise RuntimeError("all LLM providers failed: " + " | ".join(errors))
