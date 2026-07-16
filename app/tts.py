"""Text-to-speech via OpenAI TTS with sentence-pipelined streaming.

Two consumption modes:
- `synthesize_wav`: whole utterance -> WAV bytes (browser playback in the UI)
- `stream_sentences`: consumes an in-flight token stream from the LLM,
  cuts it at sentence boundaries, synthesizes each sentence to raw PCM and
  yields chunks - so the assistant starts SPEAKING while the model is still
  writing the rest of the answer (streaming TTS requirement).
"""
import re
from collections.abc import Iterable, Iterator

from loguru import logger
from openai import OpenAI

from app.config import OPENAI_API_KEY, TTS_MODEL, TTS_SAMPLE_RATE, TTS_VOICE

_client: OpenAI | None = None
_SENTENCE_END = re.compile(r"([.!?;:])\s")
_MD_NOISE = re.compile(r"[*_`#>|]|\[\d+\]")   # strip markdown + [n] citations for speech


def _oa() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY, timeout=120)
    return _client


def speakable(text: str) -> str:
    return _MD_NOISE.sub("", text).replace("\n", " ").strip()


def synthesize_wav(text: str, voice: str = "", speed: float = 1.0) -> bytes:
    text = speakable(text)[:4000]
    if not text:
        return b""
    resp = _oa().audio.speech.create(
        model=TTS_MODEL, voice=voice or TTS_VOICE, input=text,
        speed=max(0.25, min(4.0, speed)), response_format="wav")
    return resp.content


def synthesize_pcm(text: str, voice: str = "", speed: float = 1.0) -> Iterator[bytes]:
    """One utterance -> raw 16-bit mono PCM chunks at TTS_SAMPLE_RATE."""
    text = speakable(text)[:4000]
    if not text:
        return
    with _oa().audio.speech.with_streaming_response.create(
            model=TTS_MODEL, voice=voice or TTS_VOICE, input=text,
            speed=max(0.25, min(4.0, speed)), response_format="pcm") as resp:
        yield from resp.iter_bytes(chunk_size=4096)


def stream_sentences(token_stream: Iterable[str], voice: str = "",
                     speed: float = 1.0) -> Iterator[bytes]:
    """LLM token stream -> PCM audio, sentence by sentence.

    The first sentence starts synthesizing as soon as its terminator arrives,
    long before the full answer is complete.
    """
    buffer = ""
    for token in token_stream:
        buffer += token
        while True:
            m = _SENTENCE_END.search(buffer)
            if not m or m.end() < 24:   # avoid synthesizing tiny fragments
                break
            sentence, buffer = buffer[:m.end()], buffer[m.end():]
            yield from synthesize_pcm(sentence, voice, speed)
    if buffer.strip():
        yield from synthesize_pcm(buffer, voice, speed)
    logger.debug("tts sentence pipeline finished")


SAMPLE_RATE = TTS_SAMPLE_RATE
