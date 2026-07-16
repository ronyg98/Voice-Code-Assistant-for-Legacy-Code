"""Speech-to-text via Whisper (OpenAI API).

Designed for speech diversity and responsible AI:
- multilingual: language auto-detect by default, or a per-user language hint
  from the personalization profile (helps non-standard accents)
- vocabulary priming: the user's custom domain terms (class names, jargon)
  are passed as the Whisper prompt, dramatically improving recognition of
  identifiers like "InvoiceReconciler"
- confidence: derived from Whisper segment log-probs; low-confidence
  transcripts trigger a "did I get that right?" confirmation in the UI, and
  user corrections are stored (asr_feedback) to monitor bias across
  languages/accents over time.
"""
import io
import math

from loguru import logger
from openai import OpenAI

from app.config import OPENAI_API_KEY, STT_MODEL

_client: OpenAI | None = None
CONFIRM_THRESHOLD = 0.62   # below this, UI asks the user to confirm transcript


def _oa() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY, timeout=120)
    return _client


def transcribe(audio_bytes: bytes, filename: str = "audio.wav",
               language: str = "", vocabulary: str = "") -> dict:
    """Returns {text, language, confidence, needs_confirmation, segments}."""
    buf = io.BytesIO(audio_bytes)
    buf.name = filename   # OpenAI SDK infers format from the name

    kwargs: dict = {"model": STT_MODEL, "file": buf,
                    "response_format": "verbose_json"}
    if language:
        kwargs["language"] = language
    if vocabulary:
        kwargs["prompt"] = f"Technical terms that may appear: {vocabulary[:600]}"

    resp = _oa().audio.transcriptions.create(**kwargs)

    segments = getattr(resp, "segments", None) or []
    confidence = _confidence(segments)
    detected = getattr(resp, "language", "") or language
    text = (resp.text or "").strip()

    logger.info("STT [{}] conf={:.2f}: {}", detected, confidence, text[:120])
    return {
        "text": text,
        "language": detected,
        "confidence": round(confidence, 3),
        "needs_confirmation": confidence < CONFIRM_THRESHOLD or not text,
        "segments": [
            {"text": s.text, "avg_logprob": s.avg_logprob,
             "no_speech_prob": s.no_speech_prob}
            for s in segments
        ],
    }


def _confidence(segments) -> float:
    """Map Whisper segment stats to [0,1]."""
    if not segments:
        return 0.5
    probs = []
    for s in segments:
        p = math.exp(min(0.0, s.avg_logprob))          # avg token prob
        p *= (1.0 - min(1.0, max(0.0, s.no_speech_prob)))
        probs.append(p)
    return sum(probs) / len(probs)
