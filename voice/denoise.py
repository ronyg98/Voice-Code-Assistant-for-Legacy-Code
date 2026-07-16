"""Noise suppression for real-world audio.

Uses `noisereduce` (free, pure-Python spectral gating - the practical
Windows-friendly alternative to RNNoise; same role, no native build needed).
Applied to each captured utterance before STT, which measurably improves
Whisper accuracy in noisy rooms (fans, traffic, keyboards).
"""
import numpy as np
from loguru import logger

from voice.audio_utils import float_to_int16, int16_to_float

try:
    import noisereduce as nr
    _AVAILABLE = True
except Exception as _exc:                       # pragma: no cover
    _AVAILABLE = False
    logger.warning("noisereduce unavailable ({}) - denoising disabled", _exc)


def denoise(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """int16 mono in -> int16 mono out. Falls back to passthrough."""
    if not _AVAILABLE or samples.size < sample_rate // 4:
        return samples
    try:
        cleaned = nr.reduce_noise(y=int16_to_float(samples), sr=sample_rate,
                                  stationary=True, prop_decrease=0.9)
        return float_to_int16(cleaned)
    except Exception as exc:
        logger.warning("denoise failed, using raw audio: {}", exc)
        return samples
