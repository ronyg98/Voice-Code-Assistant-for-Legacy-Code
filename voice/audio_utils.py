"""Small audio helpers shared by the voice pipeline."""
import io

import numpy as np
import soundfile as sf


def int16_to_float(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32) / 32768.0


def float_to_int16(x: np.ndarray) -> np.ndarray:
    return np.clip(x * 32768.0, -32768, 32767).astype(np.int16)


def wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    """int16 or float32 mono samples -> WAV file bytes (for the Whisper API)."""
    if samples.dtype != np.int16:
        samples = float_to_int16(samples)
    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def rms(x: np.ndarray) -> float:
    if x.dtype == np.int16:
        x = int16_to_float(x)
    return float(np.sqrt(np.mean(np.square(x))) if x.size else 0.0)
