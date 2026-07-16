"""Wake-word detection with openWakeWord (free, runs locally on CPU).

Default model is the pretrained "hey_jarvis" (openWakeWord ships no
"hey assistant" model; training a custom one is supported by the library -
set WAKEWORD_MODEL in .env once you have it). Detection runs on 80 ms frames
so activation latency is low.

If openwakeword or its model download is unavailable, `WakeWord.available`
is False and the live assistant falls back to push-to-talk (Enter key).
"""
import numpy as np
from loguru import logger

from app.config import WAKEWORD_MODEL, WAKEWORD_THRESHOLD

CHUNK = 1280                     # 80 ms @ 16 kHz - openwakeword's native hop


class WakeWord:
    def __init__(self, model_name: str = WAKEWORD_MODEL,
                 threshold: float = WAKEWORD_THRESHOLD):
        self.threshold = threshold
        self.model = None
        self.name = model_name
        try:
            import openwakeword
            from openwakeword.model import Model
            try:  # fetch pretrained model files on first run (cached afterwards)
                openwakeword.utils.download_models(model_names=[model_name])
            except Exception as exc:
                logger.warning("wake-word model download issue: {}", exc)
            self.model = Model(wakeword_models=[model_name],
                               inference_framework="onnx")
            logger.info("wake word ready: '{}' (threshold {})", model_name, threshold)
        except Exception as exc:
            logger.warning("openwakeword unavailable - push-to-talk fallback: {}", exc)

    @property
    def available(self) -> bool:
        return self.model is not None

    def reset(self) -> None:
        if self.model is not None:
            self.model.reset()

    def detect(self, frame_int16: np.ndarray) -> float:
        """Feed one 1280-sample frame; returns best score in [0,1]."""
        if self.model is None:
            return 0.0
        scores = self.model.predict(frame_int16)
        return float(max(scores.values())) if scores else 0.0

    def triggered(self, frame_int16: np.ndarray) -> bool:
        return self.detect(frame_int16) >= self.threshold
