"""Voice Activity Detection with Silero VAD.

Runs the Silero ONNX model directly through onnxruntime (no PyTorch - keeps
the install ~2 GB lighter). The model file is downloaded once to
data/models/silero_vad.onnx. If the download or runtime is unavailable the
detector degrades to an adaptive energy (RMS) gate so the assistant still
works offline.

Frames must be 512 samples of 16 kHz mono (32 ms).
"""
import numpy as np
from loguru import logger

from app.config import MODELS_DIR, VAD_THRESHOLD
from voice.audio_utils import int16_to_float, rms

FRAME = 512
CONTEXT = 64          # silero v5 expects 64 samples of left context per frame
SAMPLE_RATE = 16_000
_URLS = [
    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx",
    "https://github.com/snakers4/silero-vad/raw/v5.1/src/silero_vad/data/silero_vad.onnx",
]


class SileroVAD:
    def __init__(self, threshold: float = VAD_THRESHOLD):
        self.threshold = threshold
        self.session = None
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(CONTEXT, dtype=np.float32)
        self._noise_floor = 0.01           # for the energy fallback
        try:
            import onnxruntime as ort
            path = self._ensure_model()
            if path:
                opts = ort.SessionOptions()
                opts.log_severity_level = 3
                self.session = ort.InferenceSession(
                    str(path), opts, providers=["CPUExecutionProvider"])
                logger.info("Silero VAD ready ({})", path.name)
        except Exception as exc:
            logger.warning("Silero VAD unavailable, using energy gate: {}", exc)

    def _ensure_model(self):
        path = MODELS_DIR / "silero_vad.onnx"
        if path.exists():
            return path
        import requests
        for url in _URLS:
            try:
                r = requests.get(url, timeout=60)
                if r.ok and len(r.content) > 100_000:
                    path.write_bytes(r.content)
                    logger.info("downloaded silero_vad.onnx ({} kB)", len(r.content) // 1024)
                    return path
            except requests.RequestException as exc:
                logger.warning("silero download failed from {}: {}", url, exc)
        return None

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(CONTEXT, dtype=np.float32)

    def is_speech(self, frame_int16: np.ndarray) -> bool:
        return self.probability(frame_int16) >= self.threshold

    def probability(self, frame_int16: np.ndarray) -> float:
        """Speech probability for one 512-sample 16 kHz frame."""
        if frame_int16.size != FRAME:
            frame_int16 = np.resize(frame_int16, FRAME)
        if self.session is None:
            return self._energy_prob(frame_int16)
        samples = int16_to_float(frame_int16)
        audio = np.concatenate([self._context, samples])[np.newaxis, :]
        try:
            out, self._state = self.session.run(
                None, {"input": audio, "state": self._state,
                       "sr": np.array(SAMPLE_RATE, dtype=np.int64)})
            self._context = samples[-CONTEXT:]
            return float(out[0][0])
        except Exception as exc:
            logger.warning("silero inference failed, switching to energy gate: {}", exc)
            self.session = None
            return self._energy_prob(frame_int16)

    def _energy_prob(self, frame_int16: np.ndarray) -> float:
        level = rms(frame_int16)
        # slowly track the noise floor; speech = well above it
        self._noise_floor = 0.995 * self._noise_floor + 0.005 * level
        return 1.0 if level > max(0.015, self._noise_floor * 3.0) else 0.0
