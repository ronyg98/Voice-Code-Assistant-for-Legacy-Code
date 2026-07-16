"""Controllable live voice service - wake-word mode for the web UI.

Runs the hands-free loop (wake word -> VAD capture -> denoise -> Whisper ->
RAG pipeline -> streamed TTS through the machine's speakers) inside the
backend process, controlled over HTTP:

    POST /api/live/start    start listening (repo + chat session)
    POST /api/live/trigger  capture immediately, skipping the wake word
    POST /api/live/stop     stop the loop
    GET  /api/live/status   state machine + transcript + partial answer

Because Streamlit, FastAPI, and the user share one laptop, the server's mic
and speakers ARE the user's mic and speakers. Answers are written into the
same chat session the UI displays, so spoken turns appear in the chat.
"""
import threading
import time

import numpy as np
from loguru import logger

from app import db, pipeline, stt, tts
from app.config import MIC_SAMPLE_RATE, TTS_SAMPLE_RATE
from app.observability import Trace
from voice import vad as vad_mod
from voice.audio_utils import rms, wav_bytes
from voice.denoise import denoise
from voice.wakeword import CHUNK, WakeWord

MAX_UTTERANCE_S = 18
TRAILING_SILENCE_S = 0.9
MIN_SPEECH_FRAMES = 6


class LiveVoiceService:
    def __init__(self):
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._trigger = threading.Event()
        self._status = self._fresh_status()

    # ── public API ───────────────────────────────────────────────

    def start(self, user: dict, repo: str, session_id: str) -> tuple[bool, str]:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return False, "live mode is already running"
            self._stop.clear()
            self._trigger.clear()
            self._status = self._fresh_status()
            self._status.update({"running": True, "repo": repo,
                                 "user": user["username"], "state": "starting",
                                 "session_id": session_id})
            self._thread = threading.Thread(
                target=self._run, args=(dict(user), repo, session_id),
                daemon=True, name="live-voice")
            self._thread.start()
        logger.info("live voice service started for {} on '{}'",
                    user["username"], repo)
        return True, "started"

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=5)
        self._set(running=False, state="off")
        logger.info("live voice service stopped")

    def trigger_listen(self) -> None:
        self._trigger.set()

    def status(self) -> dict:
        with self._lock:
            return dict(self._status)

    # ── internals ────────────────────────────────────────────────

    @staticmethod
    def _fresh_status() -> dict:
        return {"running": False, "state": "off", "repo": "", "user": "",
                "session_id": "", "wake_available": False, "wake_model": "",
                "mic_ok": False, "input_device": "", "mic_level": 0.0,
                "wake_score": 0.0, "heard": None, "partial_answer": "",
                "answer": None, "turns": 0, "error": "", "started_at": time.time()}

    def _set(self, **kw) -> None:
        with self._lock:
            self._status.update(kw)

    def _run(self, user: dict, repo: str, session_id: str) -> None:
        import queue

        import sounddevice as sd

        profile = db.get_profile(user["username"])
        wake = WakeWord()
        vad = vad_mod.SileroVAD()
        mic_q: queue.Queue = queue.Queue()
        buffer = np.zeros(0, dtype=np.int16)
        self._set(wake_available=wake.available, wake_model=wake.name)

        def mic_cb(indata, frames, t, status):
            mic_q.put(indata[:, 0].copy())

        def pull(n: int, timeout: float = 0.5):
            nonlocal buffer
            deadline = time.monotonic() + timeout
            while buffer.size < n:
                if self._stop.is_set():
                    return None
                try:
                    buffer = np.concatenate(
                        [buffer, mic_q.get(timeout=max(0.05, deadline - time.monotonic()))])
                except queue.Empty:
                    return None
            chunk, buffer = buffer[:n], buffer[n:]
            return chunk

        def drain():
            nonlocal buffer
            parts = []
            while not mic_q.empty():
                parts.append(mic_q.get_nowait())
            if parts:
                buffer = np.concatenate([buffer] + parts)
            take, buffer = buffer, np.zeros(0, dtype=np.int16)
            return take

        try:
            out = sd.OutputStream(samplerate=TTS_SAMPLE_RATE, channels=1, dtype="int16")
            out.start()
            mic = sd.InputStream(samplerate=MIC_SAMPLE_RATE, channels=1,
                                 dtype="int16", blocksize=CHUNK, callback=mic_cb)
            mic.start()
        except Exception as exc:
            logger.error("live voice: audio device unavailable: {}", exc)
            self._set(running=False, state="off",
                      error=f"audio device unavailable: {exc}")
            return
        try:
            device_name = sd.query_devices(kind="input")["name"]
        except Exception:
            device_name = "?"
        self._set(mic_ok=True, input_device=device_name)

        def beep(freq=880, ms=140):
            t = np.linspace(0, ms / 1000, int(TTS_SAMPLE_RATE * ms / 1000), False)
            tone = (0.25 * np.sin(2 * np.pi * freq * t) *
                    np.hanning(t.size) * 32767).astype(np.int16)
            try:
                out.write(tone)
            except Exception:
                pass

        def barge_in() -> bool:
            heard = drain()
            for i in range(0, heard.size - CHUNK + 1, CHUNK):
                if wake.triggered(heard[i:i + CHUNK]):
                    return True
            return False

        def capture() -> np.ndarray | None:
            self._set(state="capturing", partial_answer="", answer=None)
            vad.reset()
            collected, speech_frames, silence = [], 0, 0.0
            start = time.monotonic()
            while (time.monotonic() - start < MAX_UTTERANCE_S
                   and not self._stop.is_set()):
                frame = pull(vad_mod.FRAME)
                if frame is None:
                    continue
                collected.append(frame)
                self._set(mic_level=round(float(rms(frame)), 4))
                if vad.is_speech(frame):
                    speech_frames += 1
                    silence = 0.0
                else:
                    silence += vad_mod.FRAME / MIC_SAMPLE_RATE
                    if speech_frames >= MIN_SPEECH_FRAMES and silence >= TRAILING_SILENCE_S:
                        break
            if speech_frames < MIN_SPEECH_FRAMES or not collected:
                return None
            return np.concatenate(collected)

        try:
            while not self._stop.is_set():
                # ── wait for wake word (or manual trigger from the UI) ──
                self._set(state="waiting_wake")
                wake.reset()
                woke = False
                while not self._stop.is_set():
                    if self._trigger.is_set():
                        self._trigger.clear()
                        woke = True
                        break
                    frame = pull(CHUNK)
                    if frame is None:
                        continue
                    score = wake.detect(frame) if wake.available else 0.0
                    with self._lock:  # live diagnostics for the UI panel
                        self._status["mic_level"] = round(float(rms(frame)), 4)
                        self._status["wake_score"] = round(
                            max(float(score), self._status["wake_score"] * 0.9), 3)
                    if score >= wake.threshold:
                        woke = True
                        break
                if not woke:
                    break
                beep()
                drain()

                utterance = capture()
                if utterance is None:
                    self._set(state="waiting_wake",
                              heard={"text": "", "note": "no speech detected"})
                    continue

                # ── STT ──
                self._set(state="transcribing")
                trace = Trace(kind="ask", user=user["username"])
                with trace.stage("denoise"):
                    cleaned = denoise(utterance, MIC_SAMPLE_RATE)
                try:
                    with trace.stage("stt"):
                        heard = stt.transcribe(
                            wav_bytes(cleaned, MIC_SAMPLE_RATE),
                            language=profile.get("language", ""),
                            vocabulary=profile.get("vocabulary", ""))
                except Exception as exc:
                    logger.error("live STT failed: {}", exc)
                    self._set(state="waiting_wake", error=f"STT failed: {exc}")
                    continue
                question = heard["text"]
                self._set(heard={"text": question,
                                 "language": heard["language"],
                                 "confidence": heard["confidence"]}, error="")
                if not question:
                    self._set(state="waiting_wake")
                    continue

                # ── ask + speak (interruptible) ──
                self._set(state="thinking")
                events = pipeline.answer_stream(question, session_id, user,
                                                repo, trace)
                final: dict = {}

                def tokens():
                    for ev in events:
                        if self._stop.is_set():
                            return
                        if ev["type"] == "token":
                            with self._lock:
                                self._status["partial_answer"] += ev["text"]
                            yield ev["text"]
                        elif ev["type"] == "final":
                            final.update(ev)

                interrupted = False
                wake.reset()
                self._set(state="speaking")
                try:
                    for chunk in tts.stream_sentences(
                            tokens(), voice=profile.get("tts_voice", ""),
                            speed=float(profile.get("speech_rate", 1.0))):
                        samples = np.frombuffer(chunk, dtype=np.int16)
                        for i in range(0, samples.size, 2048):
                            out.write(samples[i:i + 2048])
                            if self._stop.is_set() or barge_in():
                                interrupted = True
                                break
                        if interrupted:
                            break
                except Exception as exc:
                    logger.warning("live TTS playback issue: {}", exc)

                with self._lock:
                    self._status["turns"] += 1
                    self._status["answer"] = {
                        "text": final.get("answer", self._status["partial_answer"]),
                        "confidence": final.get("confidence"),
                        "citations": final.get("citations", []),
                        "provider": final.get("provider", ""),
                        "interrupted": interrupted,
                    }
                if interrupted and not self._stop.is_set():
                    beep(660)
                    drain()
                    self._trigger.set()   # go straight back to capturing
        finally:
            for closer in (mic, out):
                try:
                    closer.stop()
                    closer.close()
                except Exception:
                    pass
            self._set(running=False, state="off")


live_service = LiveVoiceService()
