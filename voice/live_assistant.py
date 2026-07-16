"""Hands-free live voice assistant (desktop loop).

    wake word ("hey jarvis") ──> beep ──> VAD-captured utterance
        ──> noise suppression ──> Whisper STT
        ──> RAG pipeline (session memory, citations, confidence)
        ──> streaming LLM ──> sentence-pipelined TTS (starts speaking
            before the answer is finished)

Interruptions: while the assistant is speaking, the mic keeps listening;
saying the wake word again stops playback immediately (barge-in) and starts
a new capture. If openWakeWord is unavailable, press Enter to talk instead.

Run:
    .venv\\Scripts\\python.exe -m voice.live_assistant --repo <indexed-repo> ^
        --username dev --password dev123
"""
import argparse
import queue
import sys
import time

import numpy as np
import sounddevice as sd
from loguru import logger

from app import auth, db, pipeline, stt, tts
from app.config import MIC_SAMPLE_RATE, ROLES, TTS_SAMPLE_RATE
from app.logging_setup import setup_logging
from app.observability import Trace
from voice import vad as vad_mod
from voice.audio_utils import wav_bytes
from voice.denoise import denoise
from voice.wakeword import CHUNK, WakeWord

MAX_UTTERANCE_S = 18
TRAILING_SILENCE_S = 0.9
MIN_SPEECH_FRAMES = 6


class LiveAssistant:
    def __init__(self, repo: str, user: dict):
        self.repo = repo
        self.user = user
        self.profile = db.get_profile(user["username"])
        self.session_id = db.create_session(user["username"], repo=repo)
        self.wake = WakeWord()
        self.vad = vad_mod.SileroVAD()
        self.mic_q: queue.Queue[np.ndarray] = queue.Queue()
        self.buffer = np.zeros(0, dtype=np.int16)
        self.out = sd.OutputStream(samplerate=TTS_SAMPLE_RATE, channels=1,
                                   dtype="int16")
        self.out.start()

    # ── mic plumbing ─────────────────────────────────────────────

    def _mic_cb(self, indata, frames, t, status):
        if status:
            logger.debug("mic status: {}", status)
        self.mic_q.put(indata[:, 0].copy())

    def _pull(self, n: int, timeout: float = 2.0) -> np.ndarray | None:
        """Blocking read of exactly n samples from the mic."""
        deadline = time.monotonic() + timeout
        while self.buffer.size < n:
            try:
                self.buffer = np.concatenate(
                    [self.buffer, self.mic_q.get(timeout=max(0.05, deadline - time.monotonic()))])
            except queue.Empty:
                return None
        chunk, self.buffer = self.buffer[:n], self.buffer[n:]
        return chunk

    def _drain_mic(self) -> np.ndarray:
        parts = []
        while not self.mic_q.empty():
            parts.append(self.mic_q.get_nowait())
        if parts:
            self.buffer = np.concatenate([self.buffer] + parts)
        take, self.buffer = self.buffer, np.zeros(0, dtype=np.int16)
        return take

    # ── audio out ────────────────────────────────────────────────

    def _beep(self, freq=880, ms=140):
        t = np.linspace(0, ms / 1000, int(TTS_SAMPLE_RATE * ms / 1000), False)
        tone = (0.25 * np.sin(2 * np.pi * freq * t) *
                np.hanning(t.size) * 32767).astype(np.int16)
        self.out.write(tone)

    def _speak_interruptible(self, pcm_chunks) -> bool:
        """Play PCM chunks; returns True if the user barged in."""
        self.wake.reset()
        for chunk in pcm_chunks:
            samples = np.frombuffer(chunk, dtype=np.int16)
            for i in range(0, samples.size, 2048):
                self.out.write(samples[i:i + 2048])
                if self._barge_in():
                    logger.info("barge-in: playback interrupted")
                    return True
        return False

    def _barge_in(self) -> bool:
        heard = self._drain_mic()
        for i in range(0, heard.size - CHUNK + 1, CHUNK):
            if self.wake.triggered(heard[i:i + CHUNK]):
                return True
        return False

    # ── main loop ────────────────────────────────────────────────

    def run(self):
        mode = f"say the wake word ('{self.wake.name}')" if self.wake.available \
            else "press Enter to talk"
        print(f"\n🎙  Live assistant on repo '{self.repo}' - {mode}. Ctrl+C to quit.\n")
        with sd.InputStream(samplerate=MIC_SAMPLE_RATE, channels=1, dtype="int16",
                            blocksize=CHUNK, callback=self._mic_cb):
            while True:
                if self._wait_for_wake():
                    self._beep()
                    utterance = self._capture_utterance()
                    if utterance is None:
                        continue
                    self._handle(utterance)

    def _wait_for_wake(self) -> bool:
        self.wake.reset()
        if not self.wake.available:
            input("⏎  press Enter to talk...")
            self._drain_mic()   # discard stale audio
            return True
        while True:
            frame = self._pull(CHUNK)
            if frame is not None and self.wake.triggered(frame):
                logger.info("wake word detected")
                return True

    def _capture_utterance(self) -> np.ndarray | None:
        print("… listening")
        self.vad.reset()
        collected, speech_frames, silence = [], 0, 0.0
        silence_limit = TRAILING_SILENCE_S
        start = time.monotonic()
        while time.monotonic() - start < MAX_UTTERANCE_S:
            frame = self._pull(vad_mod.FRAME)
            if frame is None:
                break
            collected.append(frame)
            if self.vad.is_speech(frame):
                speech_frames += 1
                silence = 0.0
            else:
                silence += vad_mod.FRAME / MIC_SAMPLE_RATE
                if speech_frames >= MIN_SPEECH_FRAMES and silence >= silence_limit:
                    break
        if speech_frames < MIN_SPEECH_FRAMES:
            print("   (no speech detected)")
            return None
        return np.concatenate(collected)

    def _handle(self, utterance: np.ndarray):
        trace = Trace(kind="ask", user=self.user["username"])
        with trace.stage("denoise"):
            cleaned = denoise(utterance, MIC_SAMPLE_RATE)
        with trace.stage("stt"):
            try:
                heard = stt.transcribe(wav_bytes(cleaned, MIC_SAMPLE_RATE),
                                       language=self.profile.get("language", ""),
                                       vocabulary=self.profile.get("vocabulary", ""))
            except Exception as exc:
                logger.error("STT failed: {}", exc)
                return
        question = heard["text"]
        if not question:
            print("   (empty transcript)")
            return
        print(f"🗣  you said [{heard['language']}, conf {heard['confidence']:.2f}]: {question}")

        events = pipeline.answer_stream(question, self.session_id, self.user,
                                        self.repo, trace)
        final = {}

        def tokens():
            for ev in events:
                if ev["type"] == "token":
                    print(ev["text"], end="", flush=True)
                    yield ev["text"]
                elif ev["type"] == "final":
                    final.update(ev)
        print("🤖 ", end="", flush=True)
        interrupted = self._speak_interruptible(
            tts.stream_sentences(tokens(),
                                 voice=self.profile.get("tts_voice", ""),
                                 speed=float(self.profile.get("speech_rate", 1.0))))
        print()
        if interrupted:
            self._beep(660)
            utterance = self._capture_utterance()
            if utterance is not None:
                self._handle(utterance)
            return
        if final:
            conf = final.get("confidence", {})
            cites = ", ".join(f"{c['path']}:{c['start']}-{c['end']}"
                              for c in final.get("citations", []))
            print(f"   confidence: {conf.get('band')} ({conf.get('score')}) | "
                  f"citations: {cites or '-'}")


def main():
    setup_logging()
    ap = argparse.ArgumentParser(description="live voice code assistant")
    ap.add_argument("--repo", required=True, help="indexed repo name")
    ap.add_argument("--username", default="dev")
    ap.add_argument("--password", default="dev123")
    args = ap.parse_args()

    db.get_conn()
    auth.ensure_default_users()
    account = auth.authenticate(args.username, args.password)
    if not account:
        print("login failed")
        sys.exit(1)
    user = {"username": account["username"], "role": account["role"],
            **ROLES.get(account["role"], ROLES["viewer"])}

    try:
        LiveAssistant(args.repo, user).run()
    except KeyboardInterrupt:
        print("\nbye 👋")


if __name__ == "__main__":
    main()
