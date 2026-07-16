"""Sign-language recognition sidecar (multimodal gesture input).

Ported from the SignSpeak project (D:\\GenAI Prac): MediaPipe hand landmarks
+ RandomForest classifiers recognize 24 ASL fingerspelling letters and the
two-hand words HELLO / THANK YOU / PLEASE in real time from webcam frames.

Runs as a separate process on Python 3.10 (MediaPipe requirement - the main
assistant venv is 3.13) and is embedded in the Streamlit UI as the
"Sign Language" tab. Sentences built by signing can be sent to the assistant
as questions - a full non-voice input path (accessibility / multimodal
fallback).

Run with the SignSpeak interpreter:
    "D:\\GenAI Prac\\.venv\\Scripts\\python.exe" app.py
"""
import base64
import os
import threading
import time

import cv2
import joblib
import numpy as np
import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

import mediapipe as mp

from hand_utils import (
    GESTURE_MODEL_PATH,
    TWO_HAND_MODEL_PATH,
    create_landmarker,
    landmarks_to_feature_vector,
    two_hand_feature_vector,
)

# project-root .env (two levels up) so all keys live in one place
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

PORT = int(os.environ.get("SIGNLANG_PORT", "5055"))
CONFIDENCE_THRESHOLD = 0.75

app = Flask(__name__)


def load_model(path):
    if not os.path.exists(path):
        return None
    return joblib.load(path)["model"]


def predict(result, letter_clf, two_hand_clf):
    """Return (label, label_type, confidence) for the current detection."""
    hands = result.hand_landmarks
    if len(hands) == 2 and two_hand_clf is not None:
        handedness_labels = {h[0].category_name for h in result.handedness}
        if handedness_labels == {"Left", "Right"}:
            feature_vector = two_hand_feature_vector(hands, result.handedness)
            probs = two_hand_clf.predict_proba([feature_vector])[0]
            best_idx = probs.argmax()
            confidence = probs[best_idx]
            if confidence >= CONFIDENCE_THRESHOLD:
                return two_hand_clf.classes_[best_idx], "word", confidence
            return None, None, confidence

    if len(hands) >= 1:
        handedness_label = result.handedness[0][0].category_name
        feature_vector = landmarks_to_feature_vector(hands[0], handedness_label)
        probs = letter_clf.predict_proba([feature_vector])[0]
        best_idx = probs.argmax()
        confidence = probs[best_idx]
        if confidence >= CONFIDENCE_THRESHOLD:
            return letter_clf.classes_[best_idx], "letter", confidence
        return None, None, confidence

    return None, None, 0.0


letter_clf = load_model(GESTURE_MODEL_PATH)
two_hand_clf = load_model(TWO_HAND_MODEL_PATH)

# VIDEO-mode landmarker with synthesized strictly-increasing timestamps; the
# lock also serializes access (the landmarker is not thread-safe).
_landmarker = create_landmarker(num_hands=2)
_landmarker_lock = threading.Lock()
_last_ts_ms = 0


# ---------------------------------------------------------------- chatbot

SYSTEM_PROMPT = """You are Signy, the sign-language assistant embedded in a voice-enabled \
code assistant. You are an expert on sign languages and hand gestures.

You ONLY answer questions related to:
- Sign languages (ASL, BSL, ISL, and others), their grammar, history, and dialects
- Fingerspelling and how specific letters, numbers, words, or phrases are signed
- Hand gestures, gesture recognition technology, and hand-tracking (MediaPipe, landmarks, classifiers)
- Deaf culture, accessibility, and communication with deaf or hard-of-hearing people
- How to use this tab (hold a sign steady in front of the webcam to commit it; 24 static ASL \
letters A-Y excluding J and Z, plus the two-hand words HELLO, THANK YOU, PLEASE).

If the user asks about anything outside these topics, politely decline in one short sentence.
Style: warm, encouraging, concise."""

PROVIDERS = [
    {"name": "Groq", "env": "GROQ_API_KEY",
     "url": "https://api.groq.com/openai/v1/chat/completions",
     "model": "llama-3.3-70b-versatile"},
    {"name": "OpenAI", "env": "OPENAI_API_KEY",
     "url": "https://api.openai.com/v1/chat/completions",
     "model": "gpt-4o-mini"},
]

_preferred_provider = {"index": None}


def _call_provider(provider, messages):
    api_key = os.environ.get(provider["env"], "").strip()
    if not api_key:
        raise RuntimeError(f"no {provider['env']} in .env")
    resp = requests.post(
        provider["url"],
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": provider["model"],
              "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
              "temperature": 0.6, "max_tokens": 700},
        timeout=45)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def ask_llm(messages):
    order = list(range(len(PROVIDERS)))
    if _preferred_provider["index"] in order:
        order.remove(_preferred_provider["index"])
        order.insert(0, _preferred_provider["index"])
    errors = []
    for i in order:
        try:
            reply = _call_provider(PROVIDERS[i], messages)
            _preferred_provider["index"] = i
            return reply, PROVIDERS[i]["name"]
        except Exception as exc:  # noqa: BLE001 - fall through the chain
            errors.append(f"{PROVIDERS[i]['name']}: {exc}")
    raise RuntimeError(" | ".join(errors))


@app.post("/api/chat")
def api_chat():
    body = request.get_json(silent=True) or {}
    messages = [
        {"role": m["role"], "content": str(m["content"])[:4000]}
        for m in body.get("messages", [])
        if isinstance(m, dict) and m.get("role") in ("user", "assistant") and m.get("content")
    ][-16:]
    if not messages:
        return jsonify({"error": "empty message"}), 400
    try:
        reply, provider = ask_llm(messages)
        return jsonify({"reply": reply, "provider": provider})
    except RuntimeError as exc:
        return jsonify({"error": f"All providers failed. {exc}"}), 502


# ------------------------------------------------------------- recognition

@app.post("/api/predict")
def api_predict():
    body = request.get_json(silent=True) or {}
    frame_b64 = body.get("frame", "")
    if "," in frame_b64:
        frame_b64 = frame_b64.split(",", 1)[1]
    try:
        buf = np.frombuffer(base64.b64decode(frame_b64), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        img = None
    if img is None:
        return jsonify({"error": "bad frame"}), 400

    frame = cv2.flip(img, 1)   # mirror to match the training pipeline
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    global _last_ts_ms
    with _landmarker_lock:
        ts = max(_last_ts_ms + 1, int(time.monotonic() * 1000))
        _last_ts_ms = ts
        result = _landmarker.detect_for_video(mp_image, ts)

    label, label_type, confidence = predict(result, letter_clf, two_hand_clf)
    hands = [[{"x": lm.x, "y": lm.y} for lm in hand_landmarks]
             for hand_landmarks in result.hand_landmarks]
    return jsonify({"label": label, "type": label_type,
                    "confidence": round(float(confidence), 3), "hands": hands})


@app.get("/api/health")
def api_health():
    return jsonify({
        "letter_model": letter_clf is not None,
        "two_hand_model": two_hand_clf is not None,
        "providers": [p["name"] for p in PROVIDERS if os.environ.get(p["env"], "").strip()],
    })


@app.get("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    if letter_clf is None:
        print(f"WARNING: no letter model at {GESTURE_MODEL_PATH} - prediction disabled.")
    print(f"Sign-language service running on http://127.0.0.1:{PORT}")
    app.run(host="127.0.0.1", port=PORT, threaded=True)
