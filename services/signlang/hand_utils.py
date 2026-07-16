"""Shared helpers for hand landmark detection and feature extraction.

Used by collect_data.py, train_model.py, and main.py so all three agree
on exactly how a MediaPipe hand detection turns into a feature vector.
"""
import os

import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "hand_landmarker.task")
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "landmarks.csv")
GESTURE_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "gesture_model.pkl")
TWO_HAND_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "two_hand_landmarks.csv")
TWO_HAND_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "two_hand_gesture_model.pkl")

# Static ASL fingerspelling letters. J and Z are excluded because they are
# drawn as motions in real ASL, not held poses, so a single-frame landmark
# classifier cannot represent them.
LETTERS = [c for c in "ABCDEFGHIKLMNOPQRSTUVWXY"]

NUM_LANDMARKS = 21


def create_landmarker(num_hands=1):
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=num_hands,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    return vision.HandLandmarker.create_from_options(options)


def _normalized_points(hand_landmarks):
    """One hand's 21 landmarks, translated to a wrist-relative origin and
    scale-normalized by the largest distance from the wrist so the result
    doesn't depend on hand size or distance from the camera.
    """
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float64)
    pts -= pts[0]
    scale = np.linalg.norm(pts, axis=1).max()
    if scale > 1e-6:
        pts /= scale
    return pts


def landmarks_to_feature_vector(hand_landmarks, handedness_label):
    """Convert one hand's landmarks into a translation-, scale-, and
    left/right-invariant 63-dim feature vector, for single-hand
    fingerspelling letters.
    """
    pts = _normalized_points(hand_landmarks)

    # Mirror left hands onto the right-hand coordinate space so one model
    # covers both hands (fingerspelling shapes are mirror-symmetric).
    if handedness_label == "Left":
        pts[:, 0] *= -1

    return pts.flatten()


def two_hand_feature_vector(hand_landmarks_list, handedness_list):
    """Convert a two-hand detection into a fixed 126-dim vector: the left
    hand's 63 features followed by the right hand's, each independently
    translation- and scale-normalized. Unlike the single-hand letters,
    left/right is NOT mirrored away here, since which hand does what is
    part of the gesture for two-handed signs. A missing hand fills its
    half with zeros.
    """
    slots = {"Left": np.zeros(NUM_LANDMARKS * 3), "Right": np.zeros(NUM_LANDMARKS * 3)}
    for hand_landmarks, handedness in zip(hand_landmarks_list, handedness_list):
        label = handedness[0].category_name
        if label in slots:
            slots[label] = _normalized_points(hand_landmarks).flatten()
    return np.concatenate([slots["Left"], slots["Right"]])


def draw_landmarks(image, hand_landmarks_list):
    for hand_landmarks in hand_landmarks_list:
        vision.drawing_utils.draw_landmarks(
            image,
            hand_landmarks,
            vision.HandLandmarksConnections.HAND_CONNECTIONS,
            vision.drawing_styles.get_default_hand_landmarks_style(),
            vision.drawing_styles.get_default_hand_connections_style(),
        )
