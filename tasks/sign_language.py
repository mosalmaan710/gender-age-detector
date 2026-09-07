"""Task 6: Sign Language Detection.

Operational window: 6 PM - 10 PM only (per spec). Supports both
uploaded images and a live webcam snapshot. Hand landmarks are
extracted with MediaPipe and matched against a small set of known
gestures using finger-up/down state (a lightweight, explainable
classifier rather than a trained deep model, since a labelled sign
dataset was outside the scope of this build).
"""

from datetime import datetime

import numpy as np
import cv2
import streamlit as st

OPEN_HOUR, CLOSE_HOUR = 18, 22  # 6 PM - 10 PM

KNOWN_WORDS = {
    (0, 0, 0, 0, 0): "Fist / Stop",
    (1, 1, 1, 1, 1): "Hello / Open palm",
    (0, 1, 0, 0, 0): "One",
    (0, 1, 1, 0, 0): "Peace / Two",
    (1, 0, 0, 0, 1): "Call me",
}


def _finger_states(hand_landmarks):
    lm = hand_landmarks.landmark
    fingers = []
    fingers.append(1 if lm[4].x < lm[3].x else 0)  # thumb (mirrored heuristic)
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    for tip, pip in zip(tips, pips):
        fingers.append(1 if lm[tip].y < lm[pip].y else 0)
    return tuple(fingers)


def run():
    st.title("Task 6: Sign Language Detection")

    now = datetime.now()
    if not (OPEN_HOUR <= now.hour < CLOSE_HOUR):
        st.error(f"This module only operates between {OPEN_HOUR}:00 and {CLOSE_HOUR}:00. "
                  f"Current time: {now.strftime('%H:%M')}.")
        return

    try:
        import mediapipe as mp
    except ImportError:
        st.error("Install `mediapipe` (see requirements.txt) to run this module.")
        return

    mode = st.radio("Input source", ["Upload image", "Webcam snapshot"], horizontal=True)
    frame = None
    if mode == "Upload image":
        file = st.file_uploader("Upload sign image", type=["jpg", "jpeg", "png"])
        if file:
            frame = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    else:
        shot = st.camera_input("Capture sign")
        if shot:
            frame = cv2.imdecode(np.frombuffer(shot.getvalue(), np.uint8), cv2.IMREAD_COLOR)

    if frame is None:
        st.info("Provide an image to run detection.")
        return

    mp_hands = mp.solutions.hands
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

    if not result.multi_hand_landmarks:
        st.warning("No hand detected.")
        st.image(frame, channels="BGR")
        return

    hand_landmarks = result.multi_hand_landmarks[0]
    state = _finger_states(hand_landmarks)
    word = KNOWN_WORDS.get(state, "Unrecognized gesture")

    mp_drawing = mp.solutions.drawing_utils
    annotated = frame.copy()
    mp_drawing.draw_landmarks(annotated, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    st.image(annotated, channels="BGR")
    st.success(f"Recognized: **{word}**")
