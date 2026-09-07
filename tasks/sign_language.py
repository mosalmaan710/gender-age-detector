"""Task 6: Sign Language Detection.

Supports both uploaded images and a live webcam snapshot. Hand landmarks are
extracted with MediaPipe and matched against a small set of known gestures
using finger-up/down state (a lightweight, explainable classifier rather
than a trained deep model).

This version removes time restrictions and guards MediaPipe usage so the
app does not crash if MediaPipe is unavailable in the runtime.
"""

import numpy as np
# Guarded cv2 import
try:
    import cv2
    _cv2_import_error = None
except Exception as e:
    cv2 = None
    _cv2_import_error = e

import streamlit as st

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
    try:
        fingers.append(1 if lm[4].x < lm[3].x else 0)  # thumb (mirrored heuristic)
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        for tip, pip in zip(tips, pips):
            fingers.append(1 if lm[tip].y < lm[pip].y else 0)
    except Exception:
        # If landmark indexing fails, return an impossible pattern
        return (0, 0, 0, 0, 0)
    return tuple(fingers)


def run():
    st.title("Task 6: Sign Language Detection")
    st.caption("Hand landmark recognition using MediaPipe. If MediaPipe is not installed in the runtime, this module will inform you and return.")

    if cv2 is None:
        st.error(
            f"OpenCV import failed: {_cv2_import_error!s}. Ensure `opencv-python-headless` is installed."
        )
        return

    # Guard MediaPipe import so failure doesn't crash the app
    try:
        import mediapipe as mp
    except Exception:
        st.error("MediaPipe is not available in this environment. Install `mediapipe` in requirements.txt to enable this task.")
        return

    mode = st.radio("Input source", ["Upload image", "Webcam snapshot"], horizontal=True)
    frame = None

    if mode == "Upload image":
        file = st.file_uploader("Upload sign image", type=["jpg", "jpeg", "png"])
        if file:
            try:
                data = file.read()
            except Exception:
                st.error("Could not read the uploaded file. Please try another image.")
                return

            if not data:
                st.error("Uploaded file is empty.")
                return

            try:
                frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            except Exception:
                st.error("Failed to decode the uploaded image. Please upload a valid image.")
                return
    else:
        shot = st.camera_input("Capture sign")
        if shot:
            try:
                data = shot.getvalue()
            except Exception:
                st.error("Could not access captured image data.")
                return

            if not data:
                st.error("Captured image is empty.")
                return

            try:
                frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            except Exception:
                st.error("Failed to decode the captured image. Try taking another photo.")
                return

    if frame is None:
        st.info("Provide an image to run detection.")
        return

    # Process with MediaPipe safely
    try:
        mp_hands = mp.solutions.hands
        with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except Exception:
                st.error("Failed to convert image for processing.")
                return

            try:
                result = hands.process(rgb)
            except Exception as e:
                st.error(f"Hand landmark processing failed: {e}")
                return
    except Exception as e:
        st.error(f"MediaPipe processing failed to initialize: {e}")
        return

    if not result or not getattr(result, "multi_hand_landmarks", None):
        st.warning("No hand detected.")
        st.image(frame, channels="BGR")
        return

    hand_landmarks = result.multi_hand_landmarks[0]
    state = _finger_states(hand_landmarks)
    word = KNOWN_WORDS.get(state, "Unrecognized gesture")

    try:
        mp_drawing = mp.solutions.drawing_utils
        annotated = frame.copy()
        mp_drawing.draw_landmarks(annotated, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        st.image(annotated, channels="BGR")
    except Exception:
        st.image(frame, channels="BGR")

    st.success(f"Recognized: **{word}**")
