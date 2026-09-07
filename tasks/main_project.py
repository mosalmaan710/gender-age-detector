"""Main Project: Real-Time Application for Gender and Age Detection."""

import numpy as np
import cv2
import streamlit as st

from utils.helpers import detect_faces, analyze_face


def run():
    st.title("Main Project: Real-Time Gender & Age Detector")
    st.write(
        "Upload an image, or use your webcam snapshot, to detect every face's "
        "**age** and **gender** in real time."
    )

    mode = st.radio("Input source", ["Upload image", "Webcam snapshot"], horizontal=True)

    frame = None
    if mode == "Upload image":
        file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if file:
            data = np.frombuffer(file.read(), np.uint8)
            frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    else:
        shot = st.camera_input("Take a snapshot")
        if shot:
            data = np.frombuffer(shot.getvalue(), np.uint8)
            frame = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if frame is None:
        st.info("Provide an image to run detection.")
        return

    faces = detect_faces(frame)
    if len(faces) == 0:
        st.warning("No face detected.")
        st.image(frame, channels="BGR")
        return

    try:
        results = analyze_face(frame, actions=("age", "gender"))
    except Exception as e:
        st.error(f"Model inference failed: {e}")
        return

    annotated = frame.copy()
    for (x, y, w, h), res in zip(faces, results):
        gender = res.get("dominant_gender", "N/A")
        age = res.get("age", "N/A")
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{gender}, {age}y"
        cv2.putText(annotated, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    st.image(annotated, channels="BGR", caption="Detection result")

    st.subheader("Detected faces")
    for i, res in enumerate(results, start=1):
        st.write(f"**Face {i}** — Age: {res.get('age')} | Gender: {res.get('dominant_gender')}")
