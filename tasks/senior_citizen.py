"""Task 1: Senior Citizen Identification.

Detects every person in an uploaded image / real-time webcam feed,
predicts age + gender, flags anyone over 60 as a Senior Citizen,
and logs age, gender, and time of visit to a CSV file.
"""

import numpy as np
import cv2
import pandas as pd
import streamlit as st

from utils.helpers import detect_faces, analyze_face, log_to_csv, timestamp

LOG_PATH = "data/senior_citizen_log.csv"
HEADER = ["timestamp", "age", "gender", "senior_citizen"]


def run():
    st.title("Task 1: Senior Citizen Identification")
    st.write(
        "Detects multiple people in a mall/store camera feed, predicts age & gender, "
        "and flags anyone over **60** as a senior citizen. Every detection is logged."
    )

    mode = st.radio("Input source", ["Upload image", "Webcam snapshot"], horizontal=True)
    frame = None
    if mode == "Upload image":
        file = st.file_uploader("Upload store/mall footage frame", type=["jpg", "jpeg", "png"])
        if file:
            frame = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    else:
        shot = st.camera_input("Capture frame")
        if shot:
            frame = cv2.imdecode(np.frombuffer(shot.getvalue(), np.uint8), cv2.IMREAD_COLOR)

    if frame is None:
        st.info("Provide an image to run detection.")
    else:
        faces = detect_faces(frame)
        if len(faces) == 0:
            st.warning("No person detected.")
        else:
            results = analyze_face(frame, actions=("age", "gender"))
            annotated = frame.copy()
            for (x, y, w, h), res in zip(faces, results):
                age = res.get("age")
                gender = res.get("dominant_gender")
                is_senior = age is not None and age > 60
                color = (0, 165, 255) if is_senior else (0, 255, 0)
                cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
                tag = "SENIOR" if is_senior else ""
                cv2.putText(annotated, f"{gender},{age} {tag}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                log_to_csv(LOG_PATH, [timestamp(), age, gender, is_senior], HEADER)

            st.image(annotated, channels="BGR")

    st.subheader("Visit log")
    try:
        df = pd.read_csv(LOG_PATH)
        st.dataframe(df.tail(50))
        st.download_button("Download full log (CSV)", df.to_csv(index=False), "senior_citizen_log.csv")
    except FileNotFoundError:
        st.caption("No entries logged yet.")
