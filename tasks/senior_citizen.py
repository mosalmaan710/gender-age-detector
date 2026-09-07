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
            # read bytes safely
            try:
                data = file.read()
            except Exception:
                st.error("Could not read the uploaded file bytes.")
                return

            if not data:
                st.error("Uploaded file is empty.")
                return

            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                st.error("Could not decode the uploaded image. Please upload a valid JPG/PNG image.")
                return

    else:
        shot = st.camera_input("Capture frame")
        if shot:
            try:
                data = shot.getvalue()
            except Exception:
                st.error("Could not access the captured image data.")
                return

            if not data:
                st.error("Captured image is empty.")
                return

            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                st.error("Could not decode the captured image. Try taking another photo.")
                return

    # At this point frame is either a valid image (ndarray) or None
    if frame is None:
        st.info("Provide an image to run detection.")
        return

    # Detect faces safely
    try:
        faces = detect_faces(frame)
    except Exception as e:
        st.error(f"Face detection failed: {e}")
        return

    if faces is None or len(faces) == 0:
        st.warning("No person detected.")
        st.image(frame, channels="BGR")
        return

    # Analyze faces with DeepFace (protected)
    try:
        results = analyze_face(frame, actions=("age", "gender"))
    except Exception as e:
        st.error(f"Age/gender detection failed: {e}")
        return

    # Ensure results is a list we can index
    if not isinstance(results, list):
        results = [results]

    annotated = frame.copy()

    # Iterate over detected faces and corresponding results if available
    for i, (x, y, w, h) in enumerate(faces):
        res = results[i] if i < len(results) else {}
        age = res.get("age") if isinstance(res, dict) else None
        gender = res.get("dominant_gender") if isinstance(res, dict) else None

        is_senior = age is not None and isinstance(age, (int, float)) and age > 60
        color = (0, 165, 255) if is_senior else (0, 255, 0)

        try:
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            tag = "SENIOR" if is_senior else ""
            cv2.putText(annotated, f"{gender},{age} {tag}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        except Exception:
            # Drawing should not crash the app
            pass

        # Attempt to log; failure should not stop the app
        try:
            log_to_csv(LOG_PATH, [timestamp(), age, gender, is_senior], HEADER)
        except Exception:
            st.warning("Failed to write to log; continuing without persisting this detection.")

    # Show annotated image
    try:
        st.image(annotated, channels="BGR")
    except Exception:
        # If displaying fails, show the original frame as a fallback
        st.image(frame, channels="BGR")

    # Visit log display
    st.subheader("Visit log")
    try:
        df = pd.read_csv(LOG_PATH)
        st.dataframe(df.tail(50))
        st.download_button("Download full log (CSV)", df.to_csv(index=False), "senior_citizen_log.csv")
    except FileNotFoundError:
        st.caption("No entries logged yet.")
    except Exception:
        st.warning("Could not load visit log.")
