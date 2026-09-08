"""Task 3: Long Hair Identification.

Spec (deliberately counter-intuitive, as given by the task):
  - Applies ONLY to people whose predicted age is between 20 and 30.
  - Within that age band: long hair -> labelled Female (even if male),
    short hair -> labelled Male (even if female).
  - Outside 20-30: report the model's normal, unmodified gender prediction.

Hair length is estimated heuristically from the proportion of
hair-coloured pixels below the detected face box.
"""

import numpy as np
import cv2
import streamlit as st

from utils.helpers import detect_faces, analyze_face

HAIR_LOWER = np.array([0, 0, 0])
HAIR_UPPER = np.array([180, 255, 90])  # dark pixels ~ hair, HSV


def _hair_length_ratio(bgr_image, x, y, w, h):
    height, width = bgr_image.shape[:2]

    x1 = max(0, x)
    x2 = min(width, x + w)
    y1 = min(height, y + h)
    y2 = min(height, y + int(h * 1.8))

    if x1 >= x2 or y1 >= y2:
        return 0.0

    region = bgr_image[y1:y2, x1:x2]

    if region.size == 0:
        return 0.0

    # Keep the existing HSV/hair analysis code below this point.


def run():
    st.title("Task 3: Long Hair Identification")
    st.caption(
        "Ages 20-30: long hair -> labelled Female, short hair -> labelled Male "
        "(overrides the true prediction, per task spec). Outside that range: "
        "normal gender prediction is shown."
    )

    file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    if not file:
        st.info("Upload an image to run detection.")
        return

    # Read and validate image bytes
    try:
        data = file.read()
    except Exception:
        st.error("Could not read the uploaded file bytes. Please try a different image.")
        return

    if not data:
        st.error("Uploaded file is empty.")
        return

    try:
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        st.error("Failed to decode the uploaded image. Please upload a valid image file.")
        return

    if frame is None:
        st.error("Could not decode the uploaded image. Please upload a valid JPG/PNG image.")
        return

    # Face detection
    try:
        faces = detect_faces(frame)
    except Exception as e:
        st.error(f"Face detection failed: {e}")
        return

    if faces is None or len(faces) == 0:
        st.warning("No face detected.")
        st.image(frame, channels="BGR")
        return

    # DeepFace analysis protected
    try:
        results = analyze_face(frame, actions=("age", "gender"))
    except Exception as e:
        st.error(f"Age/gender detection failed: {e}")
        return

    if not isinstance(results, list):
        results = [results]

    annotated = frame.copy()

    for i, (x, y, w, h) in enumerate(faces):
        res = results[i] if i < len(results) else {}
        age = res.get("age") if isinstance(res, dict) else None
        true_gender = res.get("dominant_gender") if isinstance(res, dict) else None

        label = true_gender
        try:
            if age is not None and isinstance(age, (int, float)) and 20 <= age <= 30:
                ratio = _hair_length_ratio(frame, x, y, w, h)
                label = "Female" if ratio > 0.15 else "Male"
        except Exception:
            # If hair heuristic fails, fall back to true_gender
            label = true_gender

        try:
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            display_age = age if age is not None else "?"
            display_gender = label if label is not None else "?"
            cv2.putText(annotated, f"{display_gender}, {display_age}y", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        except Exception:
            # Drawing errors should not crash the app
            pass

    try:
        st.image(annotated, channels="BGR")
    except Exception:
        st.image(frame, channels="BGR")
