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
    y2 = min(bgr_image.shape[0], y + int(h * 1.8))
    region = bgr_image[y + h: y2, x:x + w]
    if region.size == 0:
        return 0.0
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HAIR_LOWER, HAIR_UPPER)
    return float(np.count_nonzero(mask)) / mask.size


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

    frame = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    faces = detect_faces(frame)
    if len(faces) == 0:
        st.warning("No face detected.")
        return

    results = analyze_face(frame, actions=("age", "gender"))
    annotated = frame.copy()
    for (x, y, w, h), res in zip(faces, results):
        age = res.get("age")
        true_gender = res.get("dominant_gender")
        label = true_gender
        if age is not None and 20 <= age <= 30:
            ratio = _hair_length_ratio(frame, x, y, w, h)
            label = "Female" if ratio > 0.15 else "Male"
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(annotated, f"{label}, {age}y", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    st.image(annotated, channels="BGR")
