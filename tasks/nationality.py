"""Task 4: Nationality Detection Model.

- Indian     -> nationality, emotion, age, dress colour
- US         -> age, emotion only
- African    -> emotion, dress colour only
- Other      -> nationality, emotion only

DeepFace's "race" output is used as a proxy for nationality
(it does not truly infer citizenship, only visual ethnicity cues —
noted as a limitation in the report).
"""

import numpy as np
import cv2
import streamlit as st

from utils.helpers import detect_faces, analyze_face

RACE_TO_NATIONALITY = {
    "indian": "Indian",
    "white": "United States",
    "latino hispanic": "Other",
    "middle eastern": "Other",
    "asian": "Other",
    "black": "African",
}


def _dominant_dress_colour(bgr_image, x, y, w, h):
    y1 = min(bgr_image.shape[0], y + h)
    y2 = min(bgr_image.shape[0], y + int(h * 3))
    region = bgr_image[y1:y2, x:x + w]
    if region.size == 0:
        return "Unknown"
    avg = region.reshape(-1, 3).mean(axis=0)  # B, G, R
    b, g, r = avg
    if r > g and r > b:
        return "Red-ish"
    if g > r and g > b:
        return "Green-ish"
    if b > r and b > g:
        return "Blue-ish"
    return "Neutral"


def run():
    st.title("Task 4: Nationality Detection Model")

    file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    if not file:
        st.info("Upload an image to run detection.")
        return

    frame = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    st.image(frame, channels="BGR", caption="Input preview")

    faces = detect_faces(frame)
    if len(faces) == 0:
        st.warning("No face detected.")
        return

    results = analyze_face(frame, actions=("age", "emotion", "race"))
    st.subheader("Output")
    for (x, y, w, h), res in zip(faces, results):
        race = res.get("dominant_race", "").lower()
        nationality = RACE_TO_NATIONALITY.get(race, "Other")
        emotion = res.get("dominant_emotion")
        age = res.get("age")
        dress = _dominant_dress_colour(frame, x, y, w, h)

        st.markdown(f"**Person at ({x},{y})**")
        if nationality == "Indian":
            st.write(f"Nationality: Indian | Age: {age} | Dress colour: {dress} | Emotion: {emotion}")
        elif nationality == "United States":
            st.write(f"Age: {age} | Emotion: {emotion}")
        elif nationality == "African":
            st.write(f"Emotion: {emotion} | Dress colour: {dress}")
        else:
            st.write(f"Nationality: {nationality} | Emotion: {emotion}")
