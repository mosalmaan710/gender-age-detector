"""Task 5: Car Colour Detection Model.

- Detects cars (Haar cascade) and classifies each as blue vs other.
- Blue car -> red bounding box. Other colour car -> blue bounding box.
- Also counts people present (HOG pedestrian detector) at the signal.

This version avoids runtime downloads and expects the cascade file to
be present at data/haarcascade_car.xml. If the file is missing the app
will show a clear error instead of attempting a network download.
"""

import os
import numpy as np
import cv2
import streamlit as st

CAR_CASCADE_PATH = "data/haarcascade_car.xml"


def _ensure_car_cascade():
    # Do NOT download at runtime. Require the cascade to be bundled in the repo.
    if not os.path.isfile(CAR_CASCADE_PATH):
        return None

    cascade = cv2.CascadeClassifier(CAR_CASCADE_PATH)
    # CascadeClassifier.empty() returns True if loading failed
    if cascade.empty():
        return None

    return cascade


def _is_blue(bgr_image, x, y, w, h):
    try:
        roi = bgr_image[y:y + h, x:x + w]
        if roi.size == 0:
            return False
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (100, 60, 60), (130, 255, 255))
        return (np.count_nonzero(mask) / mask.size) > 0.12
    except Exception:
        return False


def run():
    st.title("Task 5: Car Colour Detection Model")
    st.caption("Blue cars -> red box | other colours -> blue box | counts people at the signal.")

    file = st.file_uploader("Upload traffic-signal image", type=["jpg", "jpeg", "png"])
    if not file:
        st.info("Upload an image to run detection.")
        return

    # Read and validate bytes
    try:
        data = file.read()
    except Exception:
        st.error("Could not read the uploaded file. Please try a different image.")
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

    st.image(frame, channels="BGR", caption="Input preview")

    car_cascade = _ensure_car_cascade()
    if car_cascade is None:
        st.error(
            "Car cascade file is missing or failed to load.\n"
            "Please add the OpenCV Haar cascade to data/haarcascade_car.xml in the repository.\n"
            "This file should be the standard OpenCV cascade for car detection (not created here)."
        )
        return

    annotated = frame.copy()

    # Convert to gray safely
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        st.error(f"Failed to process the image for detection: {e}")
        return

    car_count = 0
    try:
        cars = car_cascade.detectMultiScale(gray, 1.1, 3, minSize=(40, 40))
        if cars is None:
            cars = []
        for (x, y, w, h) in cars:
            car_count += 1
            blue = _is_blue(frame, x, y, w, h)
            color = (0, 0, 255) if blue else (255, 0, 0)  # BGR: red / blue
            try:
                cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            except Exception:
                pass
    except Exception as e:
        st.warning(f"Car detection failed: {e}")

    # People detection (HOG) guarded
    people_count = 0
    try:
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        people, _ = hog.detectMultiScale(frame, winStride=(8, 8))
        if people is None:
            people = []
        for (x, y, w, h) in people:
            people_count += 1
            try:
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            except Exception:
                pass
    except Exception as e:
        # HOG may fail in some headless builds; show a non-fatal warning
        st.warning(f"People detection failed or is unsupported in this environment: {e}")

    try:
        st.image(annotated, channels="BGR", caption="Detection result")
    except Exception:
        st.image(frame, channels="BGR")

    st.write(f"Cars detected: **{car_count}**")
    st.write(f"People present: **{people_count}**")
