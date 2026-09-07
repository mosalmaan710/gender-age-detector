"""Task 5: Car Colour Detection Model.

- Detects cars (Haar cascade) and classifies each as blue vs other.
- Blue car -> red bounding box. Other colour car -> blue bounding box.
- Also counts people present (HOG pedestrian detector) at the signal.
"""

import os
import urllib.request

import numpy as np
import cv2
import streamlit as st

CAR_CASCADE_PATH = "data/haarcascade_car.xml"
CAR_CASCADE_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/master/data/"
    "haarcascades/haarcascade_car.xml"
)


def _ensure_car_cascade():
    if not os.path.isfile(CAR_CASCADE_PATH):
        os.makedirs("data", exist_ok=True)
        try:
            urllib.request.urlretrieve(CAR_CASCADE_URL, CAR_CASCADE_PATH)
        except Exception:
            return None
    return cv2.CascadeClassifier(CAR_CASCADE_PATH)


def _is_blue(bgr_image, x, y, w, h):
    roi = bgr_image[y:y + h, x:x + w]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (100, 60, 60), (130, 255, 255))
    return (np.count_nonzero(mask) / mask.size) > 0.12


def run():
    st.title("Task 5: Car Colour Detection Model")
    st.caption("Blue cars -> red box | other colours -> blue box | counts people at the signal.")

    file = st.file_uploader("Upload traffic-signal image", type=["jpg", "jpeg", "png"])
    if not file:
        st.info("Upload an image to run detection.")
        return

    frame = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    st.image(frame, channels="BGR", caption="Input preview")

    car_cascade = _ensure_car_cascade()
    annotated = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    car_count = 0
    if car_cascade is not None:
        cars = car_cascade.detectMultiScale(gray, 1.1, 3, minSize=(40, 40))
        for (x, y, w, h) in cars:
            car_count += 1
            blue = _is_blue(frame, x, y, w, h)
            color = (0, 0, 255) if blue else (255, 0, 0)  # BGR: red / blue
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    people, _ = hog.detectMultiScale(frame, winStride=(8, 8))
    for (x, y, w, h) in people:
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

    st.image(annotated, channels="BGR", caption="Detection result")
    st.write(f"Cars detected: **{car_count}**")
    st.write(f"People present: **{len(people)}**")
