"""Shared helpers: face detection, DeepFace wrapper, CSV logging."""

import os
import csv
from datetime import datetime

import cv2

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def detect_faces(bgr_image):
    """Return list of (x, y, w, h) face boxes."""
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    return faces


def analyze_face(bgr_image, actions=("age", "gender", "emotion", "race")):
    """
    Wraps DeepFace.analyze. DeepFace lazily downloads pretrained weights
    (VGG-Face / age-net / gender-net / emotion / race models) on first call.
    """
    from deepface import DeepFace

    results = DeepFace.analyze(
        img_path=bgr_image,
        actions=list(actions),
        enforce_detection=False,
        silent=True,
    )
    return results if isinstance(results, list) else [results]


def log_to_csv(path, row, header):
    """Append a row to a CSV file, creating it with a header if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)


def timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
