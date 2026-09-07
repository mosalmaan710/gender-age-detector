"""Shared helpers: face detection, DeepFace wrapper, CSV logging."""

import os
import csv
from datetime import datetime

# Guard cv2 import so a missing or broken OpenCV wheel doesn't crash app import
try:
    import cv2
    _cv2_import_error = None
except Exception as e:
    cv2 = None
    _cv2_import_error = e

_FACE_CASCADE = None


def _ensure_face_cascade():
    """Lazily load the Haar cascade classifier. Return None if OpenCV is unavailable or loading failed."""
    global _FACE_CASCADE
    if _FACE_CASCADE is not None:
        return _FACE_CASCADE

    if cv2 is None:
        return None

    try:
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if cascade.empty():
            return None
        _FACE_CASCADE = cascade
        return _FACE_CASCADE
    except Exception:
        return None


def detect_faces(bgr_image):
    """Return list of (x, y, w, h) face boxes.

    Raises RuntimeError with a user-friendly message if OpenCV is missing or cascade fails to load.
    """
    if cv2 is None:
        raise RuntimeError(f"OpenCV is not available in this environment: {_cv2_import_error!s}. Ensure opencv-python-headless is installed in requirements.txt.")

    cascade = _ensure_face_cascade()
    if cascade is None:
        raise RuntimeError("Failed to load face cascade. Ensure OpenCV haarcascade files are available.")

    # Validate image
    if bgr_image is None or getattr(bgr_image, "size", 0) == 0:
        return []

    try:
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    except Exception:
        return []

    try:
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        return faces if faces is not None else []
    except Exception:
        return []


def analyze_face(bgr_image, actions=("age", "gender", "emotion", "race")):
    """
    Wraps DeepFace.analyze. DeepFace lazily downloads pretrained weights on first call.
    This function keeps DeepFace usage local so import issues are isolated.
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
