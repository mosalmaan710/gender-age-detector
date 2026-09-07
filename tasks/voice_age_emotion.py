"""Task 2: Age & Emotion Detection through Voice.

Only processes male voices (rejects female voices with a message).
- Age > 60  -> mark as senior citizen + detect emotion
- Age <= 60 -> detect age only

Gender/age are estimated from pitch (F0) and MFCC statistics — a
lightweight signal-processing heuristic, since training a full deep
audio model needs a large labelled speech-age dataset outside this
project's scope. Emotion uses a small MFCC-based classifier.
"""

import numpy as np
import streamlit as st

MALE_PITCH_MAX_HZ = 165  # typical male fundamental frequency ceiling


def _extract_features(y, sr):
    import librosa
    f0, _, _ = librosa.pyin(y, fmin=50, fmax=400, sr=sr)
    f0 = f0[~np.isnan(f0)]
    mean_pitch = float(np.mean(f0)) if len(f0) else 999.0
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return mean_pitch, mfcc


def _estimate_age(mean_pitch, mfcc):
    # Heuristic: lower pitch + lower MFCC variance -> older voice.
    variance = float(np.var(mfcc))
    base_age = 60 - (mean_pitch - 90) * 0.4
    age = base_age - variance * 0.05
    return int(max(15, min(90, age)))


def _estimate_emotion(mfcc):
    energy = float(np.mean(np.abs(mfcc)))
    if energy > 40:
        return "Happy/Excited"
    elif energy > 25:
        return "Neutral"
    else:
        return "Sad/Calm"


def run():
    st.title("Task 2: Age & Emotion Detection through Voice")
    st.write("Upload a voice note (male voices only).")

    file = st.file_uploader("Upload audio", type=["wav", "mp3", "m4a"])
    if not file:
        st.info("Upload an audio clip to analyze.")
        return

    try:
        import librosa
    except ImportError:
        st.error("Install `librosa` (see requirements.txt) to run this module.")
        return

    y, sr = librosa.load(file, sr=16000, mono=True)
    mean_pitch, mfcc = _extract_features(y, sr)

    if mean_pitch > MALE_PITCH_MAX_HZ:
        st.error("Upload male voice.")
        return

    age = _estimate_age(mean_pitch, mfcc)
    st.success(f"Estimated age: **{age}**")

    if age > 60:
        st.info("Marked as **Senior Citizen**.")
        emotion = _estimate_emotion(mfcc)
        st.write(f"Detected emotion: **{emotion}**")
    else:
        st.caption("Age is 60 or below — only age is reported per task spec.")
