"""Task 2: Age & Emotion Detection through Voice.

Only processes male voices (rejects female voices with a message).
- Age > 60  -> mark as senior citizen + detect emotion
- Age <= 60 -> detect age only

Gender/age are estimated from pitch (F0) and MFCC statistics — a
lightweight signal-processing heuristic, since training a full deep
audio model needs a large labelled speech-age dataset outside this
project's scope. This module is hardened for Streamlit Cloud.
"""

import io
import numpy as np
import streamlit as st

MALE_PITCH_MAX_HZ = 165  # typical male fundamental frequency ceiling


def _extract_features(y, sr):
    """Extract pitch (f0) and MFCCs from audio. Returns (median_pitch_or_None, mfcc).

    - If pitch detection finds no valid f0 values, returns (None, mfcc).
    - Uses median of f0 values for robustness.
    """
    try:
        import librosa
    except Exception as e:
        raise RuntimeError("librosa is required for audio processing") from e

    # pitch extraction using pyin; may return None or array with NaNs
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=50, fmax=400, sr=sr)
    except Exception:
        # Pitch extraction failed; still attempt to compute MFCCs
        f0 = None

    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    except Exception:
        mfcc = None

    if f0 is None:
        return None, mfcc

    # remove NaNs
    f0_valid = f0[~np.isnan(f0)]
    if f0_valid.size == 0:
        return None, mfcc

    median_pitch = float(np.median(f0_valid))
    return median_pitch, mfcc


def _estimate_age(mean_pitch, mfcc):
    # Heuristic: lower pitch + lower MFCC variance -> older voice.
    # If mfcc is None, fall back to pitch-only heuristic.
    try:
        variance = float(np.var(mfcc)) if mfcc is not None else 0.0
    except Exception:
        variance = 0.0

    base_age = 60 - (mean_pitch - 90) * 0.4
    age = base_age - variance * 0.05
    return int(max(15, min(90, age)))


def _estimate_emotion(mfcc):
    if mfcc is None:
        return "Unknown"
    try:
        energy = float(np.mean(np.abs(mfcc)))
    except Exception:
        return "Unknown"

    if energy > 40:
        return "Happy/Excited"
    elif energy > 25:
        return "Neutral"
    else:
        return "Sad/Calm"


def run():
    st.title("Task 2: Age & Emotion Detection through Voice")
    st.write("Upload a voice note (male voices only). For best results, use WAV files.")

    # Prefer WAV for Streamlit Cloud reliability
    file = st.file_uploader("Upload audio (WAV recommended)", type=["wav"])
    if not file:
        st.info("Upload an audio clip to analyze.")
        return

    # Ensure librosa is available
    try:
        import librosa  # noqa: F401
    except Exception:
        st.error("Install `librosa` (see requirements.txt) to run this module.")
        return

    # Read bytes and load audio safely
    try:
        data = file.read()
    except Exception as e:
        st.error(f"Could not read uploaded file: {e}")
        return

    if not data:
        st.error("Uploaded audio is empty.")
        return

    try:
        # Use BytesIO so librosa/soundfile can read it as a file-like object
        audio_buf = io.BytesIO(data)
        # librosa.load accepts file-like objects (via soundfile) when readable/seekable
        y, sr = librosa.load(audio_buf, sr=16000, mono=True)
    except Exception as e:
        st.error(f"Could not load audio file: {e}")
        return

    # Validate audio content
    if y is None or y.size == 0:
        st.error("The audio file could not be decoded or is empty. Please upload a valid WAV file.")
        return

    # Extract features (pitch + mfcc) with robust handling
    try:
        mean_pitch, mfcc = _extract_features(y, sr)
    except Exception as e:
        st.error(f"Failed to extract audio features: {e}")
        return

    if mean_pitch is None:
        st.error("Could not detect a clear voice pitch. Please upload a clearer voice recording (mono, quiet background).")
        return

    # Gender filter by pitch
    try:
        if mean_pitch > MALE_PITCH_MAX_HZ:
            st.error("Voice pitch indicates a female or higher-pitched voice; please upload a male voice per task spec.")
            return
    except Exception:
        st.error("Internal error while validating voice pitch.")
        return

    # Estimate age and emotion
    try:
        age = _estimate_age(mean_pitch, mfcc)
    except Exception:
        st.error("Failed to estimate age from audio.")
        return

    st.success(f"Estimated age: **{age}**")

    try:
        if age > 60:
            st.info("Marked as **Senior Citizen**.")
            emotion = _estimate_emotion(mfcc)
            st.write(f"Detected emotion: **{emotion}**")
        else:
            st.caption("Age is 60 or below — only age is reported per task spec.")
    except Exception:
        st.warning("Age/emotion post-processing encountered an issue.")
        return
