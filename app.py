"""
Real-Time Application for Gender and Age Detection
Main Project + Tasks
Author: Muhammad Salmaan

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""

import streamlit as st

st.set_page_config(page_title="Gender & Age Detector", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose a module",
    [
        "Main Project - Real-Time Gender & Age Detector",
        "Task 2 - Age & Emotion Detection (Voice)",
        "Task 3 - Long Hair Identification",
        "Task 4 - Nationality Detection",
        "Task 5 - Car Colour Detection",
        "Task 6 - Sign Language Detection",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption("Internship submission build — SRMIST / ElevanceSkills")

if page == "Main Project - Real-Time Gender & Age Detector":
    from tasks import main_project

    main_project.run()
elif page == "Task 2 - Age & Emotion Detection (Voice)":
    from tasks import voice_age_emotion

    voice_age_emotion.run()
elif page == "Task 3 - Long Hair Identification":
    from tasks import long_hair

    long_hair.run()
elif page == "Task 4 - Nationality Detection":
    from tasks import nationality

    nationality.run()
elif page == "Task 5 - Car Colour Detection":
    from tasks import car_colour

    car_colour.run()
elif page == "Task 6 - Sign Language Detection":
    from tasks import sign_language

    sign_language.run()
