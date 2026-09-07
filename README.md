# Real-Time Application for Gender and Age Detection

Internship final project + 6 supporting tasks, built as a single multi-page
Streamlit app.

**Author:** Muhammad Salmaan (RA2511003010979)

## Modules
| Page | What it does |
|---|---|
| Main Project | Real-time gender & age detection from image/webcam |
| Task 1 | Senior citizen identification + CSV visit log |
| Task 2 | Age & emotion detection from voice (male-only) |
| Task 3 | Long hair identification (age 20-30 override logic) |
| Task 4 | Nationality detection with conditional output fields |
| Task 5 | Car colour detection + people counting at a signal |
| Task 6 | Sign language detection (operates 6 PM - 10 PM only) |

## Run locally
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy for a live URL (free)
1. Push this folder to a **GitHub** repo (see commands below).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
3. Click **New app** → select the repo → set main file to `app.py` → **Deploy**.
4. Streamlit Cloud gives you a public `https://<name>.streamlit.app` URL — that's
   your "live URL" for the submission form.

## Push to your own GitHub
```bash
cd gender-age-detector
git init
git add .
git commit -m "Gender & Age Detector - main project + 6 tasks"
git branch -M main
git remote add origin https://github.com/<your-username>/gender-age-detector.git
git push -u origin main
```

## Google Drive (datasets)
Create a Drive folder, upload:
- Any sample images/audio you tested with (`data/samples/`)
- The generated `data/senior_citizen_log.csv`
Set sharing to "Anyone with the link" and paste that link into the submission form.

## Notes / limitations
- Age, gender, emotion, and race predictions use **DeepFace**'s pretrained
  networks (VGG-Face / age-net / gender-net backbones) — standard practice for
  this class of project rather than training from scratch, which needs far
  more data/compute than a short internship task allows.
- Voice age/emotion (Task 2) and sign recognition (Task 6) use lightweight,
  explainable signal-processing heuristics (pitch/MFCC, hand-landmark finger
  states) rather than trained deep models, for the same reason — documented
  as a limitation in the report and a good "future work" callout.
