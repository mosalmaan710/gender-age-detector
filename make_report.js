const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
} = require("docx");

const h1 = (t) => new Paragraph({ text: t, heading: HeadingLevel.HEADING_1, spacing: { before: 300, after: 150 } });
const h2 = (t) => new Paragraph({ text: t, heading: HeadingLevel.HEADING_2, spacing: { before: 200, after: 100 } });
const p = (t) => new Paragraph({ children: [new TextRun(t)], spacing: { after: 120 } });
const bullet = (t) => new Paragraph({ text: t, bullet: { level: 0 }, spacing: { after: 60 } });

const doc = new Document({
  sections: [
    {
      properties: { page: { size: { width: 12240, height: 15840 } } },
      children: [
        new Paragraph({
          text: "Real-Time Application for Gender and Age Detection",
          heading: HeadingLevel.TITLE,
          alignment: AlignmentType.CENTER,
          spacing: { after: 100 },
        }),
        new Paragraph({
          text: "Internship Final Project Report",
          alignment: AlignmentType.CENTER,
          spacing: { after: 300 },
        }),
        new Paragraph({ children: [new TextRun({ text: "Name: Muhammad Salmaan", bold: true })], alignment: AlignmentType.CENTER }),
        new Paragraph({ text: "Registration No.: RA2511003010979", alignment: AlignmentType.CENTER }),
        new Paragraph({ text: "Programme: B.Tech Computer Science and Engineering, SRMIST", alignment: AlignmentType.CENTER, spacing: { after: 300 } }),

        h1("1. Introduction"),
        p("This report documents the final internship project — a real-time gender and age detection application — together with the six supporting tasks assigned during the internship. All modules are implemented as pages within a single Streamlit application so the entire body of work can be reviewed, run, and demonstrated from one repository."),

        h1("2. Main Project: Real-Time Gender & Age Detector"),
        p("The core application accepts an image (file upload or webcam snapshot), detects all faces present using an OpenCV Haar-cascade face detector, and predicts each face's age and gender using DeepFace's pretrained age-net and gender-net models. Bounding boxes and predicted labels are drawn directly on the output frame."),

        h1("3. Supporting Tasks"),

        h2("Task 1 — Senior Citizen Identification"),
        p("Extends the core detector to multi-person frames typical of a mall or store camera. Any detected face with a predicted age above 60 is flagged as a Senior Citizen alongside its gender. Every detection (timestamp, age, gender, senior-citizen flag) is appended to a CSV log that doubles as the visit record requested in the task brief."),

        h2("Task 2 — Age & Emotion Detection through Voice"),
        p("Accepts a voice-note upload. Fundamental frequency (pitch) is used to gate the input: voices above a male-pitch threshold are rejected with the message \"Upload male voice.\" For accepted (male) voices, age is estimated from pitch and MFCC statistics; if the estimated age exceeds 60 the system also reports an emotion estimate derived from MFCC energy, otherwise only age is shown, matching the task specification."),

        h2("Task 3 — Long Hair Identification"),
        p("Implements the task's intentionally counter-intuitive rule: for faces predicted to be between 20 and 30 years old, hair length (estimated from the proportion of hair-coloured pixels below the face) overrides the true gender prediction — long hair is labelled Female and short hair is labelled Male regardless of the model's actual prediction. Outside the 20-30 age band, the unmodified gender prediction is shown."),

        h2("Task 4 — Nationality Detection Model"),
        p("Uses DeepFace's race/ethnicity output as a proxy signal for nationality (clearly documented as an approximation, since visual ethnicity is not equivalent to citizenship). Depending on the inferred nationality, different fields are surfaced: Indian subjects show nationality, age, dress colour and emotion; U.S. subjects show age and emotion only; African subjects show emotion and dress colour only; all other nationalities show nationality and emotion. Dress colour is estimated by averaging pixel colour in the region below the detected face."),

        h2("Task 5 — Car Colour Detection Model"),
        p("Detects cars in a traffic-signal image with a Haar-cascade car detector and classifies each as blue or non-blue via an HSV colour mask. Per the spec, blue cars are boxed in red and all other colours in blue. A HOG pedestrian detector separately counts people present at the signal."),

        h2("Task 6 — Sign Language Detection"),
        p("Restricted to operate only between 6 PM and 10 PM, as specified. Accepts an uploaded image or webcam snapshot, extracts hand landmarks with MediaPipe Hands, computes a five-finger up/down state vector, and matches it against a small dictionary of known gestures (e.g. open palm, fist, peace sign) to display the recognized word."),

        h1("4. Technology Stack"),
        bullet("Frontend / GUI: Streamlit"),
        bullet("Face & object detection: OpenCV (Haar cascades, HOG)"),
        bullet("Age / gender / emotion / race models: DeepFace (pretrained VGG-Face-based networks)"),
        bullet("Voice processing: librosa (pitch tracking, MFCC)"),
        bullet("Hand landmark tracking: MediaPipe"),
        bullet("Data logging: pandas / CSV"),

        h1("5. Limitations & Future Work"),
        p("The voice age/emotion estimator (Task 2) and the sign-language recogniser (Task 6) use lightweight, explainable heuristics rather than models trained end-to-end on large labelled datasets, since assembling and training on such datasets was outside the time and data scope of this internship task. With access to labelled speech-age corpora (e.g. Common Voice with age metadata) and a sign-language image/video dataset (e.g. ASL Alphabet or WLASL), both modules could be upgraded to trained deep-learning classifiers for materially higher accuracy. The nationality module also relies on visual ethnicity classification as a proxy and would benefit from being reframed around self-reported or document-based nationality in a production setting."),

        h1("6. Conclusion"),
        p("All seven modules — the main gender/age detector and the six supporting tasks — are implemented, integrated into a single navigable application, and ready for deployment. The repository, live demo link, and this report together fulfil the internship's final submission requirements."),
      ],
    },
  ],
});

Packer.toBuffer(doc).then((buf) => require("fs").writeFileSync("report.docx", buf));
