import json
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch

from ultralytics import YOLO
from tomato_pipeline import load_classifier, make_transform, classify_crop


# Debug (remove later)
st.write("Python:", sys.version)


st.set_page_config(
    page_title="Tomato AI Inspector",
    page_icon="🍅",
    layout="wide"
)


st.title("🍅 Tomato AI Quality Inspector")
st.caption("YOLO Detection + EfficientNet Classification")


DETECTOR_PATH = Path("best.pt")
CLASSIFIER_PATH = Path("efficientnet_b0_best.pth")


@st.cache_resource
def load_models(device):
    detector = YOLO(str(DETECTOR_PATH))
    classifier = load_classifier(CLASSIFIER_PATH, device)
    return detector, classifier


device = "cuda" if torch.cuda.is_available() else "cpu"


uploaded = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

run = st.button("Run Detection")


if run:

    if uploaded is None:
        st.error("Upload image first")
        st.stop()

    detector, classifier = load_models(device)

    image_bytes = uploaded.getvalue()

    image_np = cv2.imdecode(
        np.frombuffer(image_bytes, np.uint8),
        cv2.IMREAD_COLOR
    )

    h, w = image_np.shape[:2]

    output = image_np.copy()

    transform = make_transform(224)

    detections = detector.predict(
        source=image_np,
        conf=0.25,
        device=device,
        verbose=False
    )

    results = []
    good = 0
    bad = 0

    if detections and detections[0].boxes is not None:

        for box in detections[0].boxes:

            x1,y1,x2,y2 = box.xyxy[0].tolist()

            x1,y1,x2,y2 = map(int,[x1,y1,x2,y2])

            crop = image_np[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            label, conf = classify_crop(
                crop, classifier, transform, device, ["bad","good"]
            )

            if label.lower() == "good":
                good += 1
                color = (0,255,0)
            else:
                bad += 1
                color = (0,0,255)

            cv2.rectangle(output,(x1,y1),(x2,y2),color,2)
            cv2.putText(output,f"{label} {conf:.2f}",(x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

            results.append(label)

    st.write(f"Total: {len(results)} | Good: {good} | Bad: {bad}")

    st.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
