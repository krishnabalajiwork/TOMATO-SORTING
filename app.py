import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
from ultralytics import YOLO
import torch
from tomato_pipeline import load_classifier, make_transform, classify_crop


# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Tomato AI Detector",
    page_icon="🍅",
    layout="wide"
)

st.title("🍅 Tomato AI Detector")
st.caption("YOLO Detection + EfficientNet Classification")


# -------------------------
# LOAD MODELS
# -------------------------
@st.cache_resource
def load_models():
    device = torch.device("cpu")
    detector = YOLO("best.pt")
    classifier = load_classifier("efficientnet_b0_best.pth", device)
    transform = make_transform(224)
    return detector, classifier, transform, device

detector, classifier, transform, device = load_models()

LABELS = ["bad", "good"]

COLORS = {
    "good": (0, 200, 0),
    "bad":  (220, 0, 0),
}


# -------------------------
# INPUT
# -------------------------
uploaded = st.file_uploader(
    "Upload Tomato Image",
    type=["jpg", "png", "jpeg"]
)

run = st.button("Run Detection")


# -------------------------
# INFERENCE
# -------------------------
if run:

    if uploaded is None:
        st.error("Please upload an image first")
        st.stop()

    image = Image.open(uploaded).convert("RGB")
    output = image.copy()
    draw = ImageDraw.Draw(output)

    img_array = np.array(image)

    # Step 1: YOLO detects tomato bounding boxes
    results = detector.predict(
        source=img_array,
        conf=0.25,
        verbose=False
    )

    total = 0
    good_count = 0
    bad_count = 0

    if results and results[0].boxes is not None:

        for box in results[0].boxes:

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Step 2: Crop each detected tomato
            crop = image.crop((x1, y1, x2, y2))

            # Step 3: EfficientNet classifies good vs bad
            label, conf = classify_crop(crop, classifier, transform, device, LABELS)

            total += 1
            if label == "good":
                good_count += 1
            else:
                bad_count += 1

            color = COLORS[label]

            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            # Draw label
            draw.text(
                (x1, max(0, y1 - 18)),
                f"{label.upper()} {conf:.2f}",
                fill=color
            )

    # -------------------------
    # DISPLAY
    # -------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Detection Result")
        st.image(output, use_container_width=True)

    st.success(f"✅ Good Tomatoes: {good_count}")
    st.error(f"❌ Bad Tomatoes: {bad_count}")
    st.info(f"🍅 Total Detected: {total}")
