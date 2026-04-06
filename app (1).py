import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Tomato AI Detector",
    page_icon="🍅",
    layout="wide"
)

st.title("🍅 Tomato AI Detector (YOLO)")
st.caption("Fast & Stable Deployment Version")


# -------------------------
# LOAD MODEL
# -------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()


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

    # read image with PIL (no cv2 needed)
    image = Image.open(uploaded).convert("RGB")
    output = image.copy()
    draw = ImageDraw.Draw(output)

    # YOLO detection (pass numpy array)
    img_array = np.array(image)
    results = model.predict(
        source=img_array,
        conf=0.25,
        verbose=False
    )

    total = 0

    if results and results[0].boxes is not None:

        for box in results[0].boxes:

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            conf = float(box.conf[0])
            total += 1

            # draw box
            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)

            # draw label
            draw.text(
                (x1, max(0, y1 - 18)),
                f"Tomato {conf:.2f}",
                fill=(0, 255, 0)
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

    st.success(f"Detected Tomatoes: {total}")
