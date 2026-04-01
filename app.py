import cv2
import numpy as np
import streamlit as st
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

    # read image
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    output = image.copy()

    # YOLO detection
    results = model.predict(
        source=image,
        conf=0.25,
        verbose=False
    )

    total = 0

    if results and results[0].boxes is not None:

        for box in results[0].boxes:

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            conf = float(box.conf[0])

            total += 1

            # draw box
            cv2.rectangle(output, (x1, y1), (x2, y2), (0,255,0), 2)

            cv2.putText(
                output,
                f"Tomato {conf:.2f}",
                (x1, y1-5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,0),
                2
            )

    # -------------------------
    # DISPLAY
    # -------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Image")
        st.image(uploaded, use_container_width=True)

    with col2:
        st.subheader("Detection Result")
        st.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB), use_container_width=True)

    st.success(f"Detected Tomatoes: {total}")
