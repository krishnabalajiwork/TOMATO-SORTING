# 🍅 Tomato Sorting System — AI Quality Inspection & Automated Grading

[![Live Demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://tomato-sorting1.streamlit.app/)
[![Tech Stack](https://img.shields.io/badge/Stack-YOLOv8_%7C_EfficientNet_%7C_PyTorch-red?style=for-the-badge)](https://github.com/krishnabalajiwork/TOMATO-SORTING)

> **Developer & Technical Documentation**  
> An automated, end-to-end computer vision and mechatronics sorting pipeline that detects tomatoes, classifies quality grades (**Good** vs. **Bad**), and integrates with physical conveyor hardware via microcontrollers.

---

## 🏗️ System Architecture & Pipeline Flow

The system combines edge object detection with deep feature classification to evaluate quality parameters in real time:

```text
[ Input Image / Live Frame ]
             │
             ▼
[ YOLOv8 Object Detection ] ──> (Bounding Box Coordinates)
                                         │
                                         ▼
                             [ Region Crop & Transform ]
                                         │
                                         ▼
                       [ EfficientNet-B0 Classification ] ──> [ Grade: Good / Bad ]
                                                                        │
                                                                        ▼
                                                         [ Streamlit UI & Hardware Actuation ]

```

---

## 📐 Hardware & CAD Prototyping

| OpenSCAD 3D Mechanical Blueprint | Physical Prototype in Action |
| --- | --- |
|  |  |

---

## ⚡ Key Capabilities

* **Hybrid AI Architecture:** Combines YOLOv8 for precise spatial localization and EfficientNet-B0 for robust texture/quality classification.
* **Real-Time Quality Grading:** Automatically segregates and counts incoming produce into distinct `Good` and `Bad` categories.
* **Interactive Web Interface:** Streamlit-powered dashboard supporting image uploads and live camera feeds.
* **Mechatronic Integration Ready:** Built to interface with Raspberry Pi 4 camera modules and Arduino microcontrollers for physical conveyor belt sorting.

---

## 🔌 Technology Stack & Models

| Component | Technology / Framework | Purpose |
| --- | --- | --- |
| **Object Detection** | YOLOv8 (`best.pt`) | Detects and localizes individual tomatoes within frames |
| **Quality Classification** | EfficientNet-B0 (`efficientnet_b0_best.pth`) | Classifies cropped regions into quality grades |
| **Deep Learning Engine** | PyTorch + Torchvision | Tensor operations, model inference, and image transforms |
| **Frontend & UI** | Streamlit + Pillow | Reactive web interface and bounding-box rendering |
| **Hardware Layer (Optional)** | Raspberry Pi 4 + Arduino | Edge inference, stepper/servo motor control for physical sorting |

---

## 📂 Repository Structure

```text
TOMATO-SORTING/
 ├── app.py                     # Main Streamlit web application interface
 ├── 1app.py                    # Alternate interface deployment script
 ├── tomato_pipeline.py         # Core computer vision inference and pipeline functions
 ├── best.pt                    # Trained YOLOv8 detection model weights
 ├── efficientnet_b0_best.pth   # Trained EfficientNet-B0 classification weights
 ├── requirements.txt           # Python package dependencies
 └── README.md                  # Project documentation

```

---

## ⚙️ Environment Configuration

Create a `.env` file or export your runtime variables if required, or ensure your PyTorch environment supports CPU/GPU execution profiles.

---

## 🚀 Quickstart & Local Setup

### Prerequisites

* **Python:** v3.9 or higher
* **PyTorch:** Compatible CPU or CUDA build

### 1. Clone & Install Dependencies

```bash
git clone [https://github.com/krishnabalajiwork/TOMATO-SORTING.git](https://github.com/krishnabalajiwork/TOMATO-SORTING.git)
cd TOMATO-SORTING
pip install -r requirements.txt

```

### 2. Run the Application

```bash
streamlit run app.py

```

Open your browser at `http://localhost:8501`.

---

## 🐛 Troubleshooting & Known Fixes

#### 1. PyTorch Model Weight Loading Errors

* **Cause:** Missing model weight files (`best.pt` or `efficientnet_b0_best.pth`) in the execution root.
* **Solution:** Ensure model weight checkpoints are placed directly in the main directory before launching the app.

---

## 👨‍💻 Author & Contact

**Chintha Krishna Balaji**

* **GitHub:** [@krishnabalajiwork](https://www.google.com/search?q=https://github.com/krishnabalajiwork)
* **Live Demo:** [tomato-sorting1.streamlit.app](https://tomato-sorting1.streamlit.app/)

---

## 📝 License

This project is open-source and released under the [MIT License](https://www.google.com/search?q=LICENSE).

```

```
