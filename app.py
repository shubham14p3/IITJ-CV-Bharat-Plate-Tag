import streamlit as st
import cv2
import pandas as pd
from pathlib import Path
import numpy as np
import math
import base64
from io import BytesIO
from ultralytics import YOLO
from src.sort import *
from src.SQLManager import DatabaseManager
import torch
import tempfile
from tensorflow.keras.models import load_model

# Set page config
st.set_page_config(
    page_title="Bharat Number Plate Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .block-container {
        padding-top: 2rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("\U0001F4F7 Bharat Number Plate Detection System")
st.markdown("""
**Detect number plates of Indian vehicles using multiple approaches:**
- **YOLOv8-based Deep Learning**
- **Traditional Computer Vision (Canny + Contours)**
- **Color-based Segmentation**
- **Edge + Morphological Filtering**
- **CNN-based Custom Classifier**
""")

# Database init
db_manager = DatabaseManager('data', 'database.db')
db_manager.create_recognized_plates_table()

# Load models
model_path = Path("models/indian_plate_detection.pt")
cnn_model_path = Path("models/cnn_plate_classifier.h5")

try:
    detection_model = YOLO(model_path)
    cnn_classifier = load_model(cnn_model_path)
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# Sidebar controls
st.sidebar.header("Upload Settings")
input_type = st.sidebar.radio("Choose Input Type", ["Image", "Video"])
method = st.sidebar.selectbox("Detection Method", [
    "YOLOv8 (Deep Learning)",
    "Traditional CV (Canny + Contours)",
    "Color Segmentation",
    "Edge + Morph Filter",
    "CNN Classifier (Custom DL)"
])
conf_threshold = st.sidebar.slider("Detection Confidence", 25, 100, 45) / 100

# Detection logic
PlatesId = list()

def detect_yolo(image):
    results = detection_model(image, stream=True, verbose=False)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = float(box.conf[0])
            if conf >= conf_threshold:
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f"Conf: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

def detect_traditional(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(blur, 30, 200)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.018 * peri, True)
        if len(approx) == 4:
            cv2.drawContours(image, [approx], -1, (255, 0, 0), 3)
            break
    return image

def detect_color_segmentation(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 50, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
    return image

def detect_edge_morph(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 0), 2)
            break
    return image

def cnn_classify_plate(image):
    try:
        resized = cv2.resize(image, (64, 64))
        normalized = resized / 255.0
        input_tensor = np.expand_dims(normalized, axis=0)
        prediction = cnn_classifier.predict(input_tensor)[0][0]
        label = "Plate" if prediction > 0.5 else "No Plate"
        color = (0, 255, 0) if prediction > 0.5 else (0, 0, 255)
        cv2.putText(image, label, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        return image
    except Exception as e:
        st.error(f"Error in CNN prediction: {e}")
        return image

# Unified detection handler
def run_detection(img):
    if method == "YOLOv8 (Deep Learning)":
        return detect_yolo(img)
    elif method == "Traditional CV (Canny + Contours)":
        return detect_traditional(img)
    elif method == "Color Segmentation":
        return detect_color_segmentation(img)
    elif method == "Edge + Morph Filter":
        return detect_edge_morph(img)
    elif method == "CNN Classifier (Custom DL)":
        return cnn_classify_plate(img)
    else:
        return img

# Image Upload
if input_type == "Image":
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        output = run_detection(image.copy())
        st.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB), caption="Detection Result", use_column_width=True)

# Video Upload
elif input_type == "Video":
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            output = run_detection(frame.copy())
            stframe.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
        cap.release()

# Optional DB Table Display
if st.sidebar.checkbox("Show Recognized Data"):
    st.subheader("Recognized Plates Log")
    dataframe = db_manager.get_all_recognized_plates()
    dataframe = dataframe.drop('id', axis=1)
    st.dataframe(dataframe, use_container_width=True)

    def create_download_link(df, filename, link_text):
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            df.to_excel(writer, index=False)
        b64 = base64.b64encode(excel_buffer.getvalue()).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">{link_text}</a>'
        return href

    download_link = create_download_link(dataframe, 'plates_log.xlsx', 'Download Excel Log')
    st.markdown(download_link, unsafe_allow_html=True)
