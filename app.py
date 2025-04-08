import sys
# On Windows, set the event loop policy to avoid "no running event loop" errors
if sys.platform.startswith("win"):
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from io import BytesIO
import base64
import os
import tempfile
from ultralytics import YOLO
import pytesseract
from tensorflow.keras.models import load_model

# Import your modules from src
from src.SQLManager import DatabaseManager
from src.sort import *
from src.PlateGen import PlateGen
from src.cnn_plate_pipeline import (
    detect_license_plate,
    segment_characters,
    predict_plate_number,
    load_or_train_cnn_model
)

# Import components
from components.header import render_header
from components.footer import render_footer
from components.sidebar import render_sidebar
from components.login import render_login
# ==== Page Config ====
st.set_page_config(page_title="Bharat Number Plate Detector", layout="wide")

# ==== Session Setup ====
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


# ==== Login Interface ====
if not st.session_state.logged_in:
    render_login()
    st.stop()

# ==== Top Navigation Bar ====
render_header()

# ==== Database Setup ====
db_manager = DatabaseManager("data", "database.db")
db_manager.create_recognized_plates_table()

# ==== Model Loading for YOLO (CNN model is loaded later within functions) ====
yolo_model_path = Path("models/indian_plate_detection.pt")
try:
    yolo_model = YOLO(yolo_model_path)
except Exception as e:
    st.error(f"YOLO model loading failed: {e}")
    st.stop()

# ==== Sidebar ====
input_type, method, conf_threshold, show_db = render_sidebar()

# --- Reset cached results if sidebar selections change ---
if "prev_method" not in st.session_state:
    st.session_state.prev_method = method
if "prev_input_type" not in st.session_state:
    st.session_state.prev_input_type = input_type

if st.session_state.prev_method != method or st.session_state.prev_input_type != input_type:
    if "detection_result" in st.session_state:
        del st.session_state["detection_result"]
    st.session_state.prev_method = method
    st.session_state.prev_input_type = input_type
    st.rerun()

# ---------- Helper to overlay text onto an image ----------
def overlay_plate_text(image, plate_text):
    """
    Overlays the text "Number Plate: <plate_text>" in the top-left corner of the provided image.
    The text is drawn in white on a black-filled rectangle.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    text = f"Number Plate: {plate_text}"
    (w, h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    # Define top-left corner (10, h+10) and draw a black rectangle behind the text
    x, y = 10, h + 10
    cv2.rectangle(image, (x - 5, y - h - 5), (x + w + 5, y + baseline + 5), (0, 0, 0), -1)
    cv2.putText(image, text, (x, y), font, font_scale, (255, 255, 255), thickness)
    return image

# ---------- Detection Functions ----------
def detect_yolo(image):
    results = yolo_model(image, stream=True, verbose=False)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            if conf >= conf_threshold:
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    image,
                    f"{conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
    return image

def detect_traditional(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(blur, 30, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
        approx = cv2.approxPolyDP(cnt, 0.018 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            cv2.drawContours(image, [approx], -1, (255, 0, 0), 2)
            break
    return image

def detect_color(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower, upper = np.array([0, 0, 200]), np.array([180, 50, 255])
    mask = cv2.inRange(hsv, lower, upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
    return image

def detect_morph(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 0), 2)
            break
    return image

def detect_cnn(image):
    cnn_model = load_or_train_cnn_model()
    plate_img, cropped_plate = detect_license_plate(image)
    st.image(cropped_plate, caption="🧪 Cropped Plate Region", channels="BGR", use_container_width=True)
    if cropped_plate is None:
        st.warning("⚠️ No license plate detected.")
        return image
    char_imgs = segment_characters(cropped_plate)
    st.write(f"🧩 Number of segmented characters: {len(char_imgs)}")
    if len(char_imgs) == 0:
        st.warning("⚠️ Could not segment characters from plate.")
        return image
    plate_text = predict_plate_number(cnn_model, char_imgs)
    final_img, _ = detect_license_plate(image, plate_text)
    stylized_plate = PlateGen(plate_text)
    st.markdown("<div style='margin-left: 20px;'>", unsafe_allow_html=True)
    st.image(cv2.cvtColor(stylized_plate, cv2.COLOR_BGR2RGB),
             caption=f"Predicted Plate: {plate_text}",
             use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    return final_img

def detect_ocr_plate(image):
    results = yolo_model(image, stream=True, verbose=False)
    recognized_texts = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            if conf >= conf_threshold:
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f"{conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                plate_roi = image[y1:y2, x1:x2]
                recognized_text = pytesseract.image_to_string(plate_roi, config="--psm 7").strip()
                recognized_text = "".join(char for char in recognized_text if char.isalnum())
                if recognized_text:
                    recognized_texts.append(recognized_text)
                    stylized_plate = PlateGen(recognized_text)
                    st.image(cv2.cvtColor(stylized_plate, cv2.COLOR_BGR2RGB),
                             caption=f"Recognized Plate: {recognized_text}",
                             use_container_width=True)
    return image, recognized_texts

def recognize_plate_text(image):
    """
    Common function to extract the plate from image and recognize text using the CNN pipeline.
    """
    _, cropped_plate = detect_license_plate(image)
    if cropped_plate is None:
        return None
    char_imgs = segment_characters(cropped_plate)
    if len(char_imgs) == 0:
        return None
    cnn_model = load_or_train_cnn_model()
    return predict_plate_number(cnn_model, char_imgs)

def run_detection(image):
    if method == "YOLOv8 (Deep Learning)":
        detected_img = detect_yolo(image)
    elif method == "Traditional CV (Canny + Contours)":
        detected_img = detect_traditional(image)
    elif method == "Color Segmentation":
        detected_img = detect_color(image)
    elif method == "Edge + Morph Filter":
        detected_img = detect_morph(image)
    elif method == "CNN Classifier (Custom DL)":
        return detect_cnn(image)
    elif method == "OCR Plate Recognition":
        return detect_ocr_plate(image)
    else:
        detected_img = image

    plate_text = recognize_plate_text(image.copy())
    if plate_text is not None:
        final_img, _ = detect_license_plate(image, plate_text)
        # Overlay the recognized plate text in the top-left corner with black background and white text
        final_img = overlay_plate_text(final_img, plate_text)
        return final_img
    else:
        return detected_img

# -------- Helper: Overlay Plate Text on an image --------
def overlay_plate_text(image, plate_text):
    """
    Overlays "Number Plate: <plate_text>" in the top left corner of the image with a black rectangle background.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    text = f"Number Plate: {plate_text}"
    (w, h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    # Define top-left point (for example, (10, h+10))
    x, y = 10, h + 10
    # Draw filled rectangle (black) as background
    cv2.rectangle(image, (x - 5, y - h - 5), (x + w + 5, y + baseline + 5), (0, 0, 0), -1)
    # Put white text over it
    cv2.putText(image, text, (x, y), font, font_scale, (255, 255, 255), thickness)
    return image

# ---------- Image Upload ----------
if input_type == "Image":
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        img_data = np.frombuffer(uploaded_image.read(), np.uint8)
        image = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        # Show the uploaded image first
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                 caption="Uploaded Image", use_container_width=True)
        result = run_detection(image.copy())
        if method == "OCR Plate Recognition":
            annotated_image, recognized_texts = result
            st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB),
                     caption="Detection Result",
                     use_container_width=True)
            st.write("Recognized Plate(s):", recognized_texts)
        else:
            st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
                     caption="Detection Result",
                     use_container_width=True)

# ---------- Video Upload ----------
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
            result = run_detection(frame.copy())
            if isinstance(result, tuple):
                annotated_frame, _ = result
            else:
                annotated_frame = result
            stframe.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
                          use_container_width=True, channels="RGB")
        cap.release()

# ---------- Database Viewer ----------
if show_db:
    st.subheader("📋 Plate Detection Log")
    df = db_manager.get_all_recognized_plates().drop("id", axis=1)
    st.dataframe(df)
    def download_excel(df, name="plates_log.xlsx"):
        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, index=False)
        b64 = base64.b64encode(output.getvalue()).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{name}">📥 Download Log</a>'
        return href
    st.markdown(download_excel(df), unsafe_allow_html=True)

# ---------- Render Footer ----------
render_footer()
