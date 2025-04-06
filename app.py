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
from tensorflow.keras.models import load_model
from src.SQLManager import DatabaseManager
from src.sort import *
import pytesseract
from src.PlateGen import PlateGen

# ==== Page Config ====
st.set_page_config(page_title="Bharat Number Plate Detector", layout="wide")

# ==== Session Setup ====
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# ==== Centered Logo Helper ====
def show_centered_logo(image_path, width=160):
    if Path(image_path).exists():
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode()
        st.markdown(
            f"""
            <div style='text-align: center;'>
                <img src='data:image/png;base64,{encoded}' width='{width}'/>
            </div>
            """,
            unsafe_allow_html=True
        )

# ==== Login Interface ====
if not st.session_state.logged_in:
    show_centered_logo("image/plate_template.png", width=160)
    st.markdown("<h2 style='text-align: center;'>Login to Bharat Plate Detector</h2>", unsafe_allow_html=True)

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "1234":
            st.session_state.logged_in = True
            st.success("✅ Login successful!")
            st.rerun()
        else:
            st.error("❌ Invalid credentials. Try again.")

    st.stop()

# ==== Top Navigation Bar ====
logo_encoded = base64.b64encode(open("image/plate_template.png", "rb").read()).decode()
st.markdown(f"""
    <style>
    .top-nav {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        background-color: #0078D4;
        padding: 0.8rem 2rem;
        color: white;
        border-radius: 0 0 6px 6px;
    }}
    .top-nav .title {{
        font-size: 20px;
        font-weight: bold;
        display: flex;
        align-items: center;
    }}
    .top-nav img {{
        height: 40px;
        margin-right: 15px;
    }}
    .top-nav .actions button {{
        margin-left: 12px;
        background-color: #004A99;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 6px 12px;
        cursor: pointer;
    }}
    </style>

    <div class="top-nav">
        <div class="title">
            <img src="data:image/png;base64,{logo_encoded}" />
            Bharat Plate Detection
        </div>
        <div class="actions">
            <button onclick="window.location.reload(); return false;">🔄 Refresh</button>
            <button onclick="window.location.href=''; return false;">❌ Exit</button>
        </div>
    </div>
""", unsafe_allow_html=True)

# ==== CSS Styling ====
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        color: black;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        margin-top: 5rem;
        # background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 0 10px rgba(0,0,0,0.05);
    }
    .stTextInput>div>div>input {
        color: black;
    }
    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        background-color: #f0f2f6;
        color: #666;
        text-align: center;
        padding: 10px;
        font-size: 14px;
    }
    </style>
""", unsafe_allow_html=True)

# ==== Database Setup ====
db_manager = DatabaseManager('data', 'database.db')
db_manager.create_recognized_plates_table()

# ==== Model Loading ====
yolo_model_path = Path("models/indian_plate_detection.pt")
cnn_model_path = Path("models/cnn_plate_classifier.h5")

try:
    yolo_model = YOLO(yolo_model_path)
    cnn_model = load_model(cnn_model_path)
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# ==== Sidebar ====
st.sidebar.header("⚙️ Detection Settings")
input_type = st.sidebar.radio("Choose Input Type", ["Image", "Video"])
method = st.sidebar.selectbox("Detection Method", [
    "YOLOv8 (Deep Learning)",
    "Traditional CV (Canny + Contours)",
    "Color Segmentation",
    "Edge + Morph Filter",
    "CNN Classifier (Custom DL)",
    "OCR Plate Recognition"
])
conf_threshold = st.sidebar.slider("Confidence Threshold", 25, 100, 45) / 100
show_db = st.sidebar.checkbox("Show Plate Log")
if st.sidebar.button("🚪 Logout"):
    st.session_state.logged_in = False
    st.rerun()

# ==== Detection Functions ====
def detect_yolo(image):
    results = yolo_model(image, stream=True, verbose=False)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            if conf >= conf_threshold:
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f"{conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
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
    resized = cv2.resize(image, (64, 64)) / 255.0
    pred = cnn_model.predict(np.expand_dims(resized, axis=0))[0][0]
    label = "Plate" if pred > 0.5 else "No Plate"
    color = (0, 255, 0) if pred > 0.5 else (0, 0, 255)
    cv2.putText(image, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return image
def detect_ocr_plate(image):
    # Use YOLO to detect the plate region(s)
    results = yolo_model(image, stream=True, verbose=False)
    recognized_texts = []  # To store recognized plate texts

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            if conf >= conf_threshold:
                # Draw bounding box for visual feedback
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f"{conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Crop the plate region
                plate_roi = image[y1:y2, x1:x2]
                
                # Use Tesseract to extract text from the cropped region
                recognized_text = pytesseract.image_to_string(plate_roi, config="--psm 7").strip()
                # Clean the text (optional): remove non-alphanumeric characters
                recognized_text = "".join(char for char in recognized_text if char.isalnum())
                
                if recognized_text:
                    recognized_texts.append(recognized_text)
                    # Generate the stylized plate image using PlateGen
                    stylized_plate = PlateGen(recognized_text)
                    # Display the generated plate image
                    st.image(
                        cv2.cvtColor(stylized_plate, cv2.COLOR_BGR2RGB),
                        caption=f"Recognized Plate: {recognized_text}",
                        use_container_width=True
                    )
    return image, recognized_texts

def run_detection(image):
    if method == "YOLOv8 (Deep Learning)": return detect_yolo(image)
    elif method == "Traditional CV (Canny + Contours)": return detect_traditional(image)
    elif method == "Color Segmentation": return detect_color(image)
    elif method == "Edge + Morph Filter": return detect_morph(image)
    elif method == "CNN Classifier (Custom DL)": return detect_cnn(image)
    elif method == "OCR Plate Recognition": return detect_ocr_plate(image)
    return image

# ==== Image Upload ====
if input_type == "Image":
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        img_data = np.frombuffer(uploaded_image.read(), np.uint8)
        image = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        
        result = run_detection(image.copy())
        
        if method == "OCR Plate Recognition":
            # Unpack the tuple returned by detect_ocr_plate
            annotated_image, recognized_texts = result
            st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB),
                     caption="Detection Result",
                     use_container_width=True)
            # Optionally, you can display the recognized text(s)
            st.write("Recognized Plate(s):", recognized_texts)
        else:
            st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
                     caption="Detection Result",
                     use_container_width=True)

# ==== Video Upload ====
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
            # If result is a tuple (OCR method), unpack it
            if isinstance(result, tuple):
                annotated_frame, _ = result  # Ignore recognized texts for video display
            else:
                annotated_frame = result
            
            # Now, annotated_frame should be an image array
            stframe.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
                          use_container_width=True, channels="RGB")
        cap.release()


# ==== Database Viewer ====
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

# ==== Footer ====
st.markdown("<div class='footer'>© 2025 BharatAI Labs | Built with ❤️ using Streamlit</div>", unsafe_allow_html=True)
