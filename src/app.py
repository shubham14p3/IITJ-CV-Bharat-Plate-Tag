    # Complete polished Bharat Number Plate Detector App with 5 model options

    import streamlit as st
    import cv2
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from io import BytesIO
    import base64
    import torch
    from ultralytics import YOLO
    from tensorflow.keras.models import load_model
    from settings import settings
    from src.SQLManager import DatabaseManager
    from src.sort import Sort

    # Setup
    st.set_page_config(page_title="Bharat Number Plate Detector", layout="wide")
    st.markdown("<h1 style='text-align: center;'>🇮🇳 Bharat Number Plate Detection System</h1>", unsafe_allow_html=True)

    # CSS for UI
    st.markdown("""
    <style>
        .block-container {padding-top: 1.5rem;}
        .stButton>button {background-color: #0078D4; color: white; font-weight: bold;}
        .stRadio > div {flex-direction: row;}
    </style>
    """, unsafe_allow_html=True)

    # DB connection
    db_manager = DatabaseManager("data", "database.db")
    db_manager.create_recognized_plates_table()

    # Load models
    yolo_model_path = Path("models/indian_plate_detection.pt")
    cnn_model_path = Path("models/cnn_plate_classifier.h5")
    try:
        yolo_model = YOLO(yolo_model_path)
        cnn_model = load_model(cnn_model_path)
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        st.stop()

    # Sidebar Config
    st.sidebar.header("Detection Controls")
    input_type = st.sidebar.radio("Upload Type", ["Image", "Video"])
    method = st.sidebar.selectbox("Choose Detection Approach", [
        "YOLOv8 (Deep Learning)",
        "Traditional CV (Canny + Contours)",
        "Color Segmentation",
        "Morphology + Edge Filter",
        "CNN Classifier (Custom DL)"
    ])
    conf_threshold = st.sidebar.slider("YOLO Confidence", 25, 100, 45) / 100
    show_data = st.sidebar.checkbox("Show Plate Log")

    # Detection Functions
    def detect_yolo(img):
        results = yolo_model(img, stream=True, verbose=False)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                if conf >= conf_threshold:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(img, f"{conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        return img

    def detect_traditional(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.bilateralFilter(gray, 11, 17, 17)
        edges = cv2.Canny(blur, 30, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
            approx = cv2.approxPolyDP(cnt, 0.018 * cv2.arcLength(cnt, True), True)
            if len(approx) == 4:
                cv2.drawContours(img, [approx], -1, (255, 0, 0), 2)
                break
        return img

    def detect_color(img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower, upper = np.array([0, 0, 200]), np.array([180, 50, 255])
        mask = cv2.inRange(hsv, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > 500:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,255), 2)
        return img

    def detect_morph(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(img, (x, y), (x+w, y+h), (255,255,0), 2)
                break
        return img

    def detect_cnn(img):
        resized = cv2.resize(img, (64, 64)) / 255.0
        pred = cnn_model.predict(np.expand_dims(resized, axis=0))[0][0]
        label = "Plate" if pred > 0.5 else "No Plate"
        color = (0, 255, 0) if pred > 0.5 else (0, 0, 255)
        cv2.putText(img, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        return img

    def run_detection(image):
        if method == "YOLOv8 (Deep Learning)": return detect_yolo(image)
        elif method == "Traditional CV (Canny + Contours)": return detect_traditional(image)
        elif method == "Color Segmentation": return detect_color(image)
        elif method == "Morphology + Edge Filter": return detect_morph(image)
        elif method == "CNN Classifier (Custom DL)": return detect_cnn(image)
        return image

    # Image Section
    if input_type == "Image":
        img_upload = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
        if img_upload:
            bytes_img = np.frombuffer(img_upload.read(), np.uint8)
            img = cv2.imdecode(bytes_img, cv2.IMREAD_COLOR)
            result = run_detection(img.copy())
            st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), use_column_width=True, caption="Result")

    # Video Section
    elif input_type == "Video":
        vid_upload = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
        if vid_upload:
            import tempfile
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(vid_upload.read())
            cap = cv2.VideoCapture(tfile.name)
            stframe = st.empty()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                output = run_detection(frame.copy())
                stframe.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB), use_column_width=True, channels="RGB")
            cap.release()

    # DB Table
    if show_data:
        st.subheader("🔍 Recognized Plate Log")
        df = db_manager.get_all_recognized_plates().drop("id", axis=1)
        st.dataframe(df)
        def download_link(df, name, text):
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                df.to_excel(writer, index=False)
            b64 = base64.b64encode(buffer.getvalue()).decode()
            return f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{name}">{text}</a>'
        st.markdown(download_link(df, "plates_log.xlsx", "Download Log"), unsafe_allow_html=True)
