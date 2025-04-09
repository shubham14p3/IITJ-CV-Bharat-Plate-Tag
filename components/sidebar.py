# components/sidebar.py
import streamlit as st

def render_sidebar():
    st.sidebar.header("⚙️ Detection Settings")
    input_type = st.sidebar.radio("Choose Input Type", ["Image", "Video"])
    method = st.sidebar.selectbox(
        "Detection Method",
        [
            "YOLOv8 (Car)",
            "Traditional CV (Canny + Contours)",
            "Color Segmentation",
            "Edge + Morph Filter (Bike)",
            "CNN Classifier (Bike/Car)",
            "OCR Plate Recognition (Bike/Car)",
        ],
    )
    conf_threshold = st.sidebar.slider("Confidence Threshold", 25, 100, 45) / 100
    show_db = st.sidebar.checkbox("Show Plate Log")
    if st.sidebar.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.rerun()
    
    return input_type, method, conf_threshold, show_db
