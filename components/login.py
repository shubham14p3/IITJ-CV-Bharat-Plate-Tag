# components/login.py
import streamlit as st
import base64
from pathlib import Path

def show_centered_logo(image_path="assets/logo.png", width=160):
    if Path(image_path).exists():
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode()
        st.markdown(
            f"""
            <div style='text-align: center; margin-top: 40px;'>
                <img src='data:image/png;base64,{encoded}' width='{width}'/>
            </div>
            """, unsafe_allow_html=True)

def render_login():
    show_centered_logo("assets/logo.png", width=160)

    st.markdown("""
        <style>
        /* Container centers the login form */
        .login-container {
            display: flex;
            justify-content: center;
            margin-top: 30px;
        }
        /* Login box styling: remove the background color and box shadow */
        .login-box {
            /* background-color: #f5f7fa; */  /* Removed for transparent background */
            padding: 2rem;
            width: 50%;
            max-width: 450px;
            border-radius: 10px;
            /* box-shadow: 0 4px 10px rgba(0,0,0,0.1); */  /* Removed */
            text-align: center;
        }
        /* Force the input fields to not take full width if desired */
        .login-box .stTextInput>div>div>input {
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h3 style='text-align: center;'>Login to Bharat Plate Detector</h3>", unsafe_allow_html=True)
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "1234":
            st.session_state.logged_in = True
            st.success("✅ Login successful!")
            st.rerun()
        else:
            st.error("❌ Invalid credentials. Try again.")

    st.markdown("</div></div>", unsafe_allow_html=True)
