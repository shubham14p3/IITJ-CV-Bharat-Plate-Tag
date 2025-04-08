import streamlit as st
import base64

def render_header(logo_path="assets/logo.png", title="Bharat Plate Detection"):
    with open(logo_path, "rb") as img_file:
        logo_encoded = base64.b64encode(img_file.read()).decode()

    st.markdown(f"""
    <style>
    .top-nav-wrapper {{
        display: flex;
        justify-content: center;
        margin-top: 40px;
        margin-bottom: 20px;
    }}

    .top-nav {{
        display: flex;
        align-items: center;
        justify-content: center;
        background-color: #0078D4;
        padding: 1rem 2rem;
        color: white;
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.15);
        max-width: 500px;
        width: 90%;
    }}

    .top-nav .title {{
        font-size: 22px;
        font-weight: bold;
        display: flex;
        align-items: center;
        justify-content: center;
    }}

    .top-nav img {{
        height: 40px;
        margin-right: 15px;
    }}
    </style>

    <div class="top-nav-wrapper">
        <div class="top-nav">
            <div class="title">
                <img src="data:image/png;base64,{logo_encoded}" />
                {title}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
