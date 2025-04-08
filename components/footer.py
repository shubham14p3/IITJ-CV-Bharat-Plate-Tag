# components/footer.py
import streamlit as st
import base64

def render_footer(logo_path="assets/logo.png"):
    with open(logo_path, "rb") as image_file:
        logo_encoded = base64.b64encode(image_file.read()).decode()

    st.markdown(f"""
    <style>
    .footer {{
        position: fixed;
        bottom: 0;
        width: 100%;
        background-color: #f0f2f6;
        color: #666;
        padding: 10px 30px;
        font-size: 14px;
        border-top: 1px solid #ddd;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }}
    .footer-left {{
        width: 10%;
    }}
    .footer-left img {{
        max-height: 50px;
    }}
    .footer-right {{
        width: 90%;
        text-align: left;
    }}
    .team {{
        display: flex;
        justify-content: flex-start;
        flex-wrap: wrap;
        margin-top: 8px;
    }}
    .team-member {{
        margin: 0 20px;
        text-align: left;
        min-width: 140px;
    }}
    </style>
    <div class="footer">
        <div class="footer-left">
            <img src="data:image/png;base64,{logo_encoded}" />
        </div>
        <div class="footer-right">
            <div>© 2025 Bharat Plate Tag | Project of CV | IITJ </div>
            <div class="team">
                <div class="team-member">
                    <strong>m24de3076</strong><br>
                    Shubham Raj<br>
                    m24de3076@iitj.ac.in
                </div>
                <div class="team-member">
                    <strong>M24DE3022</strong><br>
                    Bhavesh Arora<br>
                    m24de3022@iitj.ac.in
                </div>
                <div class="team-member">
                    <strong>M24DE3043</strong><br>
                    Kanishka Dhindhwal<br>
                    m24de3043@iitj.ac.in
                </div>
                <div class="team-member">
                    <strong>M24DE3062</strong><br>
                    Pratyush Solanki<br>
                    m24de3062@iitj.ac.in
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
