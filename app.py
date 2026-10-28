"""
Project Synthexis - Desktop UI Integration
Serves the Google Stitch desktop workspace with real-time FastAPI integration.
"""

import os
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Synthexis — Multimodal Synthesis Workspace",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Clean minimalist embedding styles (hiding Streamlit header/menus)
st.markdown("""
<style>
    #MainMenu {visibility: hidden !important; display: none !important;}
    footer {visibility: hidden !important; display: none !important;}
    header {visibility: hidden !important; display: none !important;}
    [data-testid="stToolbar"] {visibility: hidden !important; display: none !important;}
    [data-testid="stDecoration"] {visibility: hidden !important; display: none !important;}
    [data-testid="stStatusWidget"] {visibility: hidden !important; display: none !important;}
    .block-container {
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
        padding-left: 0rem !important;
        padding-right: 0rem !important;
        max-width: 100% !important;
    }
    iframe {
        border: none !important;
        width: 100% !important;
        height: 100vh !important;
        min-height: 960px !important;
    }
</style>
""", unsafe_allow_html=True)

APP_DIR = Path(__file__).resolve().parent
target_file = APP_DIR / "index.html"

if target_file.exists():
    with open(target_file, "r", encoding="utf-8") as f:
        html_code = f.read()
    components.html(html_code, height=980, scrolling=True)
else:
    st.error("Dashboard template index.html not found.")
