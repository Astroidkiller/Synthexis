"""
Project Synthexis - Minimalist Multimodal Analysis Dashboard
Clean, high-contrast, distraction-free UI for document ingestion and cross-synthesis.
"""

import os
import json
import base64
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional

import streamlit as st

# ==============================================================================
# STREAMLIT CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Synthexis",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

APP_DIR = Path(__file__).resolve().parent
STYLE_FILE = APP_DIR / "style.css"
DEFAULT_API_URL = os.getenv("SYNTHEXIS_API_URL", "http://127.0.0.1:8000")

# ==============================================================================
# INJECT MINIMAL THEME
# ==============================================================================
def load_custom_styles():
    if STYLE_FILE.exists():
        with open(STYLE_FILE, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_custom_styles()

# ==============================================================================
# SESSION STATE
# ==============================================================================
if "api_url" not in st.session_state:
    st.session_state.api_url = DEFAULT_API_URL

if "latest_response" not in st.session_state:
    st.session_state.latest_response = None

if "query_input" not in st.session_state:
    st.session_state.query_input = ""

# ==============================================================================
# API CLIENT HELPERS
# ==============================================================================
def get_backend_health(url: str) -> bool:
    try:
        res = requests.get(f"{url}/health", timeout=3)
        return res.status_code == 200
    except Exception:
        return False

def get_documents(url: str) -> Dict[str, Any]:
    try:
        res = requests.get(f"{url}/api/documents", timeout=5)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return {"count": 0, "total_chunks": 0, "documents": []}

def upload_files(url: str, files: list) -> Optional[Dict[str, Any]]:
    payload = [("files", (f.name, f.getvalue(), "application/pdf")) for f in files]
    try:
        res = requests.post(f"{url}/api/upload", files=payload, timeout=90)
        if res.status_code == 200:
            return res.json()
        st.error(f"Upload error: {res.text}")
    except Exception as e:
        st.error(f"Connection failed: {e}")
    return None

def run_query(url: str, query: str, top_k: int = 5) -> Optional[Dict[str, Any]]:
    try:
        res = requests.post(f"{url}/api/query", json={"query": query, "top_k": top_k}, timeout=60)
        if res.status_code == 200:
            return res.json()
        st.error(f"Query error: {res.text}")
    except Exception as e:
        st.error(f"Connection failed: {e}")
    return None

def clear_vault(url: str) -> bool:
    try:
        res = requests.delete(f"{url}/api/documents", timeout=10)
        return res.status_code == 200
    except Exception:
        return False

# ==============================================================================
# SIDEBAR
# ==============================================================================
is_online = get_backend_health(st.session_state.api_url)
doc_data = get_documents(st.session_state.api_url) if is_online else {"count": 0, "total_chunks": 0, "documents": []}

with st.sidebar:
    st.markdown("### Synthexis")
    st.markdown("<p style='font-size: 0.85rem; color: #64748b; margin-top: -0.5rem;'>Multimodal Document Intelligence</p>", unsafe_allow_html=True)
    
    # Status indicator
    if is_online:
        st.markdown("""
            <div class="status-indicator" style="margin-bottom: 1.5rem;">
                <span class="dot-online"></span> API Connected
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="status-indicator" style="margin-bottom: 1.5rem;">
                <span class="dot-offline"></span> API Offline (127.0.0.1:8000)
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Configuration")
    
    new_api_url = st.text_input(
        "API Endpoint",
        value=st.session_state.api_url,
        help="Backend host address"
    )
    if new_api_url != st.session_state.api_url:
        st.session_state.api_url = new_api_url
        st.rerun()

    top_k_chunks = st.slider(
        "Context Chunks (Top-K)",
        min_value=1,
        max_value=15,
        value=5,
        help="Number of retrieved chunks used for synthesis"
    )

    st.markdown("---")
    st.markdown(f"#### Active Documents ({doc_data.get('count', 0)})")
    
    docs_list = doc_data.get("documents", [])
    if docs_list:
        for d in docs_list:
            st.markdown(
                f"<div style='font-size: 0.82rem; padding: 0.4rem 0; border-bottom: 1px solid #e2e8f0;'>"
                f"<b>{d['filename']}</b><br/>"
                f"<span style='color: #64748b;'>{d['chunk_count']} chunks • {d['image_count']} visuals</span>"
                f"</div>",
                unsafe_allow_html=True
            )
        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
        if st.button("Clear Vault", use_container_width=True, type="secondary"):
            if clear_vault(st.session_state.api_url):
                st.session_state.latest_response = None
                st.success("Corpus reset.")
                st.rerun()
    else:
        st.markdown("<p style='font-size: 0.82rem; color: #94a3b8;'>No documents indexed yet.</p>", unsafe_allow_html=True)

# ==============================================================================
# MAIN PAGE HEADER
# ==============================================================================
st.markdown("""
    <div class="hero-minimal-banner">
        <h1 class="hero-title">Synthexis</h1>
        <p class="hero-subtitle">Cross-document multimodal comparison and evidence-grounded synthesis.</p>
    </div>
""", unsafe_allow_html=True)

# Overview Metrics
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""
        <div class="metric-tile">
            <div class="metric-tile-label">Indexed Documents</div>
            <div class="metric-tile-number">{doc_data.get('count', 0)}</div>
        </div>
    """, unsafe_allow_html=True)
with c2:
    st.markdown(f"""
        <div class="metric-tile">
            <div class="metric-tile-label">Indexed Chunks</div>
            <div class="metric-tile-number">{doc_data.get('total_chunks', 0)}</div>
        </div>
    """, unsafe_allow_html=True)
with c3:
    st.markdown("""
        <div class="metric-tile">
            <div class="metric-tile-label">Retrieval Mode</div>
            <div class="metric-tile-number" style="font-size: 1.15rem; font-weight: 600; padding-top: 0.2rem;">Hybrid (BM25 + Dense)</div>
        </div>
    """, unsafe_allow_html=True)
with c4:
    st.markdown("""
        <div class="metric-tile">
            <div class="metric-tile-label">Synthesis Engine</div>
            <div class="metric-tile-number" style="font-size: 1.15rem; font-weight: 600; padding-top: 0.2rem;">Multimodal Vision</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

# ==============================================================================
# WORKSPACES
# ==============================================================================
tab_analysis, tab_vault = st.tabs([
    "Analysis & Synthesis",
    "Document Vault (Upload)"
])

# ------------------------------------------------------------------------------
# TAB 1: ANALYSIS & SYNTHESIS
# ------------------------------------------------------------------------------
with tab_analysis:
    st.markdown("#### Document Query & Comparison")
    
    # Prompt presets
    st.markdown("<p style='font-size: 0.8rem; color: #64748b; margin-bottom: 0.35rem;'>Quick prompts:</p>", unsafe_allow_html=True)
    p_col1, p_col2, p_col3 = st.columns(3)
    
    with p_col1:
        if st.button("Compare methodologies across documents", use_container_width=True, type="secondary"):
            st.session_state.query_input = "Compare the methodologies, dataset sizes, and evaluation approaches used across the documents."
    with p_col2:
        if st.button("Extract benchmark metrics & tables", use_container_width=True, type="secondary"):
            st.session_state.query_input = "Extract all key benchmark metrics, latency figures, and accuracy scores from the text and charts."
    with p_col3:
        if st.button("Analyze diagrams & visual figures", use_container_width=True, type="secondary"):
            st.session_state.query_input = "Summarize the architectural diagrams and charts present in the documents, highlighting core tradeoffs."

    query_text = st.text_area(
        "Enter query or comparison instruction:",
        value=st.session_state.query_input,
        placeholder="e.g. Compare the retrieval latency and storage efficiency discussed across the papers.",
        height=95
    )

    if st.button("Run Synthesis", type="primary"):
        if not is_online:
            st.error("API backend is offline. Ensure `main_api.py` is running.")
        elif doc_data.get("count", 0) == 0:
            st.warning("Please upload one or more PDF files in the 'Document Vault' tab first.")
        elif not query_text.strip():
            st.warning("Please enter a question or instruction.")
        else:
            with st.spinner("Retrieving evidence and generating synthesis..."):
                result = run_query(st.session_state.api_url, query_text.strip(), top_k=top_k_chunks)
                if result:
                    st.session_state.latest_response = result

    # Results Section
    if st.session_state.latest_response:
        res = st.session_state.latest_response
        st.markdown("---")
        
        # Header with latency badge
        st.markdown(f"""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <h4 style="margin: 0;">Synthesis Result</h4>
                <span class="clean-badge">Latency: {res.get('latency_ms', 0)} ms</span>
            </div>
        """, unsafe_allow_html=True)

        # Markdown answer
        st.markdown(f"""
            <div class="synthesis-result-card">
                {res.get('synthesized_answer', '')}
            </div>
        """, unsafe_allow_html=True)

        # Visual context (charts & diagrams)
        images = res.get("images", [])
        if images:
            st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
            st.markdown("#### Extracted Figures & Charts")
            img_cols = st.columns(min(len(images), 3))
            for idx, img in enumerate(images):
                with img_cols[idx % len(img_cols)]:
                    st.markdown(f"""
                        <div class="visual-artifact-card">
                            <img src="{img.get('base64_data')}" style="max-width: 100%; border-radius: 4px; max-height: 240px; object-fit: contain;" />
                            <p style="font-size: 0.75rem; color: #64748b; margin: 0.5rem 0 0 0;">
                                <b>{img.get('image_id')}</b> ({img.get('source_doc')})
                            </p>
                        </div>
                    """, unsafe_allow_html=True)

        # Evidence Inspector
        retrieved_chunks = res.get("retrieved_chunks", [])
        if retrieved_chunks:
            st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
            with st.expander(f"Retrieved Evidence Chunks ({len(retrieved_chunks)})", expanded=False):
                st.caption(f"Method: {res.get('retrieval_method')}")
                for idx, chunk in enumerate(retrieved_chunks):
                    st.markdown(f"""
                        <div class="source-chunk-card">
                            <div style="font-weight: 600; font-size: 0.78rem; margin-bottom: 0.25rem; color: #0f172a;">
                                Evidence #{idx + 1}
                            </div>
                            {chunk}
                        </div>
                    """, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# TAB 2: DOCUMENT VAULT
# ------------------------------------------------------------------------------
with tab_vault:
    st.markdown("#### Document Ingestion")
    st.markdown("<p style='font-size: 0.88rem; color: #64748b;'>Upload PDF documents to parse text chunks and isolate embedded diagrams.</p>", unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        st.markdown(f"**Selected {len(uploaded_files)} file(s):**")
        for f in uploaded_files:
            st.markdown(
                f"<div style='font-size: 0.85rem; padding: 0.4rem 0.6rem; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 4px; margin-bottom: 0.35rem;'>"
                f"📄 <b>{f.name}</b> ({round(len(f.getvalue()) / 1024, 1)} KB)"
                f"</div>",
                unsafe_allow_html=True
            )

        if st.button("Ingest & Index", type="primary"):
            if not is_online:
                st.error("API backend is offline. Start `main_api.py` first.")
            else:
                progress = st.progress(0, text="Uploading documents...")
                progress.progress(40, text="Parsing text and extracting visual figures...")
                res = upload_files(st.session_state.api_url, uploaded_files)
                if res:
                    progress.progress(100, text="Complete.")
                    st.success(res.get("message", "Documents indexed successfully."))
                    st.rerun()

    # Active Documents Overview
    st.markdown("---")
    st.markdown("#### Indexed Document Vault")
    if docs_list:
        for doc in docs_list:
            st.markdown(f"""
                <div class="doc-card">
                    <div>
                        <div style="font-weight: 600; font-size: 0.9rem; color: #0f172a;">{doc['filename']}</div>
                        <div style="font-size: 0.78rem; color: #64748b;">
                            {doc['chunk_count']} text chunks • {doc['image_count']} visual artifacts • Uploaded: {doc.get('uploaded_at', 'Recently')}
                        </div>
                    </div>
                    <span class="clean-badge">{round(doc['filesize_bytes'] / 1024, 1)} KB</span>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("<p style='font-size: 0.85rem; color: #94a3b8;'>No documents currently stored in the vault.</p>", unsafe_allow_html=True)
