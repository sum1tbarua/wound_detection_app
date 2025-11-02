# app/app.py

"""
Streamlit front-end for the wound detection demo.

- lets the user upload an image (hand/arm/leg)
- runs YOLO inference and shows: original | detection | XAI
- calls the local LLM (via src.llm.first_aid) to generate first-aid
- if detection is low-confidence or 'wound_unknown', it asks the user to clarify
- designed to be the entry point for the GenAI course project
"""


import sys
import os
from pathlib import Path
import tempfile

import streamlit as st
from PIL import Image

# To make src importable 
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.detection.yolo_infer import run_detection
from src.llm.first_aid import generate_first_aid
# from src.prompts import build_user_prompt  # not needed right now

# -----------------------------------------------------------------------------
# PAGE / SIDEBAR
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Wound Detection + First-Aid Assistant",
    layout="wide",
    page_icon="🩺",
)

# Sidebar: status + LLM info
st.sidebar.markdown("## ⚙️ App Status")
st.sidebar.success("✅ Vision model loaded")
st.sidebar.info(f"🧠 LLM: Ollama ({os.getenv('OLLAMA_MODEL', 'mistral')})")
st.sidebar.warning("⚠️ If detection is uncertain, the app will ask you for details.")

with st.sidebar.expander("ℹ️ How to use"):
    st.write(
        "1. Upload a wound/hand/leg/arm image\n"
        "2. The model will detect and highlight the area\n"
        "3. If it's not sure, it'll ask you for further clarification\n"
        "4. Then it will generate first-aid guidance"
    )

# -----------------------------------------------------------------------------
# HEADER
# -----------------------------------------------------------------------------
st.markdown(
    """
    <h1 style='text-align: center;'>🩺 AI-Powered Wound Detection & Explainable First-Aid</h1>
    <p style='text-align: center; color: #aaa;'>
        Detect wounds using computer vision, interpret model attention via XAI, and generate first-aid guidance through a local LLM
    </p>
    """,
    unsafe_allow_html=True,
)

st.write("")  # small spacing

# -----------------------------------------------------------------------------
# FILE UPLOAD
# -----------------------------------------------------------------------------
uploaded = st.file_uploader("📤 Upload wound image", type=["jpg", "jpeg", "png"])

if uploaded:
    # 1) save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    # original image for display
    orig_img = Image.open(tmp_path).convert("RGB")

    # 2) run YOLO + get overlay images
    with st.spinner("🔍 Running wound detection..."):
        try:
            detections, detection_image_path, xai_image_path = run_detection(tmp_path)
        except Exception as e:
            st.error(f"Detection failed: {e}")
            st.stop()

    st.success("✅ Detection complete")

    # -----------------------------------------------------------------------------
    # LAYOUT: 3 views on top
    # -----------------------------------------------------------------------------
    tab1, tab2 = st.tabs(["🖼️ Visual Results", "📦 Raw / Debug Info"])

    with tab1:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Original")
            st.image(orig_img, use_container_width=True)
            st.markdown(
                """
                **Original Image:**  
                This is the uploaded wound image before any processing.
                """
            )

        with col2:
            st.subheader("Detection")
            st.image(str(detection_image_path), use_container_width=True)
            st.markdown(
                """
                **Detection Output:**  
                The YOLO segmentation model highlights detected wound regions 
                and labels them with confidence scores.
                """
            )
            
        with col3:
            st.subheader("XAI / Grad-CAM")
            st.image(str(xai_image_path), use_container_width=True)
            st.markdown(
                 """
                    **Explainable AI (XAI):**  
                    The red-tinted regions highlight *where* the vision model focused its attention 
                    when analyzing the image.  
                    This feature adds transparency by helping users understand the model’s reasoning 
                    instead of treating it as a black box.
                """
            )
        st.markdown("---")

    # -----------------------------------------------------------------------------
    # RAW + CONFIDENT FILTER
    # -----------------------------------------------------------------------------
    with tab2:
        st.markdown("### Detected objects (raw)")
        st.json(detections)

    # filter out noisy ones for normal users
    confident_dets = [d for d in detections if d["confidence"] >= 0.35]
    has_confident = len(confident_dets) > 0
    has_unknown = any(d["label"] == "wound_unknown" for d in detections)

    st.markdown("---")

    # -----------------------------------------------------------------------------
    # PRETTY SUMMARY OF DETECTIONS
    # -----------------------------------------------------------------------------
    st.markdown("### 🧾 Detection summary")

    if has_confident:
        for d in confident_dets:
            conf_pct = int(d["confidence"] * 100)
            st.markdown(
                f"🟢 **{d['label']}** — confidence: `{conf_pct}%`"
            )
    else:
        st.warning("⚠️ The vision model is not confident about this image.")

    # -----------------------------------------------------------------------------
    # LLM / FIRST-AID SECTION
    # -----------------------------------------------------------------------------
    st.markdown("---")

    # CASE 1: model not sure → ask user
    if has_unknown or not has_confident:
        st.subheader("🤔 I need a little more detail")
        st.write(
            "I detected a hand/arm/leg but I'm not fully sure about the wound type. "
            "Tell me what it is so I can generate the right first-aid instructions."
        )

        user_clarification = st.text_input(
            "Describe the wound (example: 'burn on the hand', 'small cut on arm', 'deep cut on leg'):"
        )

        if user_clarification:
            clarified = detections + [
                {"label": "user_clarification", "text": user_clarification}
            ]
            with st.spinner("✍️ Generating first-aid..."):
                fa_md = generate_first_aid(clarified)
            st.markdown("### 🩹 First-Aid Instructions")
            st.markdown(fa_md)
    # CASE 2: model is confident → just generate
    else:
        st.subheader("🩹 First-Aid Instructions")
        with st.spinner("✍️ Generating first-aid..."):
            fa_md = generate_first_aid(detections)
        st.markdown(fa_md)

    # -----------------------------------------------------------------------------
    # DISCLAIMER
    # -----------------------------------------------------------------------------
    st.info(
        "⚠️ This application is for educational / classroom demonstration only and "
        "must not be used as professional medical advice."
    )

else:
    # no image yet
    st.info("👆 Upload an image to start.")
