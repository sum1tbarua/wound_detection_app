# app/app.py
"""
Streamlit front-end for the wound detection demo.

- lets the user upload an image (hand/arm/leg)
- runs YOLO inference and shows: original | detection | XAI
- calls the local LLM (via src.llm.first_aid) to generate first-aid
- if detection is low-confidence or 'wound_unknown', it asks the user to clarify
- designed to be the entry point for the GenAI course project
"""

import os
import sys
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image

# Make project root importable (…/wound_detection_app)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Local modules
from src.detection.yolo_infer import run_detection
from src.llm.first_aid import generate_first_aid

# --------------------------
# Page / Sidebar
# --------------------------
st.set_page_config(
    page_title="Wound Detection & First-Aid",
    layout="wide",
    page_icon="🩹",
)

st.sidebar.header("App Status")
st.sidebar.success("✅ Vision pipeline ready")
st.sidebar.info(f"🧠 LLM: Ollama ({os.getenv('OLLAMA_MODEL', 'mistral')})")
st.sidebar.caption(
    "If the model is unsure, the app will ask for clarification before generating first-aid."
)

# UI threshold for “confident enough” (independent of YOLO runtime conf)
UI_CONFIDENCE = 0.35

with st.sidebar.expander("ℹ️ How to use"):
    st.write(
        "1. Upload a wound/hand/leg/arm image\n"
        "2. The model will detect and highlight the area\n"
        "3. If it's not sure, it'll ask you for further clarification\n"
        "4. Then it will generate first-aid guidance"
    )

# --------------------------
# Header
# --------------------------
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

# --------------------------
# Upload
# --------------------------
uploaded = st.file_uploader("Upload wound image", type=["jpg", "jpeg", "png"])

if uploaded:
    # Save upload to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    # Load original for display
    orig_img = Image.open(tmp_path).convert("RGB")

    # Run detection + (light) XAI
    try:
        detections, detection_image_path, xai_image_path = run_detection(tmp_path)
    except Exception as e:
        st.error(f"Detection failed: {e}")
        st.stop()

    # --------------------------
    # Visuals: Original | Detection | XAI
    # --------------------------
    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.subheader("Original")
        st.image(orig_img, use_container_width=True)
        st.caption("Uploaded image")

    with col2:
        st.subheader("Detection")
        st.image(str(detection_image_path), use_container_width=True)
        st.caption("YOLOv8-seg predicted bounding box / mask. Uses your UI confidence.")

    with col3:
        st.subheader("XAI / Grad-CAM")
        st.image(str(xai_image_path), use_container_width=True)
        st.caption("Explainable AI (XAI): red/pink areas show **where** the model focused.")

    # Show raw detections
    st.markdown("### Detected objects")
    st.json(detections)

    # Cache detections for the rerun cycle
    st.session_state.last_detections = detections

    # Decide if we’re confident or need human-in-the-loop
    has_unknown = any(d.get("label") == "wound_unknown" for d in detections)
    confident_dets = [
        d
        for d in detections
        if d.get("label") != "wound_unknown" and float(d.get("confidence", 0)) >= UI_CONFIDENCE
    ]
    has_confident = len(confident_dets) > 0

    st.markdown("---")

    # -----------------------------
    # LLM section
    # -----------------------------
    if has_unknown:
        st.subheader("🤔 Model is uncertain")
        st.caption(
            "Please describe what you see (e.g., ‘cut on the hand’, ‘burn on the leg’) so we can generate the right first-aid."
        )

        # Ask a polite question first (generate_first_aid should return a short question here)
        followup_md = generate_first_aid(st.session_state.last_detections)
        if followup_md:
            st.markdown("### 📝 Clarification needed")
            st.markdown(followup_md)

        # Collect a short user clarification and then generate final guidance
        with st.form("clarification_form"):
            user_clarification = st.text_input(
                "Short description (e.g., 'hand cut', 'arm burn near wrist')",
                placeholder="e.g., hand cut",
                key="clarification_text",
            )
            submitted = st.form_submit_button("Generate first-aid")

        if submitted and user_clarification.strip():
            clarified = st.session_state.last_detections + [
                {"label": "user_clarification", "text": user_clarification.strip()}
            ]
            with st.spinner("Generating first-aid from local LLM…"):
                final_md = generate_first_aid(clarified)
            st.markdown("### 🩹 First-Aid Instructions")
            st.markdown(final_md)
            

    elif has_confident:
        # Confident → go straight to final instructions
        st.subheader("🩹 First-Aid Instructions")
        with st.spinner("Generating first-aid from local LLM…"):
            fa_md = generate_first_aid(detections)
        st.markdown(fa_md)
        

    else:
        # Not unknown but also not confident → prompt for help
        st.subheader("🤔 Low confidence")
        st.caption("Could you briefly describe what you see so we can tailor the guidance?")
        with st.form("low_conf_form"):
            user_clarification = st.text_input(
                "Short description (e.g., 'small hand cut', 'minor arm burn')",
                key="low_conf_text",
            )
            submitted = st.form_submit_button("Generate first-aid")
        if submitted and user_clarification.strip():
            clarified = st.session_state.last_detections + [
                {"label": "user_clarification", "text": user_clarification.strip()}
            ]
            with st.spinner("Generating first-aid from local LLM…"):
                final_md = generate_first_aid(clarified)
            st.markdown("### 🩹 First-Aid Instructions")
            st.markdown(final_md)


    # -----------------------------
    # Footer / Disclaimer
    # -----------------------------
    st.markdown("---")
    st.info(
        "⚠️ This application is for educational / research purpose only and "
        "must not be used as professional medical advice."
    )

else:
    st.info("Upload an image to start.")
