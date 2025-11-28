import os
from pathlib import Path
import json
import tempfile
from typing import Optional

import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models

import streamlit as st
from ultralytics import YOLO
import requests

# ============================================================
# CONFIG / PATHS
# ============================================================
ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

YOLO_MODEL_PATH = MODELS_DIR / "yolo_best.pt"
BODY_MODEL_PATH = MODELS_DIR / "body_classifier_resnet50_best.pth"
CLASS_JSON_PATH = MODELS_DIR / "body_classifier_classes.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOW_WOUND_CONF = 0.40  # threshold for "low confidence" wound detection

# ============================================================
# PAGE CONFIG + GLOBAL STYLING
# ============================================================
st.set_page_config(page_title="Wound Detection and First-Aid", layout="wide")


st.markdown(
    """
    <style>
        

        /* -------- Sidebar styling -------- */
        [data-testid="stSidebar"] {
            background-color: #0A1A2F !important;
        }
        [data-testid="stSidebar"] * {
            color: #FFFFFF !important;
        }
        [data-testid="stSidebar"] ::-webkit-scrollbar {
            width: 6px;
        }
        [data-testid="stSidebar"] ::-webkit-scrollbar-thumb {
            background-color: #1f3b5c;
            border-radius: 10px;
        }

        /* -------- Main area -------- */
        .main {
            background-color: #F7F9FC;
        }

        /* -------- Headers / text -------- */
        h1, h2, h3, h4 {
            color: #0A1A2F !important;
        }
        
        .banner h2 {
            color: white !important;
        }
        .banner p {
            color: #dce4ec !important;
        }

        /* -------- Buttons -------- */
        .stButton>button {
            background-color: #0A1A2F !important;
            color: white !important;
            border-radius: 6px;
            height: 3em;
            min-width: 12em;
            font-size: 15px;
        }
        .stButton>button:hover {
            background-color: #102b4d !important;
        }

        /* Small badges */
        .badge-ok {
            display:inline-block;
            padding:2px 8px;
            border-radius:10px;
            background:#d1fae5;
            color:#065f46;
            font-size:0.8rem;
            margin-left:6px;
        }
        .badge-warn {
            display:inline-block;
            padding:2px 8px;
            border-radius:10px;
            background:#fef3c7;
            color:#92400e;
            font-size:0.8rem;
            margin-left:6px;
        }
        .badge-err {
            display:inline-block;
            padding:2px 8px;
            border-radius:10px;
            background:#fee2e2;
            color:#b91c1c;
            font-size:0.8rem;
            margin-left:6px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# UTILS – CLASS NAMES
# ============================================================
def load_class_names(json_path: Path):
    if not json_path.exists():
        return ["arm", "hand", "leg", "other"]

    with open(json_path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict):
        items = sorted(data.items(), key=lambda kv: int(kv[0]))
        class_names = [v for _, v in items]
    elif isinstance(data, list):
        class_names = data
    else:
        raise ValueError("Unsupported format in body_classifier_classes.json")

    return class_names


CLASS_NAMES = load_class_names(CLASS_JSON_PATH)

# ============================================================
# MODEL LOADING
# ============================================================
@st.cache_resource(show_spinner=True)
def load_yolo_model():
    if not YOLO_MODEL_PATH.exists():
        st.error(f"YOLO model not found at {YOLO_MODEL_PATH}")
        st.stop()
    return YOLO(str(YOLO_MODEL_PATH))


@st.cache_resource(show_spinner=True)
def load_body_model():
    if not BODY_MODEL_PATH.exists():
        st.error(f"Body classifier model not found at {BODY_MODEL_PATH}")
        st.stop()

    num_classes = len(CLASS_NAMES)
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    state = torch.load(BODY_MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


YOLO_MODEL = load_yolo_model()
BODY_MODEL = load_body_model()

# ============================================================
# SIDEBAR CONTENT
# ============================================================
with st.sidebar:
    
    st.markdown("<h1>About</h1>", unsafe_allow_html=True)
    st.write(
        "Prototype multimodal assistant combining:\n"
        "- **YOLOv11:** Wound Detection\n"
        "- **ResNet-50:** Body-region classifier + Grad-CAM\n"
        "- **Ollama Mistral:** First-aid Generation"
    )

    st.markdown("---")
    st.subheader("📦 Model Status")
    st.markdown("Vision Models <span> ✅</span>", unsafe_allow_html=True)
    st.markdown("LLM (Ollama) <span> ✅</span>", unsafe_allow_html=True)


    st.markdown("---")
    st.subheader("👨‍💻 Developers")
    st.write("**Sumit Barua**")
    st.write("Computer Science, Western Michigan University")

    st.write("**Ruth Bahre**")
    st.write("Electrical and Computer Engineering, Western Michigan University")


# ============================================================
# IMAGE TRANSFORMS + HELPERS
# ============================================================
IMG_SIZE = 224

body_transform = T.Compose(
    [
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),  # no normalization – matches your training
    ]
)


def pil_to_cv2(pil_img: Image.Image):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv2_img: np.ndarray):
    return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))


# ============================================================
# YOLO WOUND DETECTION
# ============================================================
def yolo_wound_detect(image_pil: Image.Image, conf_thres=0.3):
    img_bgr = pil_to_cv2(image_pil)

    results = YOLO_MODEL(
        img_bgr,
        conf=conf_thres,
        verbose=False,
    )
    res = results[0]

    # overlay
    overlay_bgr = res.plot()
    overlay_pil = cv2_to_pil(overlay_bgr)

    if res.boxes is None or len(res.boxes) == 0:
        wound_type = "wound_unknown"
        wound_conf = 0.0
    else:
        boxes = res.boxes
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        idx = int(confs.argmax())
        wound_conf = float(confs[idx])
        cls_id = int(cls_ids[idx])
        wound_type = res.names.get(cls_id, f"class_{cls_id}")

    return overlay_pil, wound_type, wound_conf


# ============================================================
# BODY CLASSIFIER + GRAD-CAM
# ============================================================
def classify_body_and_gradcam(image_pil: Image.Image):
    BODY_MODEL.eval()

    img_tensor = body_transform(image_pil).unsqueeze(0).to(DEVICE)

    conv_output = None
    conv_grad = None

    def fwd_hook(module, inp, out):
        nonlocal conv_output
        conv_output = out

    def bwd_hook(module, grad_in, grad_out):
        nonlocal conv_grad
        conv_grad = grad_out[0]

    handle_fwd = BODY_MODEL.layer4.register_forward_hook(fwd_hook)
    handle_bwd = BODY_MODEL.layer4.register_full_backward_hook(bwd_hook)

    img_tensor.requires_grad_(True)
    BODY_MODEL.zero_grad()

    logits = BODY_MODEL(img_tensor)
    probs_t = torch.softmax(logits, dim=1)[0]
    probs = probs_t.detach().cpu().numpy()

    top_idx = int(np.argmax(probs))
    body_label = CLASS_NAMES[top_idx]
    body_conf = float(probs[top_idx])

    score = logits[0, top_idx]
    score.backward()

    with torch.no_grad():
        if conv_output is None or conv_grad is None:
            handle_fwd.remove()
            handle_bwd.remove()
            return body_label, body_conf, image_pil

        conv_output_ = conv_output[0]
        conv_grad_ = conv_grad[0]

        weights = conv_grad_.mean(dim=(1, 2))
        cam = torch.zeros_like(conv_output_[0])
        for c, w in enumerate(weights):
            cam += w * conv_output_[c]

        cam = torch.relu(cam)
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        cam_np = cam.cpu().numpy()

        img_np = np.array(image_pil)
        h, w, _ = img_np.shape

        cam_np = cv2.resize(cam_np, (w, h))
        cam_np = np.uint8(255 * cam_np)

        heatmap = cv2.applyColorMap(cam_np, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        overlay = (0.5 * img_np + 0.5 * heatmap).astype(np.uint8)
        overlay_pil = Image.fromarray(overlay)

    handle_fwd.remove()
    handle_bwd.remove()

    return body_label, body_conf, overlay_pil


# ============================================================
# LLM PROMPTS
# ============================================================
def build_first_aid_prompt_from_models(
    wound_type: str,
    wound_conf: float,
    body_location: str,
    body_conf: Optional[float] = None,
    user_body_override: Optional[str] = None,
) -> str:
    """
    Used when the model is confident, possibly with a user override
    for body location.
    """
    if user_body_override:
        location_desc = user_body_override
    else:
        location_desc = body_location

    lines = [
        "You are a cautious first-aid helper, not a doctor.\n",
        "You MUST talk ONLY about the single wound described below.\n",
        "Do NOT give general first-aid for unrelated emergencies such as "
        "unconsciousness, choking, broken bones, spinal injury, poisoning, "
        "allergic reactions, etc.\n",
        "Focus just on basic wound care for this one wound and when to seek urgent care.\n\n",
        "Wound information:\n",
        f"- Model wound label: {wound_type}\n",
        f"- Wound detection confidence: {wound_conf:.2f}\n",
        f"- Body location (approximate): {location_desc}\n",
    ]
    if body_conf is not None and not user_body_override:
        lines.append(f"- Body-location confidence: {body_conf:.2f}\n")

    lines.extend(
        [
            "\nTask:\n",
            "Write 5–7 short bullet points of general, non-diagnostic first-aid "
            "advice for THIS wound only.\n",
            "- Use simple language.\n",
            "- Mention gentle cleaning, protection, monitoring for infection, and "
            "when to seek professional or emergency care.\n",
            "- Do NOT talk about CPR, unconsciousness, seizures, choking, spinal "
            "injuries, poisoning, or any other unrelated condition.\n",
        ]
    )
    return "".join(lines)


def build_first_aid_prompt_from_user_description(user_desc: str) -> str:
    """
    Used when the model is not confident and we rely on the user's description.
    """
    return (
        "You are a cautious first-aid helper, not a doctor.\n"
        "The user will briefly describe a single wound and where it is.\n"
        "Use ONLY that description to give basic wound-care advice.\n\n"
        f"User description of the wound:\n{user_desc}\n\n"
        "Task:\n"
        "Write 5–7 short bullet points of general, non-diagnostic first-aid "
        "advice for THIS wound only.\n"
        "- Use simple language.\n"
        "- Assume this is a minor to moderate injury; if it sounds severe, "
        "tell them to seek emergency care.\n"
        "- Talk about gentle cleaning, keeping it covered, monitoring for "
        "infection, and when to see a doctor.\n"
        "- Do NOT discuss unrelated emergencies (e.g., CPR, poisoning, "
        "broken bones, spinal injury).\n"
    )

def build_xai_explanation_prompt(
    wound_type: str,
    wound_conf: float,
    body_location: str,
    body_conf: float,
    first_aid_text: str,
) -> str:
    """
    Prompt used to get a layperson-friendly explanation of
    why the first-aid instructions make sense, given the
    model outputs and Grad-CAM body heatmap.
    """
    return (
        "You are explaining how an AI vision + language pipeline produced "
        "the following first-aid suggestions.\n\n"
        "FIRST-AID INSTRUCTIONS:\n"
        f"{first_aid_text}\n\n"
        "VISION MODEL INFORMATION:\n"
        f"- Wound detector label: {wound_type} (confidence {wound_conf:.2f})\n"
        f"- Body-location classifier label: {body_location} (confidence {body_conf:.2f})\n"
        "- The system also uses a Grad-CAM heatmap over the body image to "
        "highlight the region it thinks contains the wound.\n\n"
        "TASK:\n"
        "Explain in 3–5 SHORT bullet points, in non-technical language, "
        "why these instructions are reasonable for this wound type and "
        "body location.\n"
        "- Mention that the wound detector focused on the region inside the "
        "bounding box around the wound.\n"
        "- Mention that the body-location classifier and its Grad-CAM heatmap "
        "focused on the predicted body region (e.g., hand, arm, leg).\n"
        "- Relate the advice to the wound being a cut, burn, or unknown type "
        "and to its approximate body location.\n"
        "- Emphasize that this is NOT a diagnosis and that users must consult "
        "a clinician for real medical decisions.\n"
        "- Do NOT introduce other unrelated emergencies like CPR, choking, "
        "spinal injury, poisoning, etc.\n"
    )


def run_llm_locally(prompt: str, model_name: str = "mistral") -> str:
    """
    Call a local Ollama model (e.g., 'mistral') via HTTP API.
    """
    url = "http://localhost:11434/api/chat"

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a cautious first-aid helper. Follow the prompt "
                    "instructions exactly and never drift into unrelated topics."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }

    try:
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        if "message" in data and "content" in data["message"]:
            return data["message"]["content"].strip()
        return f"[Unexpected Ollama response]\n{data}"
    except Exception as e:
        return (
            "[LLM ERROR] Could not reach Ollama / mistral.\n"
            f"Error: {e}\n\n"
            "Here is the prompt that *would* have been sent:\n"
            + prompt[:800]
        )


# ============================================================
# MAIN UI
# ============================================================
st.markdown(
    """
    <div class='banner' style='padding:18px; background-color:#0A1A2F; border-radius:10px; margin-bottom:20px;'>
        <h2 style='text-align:center; margin-bottom:4px;'>
            AI System for Wound Detection & Explainable First-Aid Recommendations
        </h2>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style='background-color:white; padding:18px; border-radius:10px;
                box-shadow:0 2px 8px rgba(0,0,0,0.06); margin-bottom:18px;'>
        <h3>Upload an image</h3>
        <p style='color:#4b5563; margin-bottom:0;'>
            Upload a skin wound photo. The system will detect the wound region,
            estimate the body location, visualize Grad-CAM, and then either ask
            for your clarification or generate first-aid guidance.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

conf_thres = st.sidebar.slider(
    "YOLO detection confidence",
    min_value=0.1,
    max_value=0.9,
    value=0.3,
    step=0.05,
)

uploaded_file = st.file_uploader("Choose a wound image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, uploaded_file.name)
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.read())

    image_pil = Image.open(tmp_path).convert("RGB")
    
    yolo_overlay_pil, wound_type, wound_conf = yolo_wound_detect(
            image_pil, conf_thres=conf_thres
        )
    
    body_label, body_conf, gradcam_overlay_pil = classify_body_and_gradcam(
            image_pil
        )

    # ------------------------------------------------------------------
    # Decide mode for LLM
    # ------------------------------------------------------------------
    wound_unknown_flag = "unknown" in wound_type.lower()
    low_wound_conf = wound_conf < LOW_WOUND_CONF
    body_other_flag = body_label == "other"

    # Modes:
    #  - "confident"       -> auto first-aid
    #  - "body_unknown"    -> ask only for body location, then first-aid
    #  - "uncertain_wound" -> ask for full description, then first-aid
    if wound_unknown_flag or low_wound_conf:
        llm_mode = "uncertain_wound"
    elif body_other_flag:
        llm_mode = "body_unknown"
    else:
        llm_mode = "confident"

    # ------------------------------------------------------------------
    # Tabs: Vision vs LLM
    # ------------------------------------------------------------------
    st.markdown("""
    <style>

    /* Make tab labels MUCH bigger */
    .stTabs [data-baseweb="tab"] p {
        font-size: 22px !important;
        font-weight: 700 !important;
        padding-top: 8px !important;
        padding-bottom: 8px !important;
    }

    /* Increase spacing inside tab container */
    .stTabs [data-baseweb="tab"] {
        padding-left: 20px !important;
        padding-right: 20px !important;
    }

    /* Active tab — red underline + red text */
    .stTabs [data-baseweb="tab"][aria-selected="true"] p {
        color: #E63946 !important;   /* premium red */
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        border-bottom: 4px solid #E63946 !important;
    }

    /* Inactive tab muted */
    .stTabs [data-baseweb="tab"]:not([aria-selected="true"]) p {
        color: #555 !important;
    }

    </style>
""", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🔍 Vision Outputs", "🧰 First-Aid Assistance"])

    # ========================= TAB 1 – VISION =========================
    with tab1:
        
        st.markdown("")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Original image")
            st.image(image_pil, width='stretch')

        with col2:
            st.subheader("Wound Detection")
            st.image(yolo_overlay_pil, width='stretch')
            # st.caption(f"Prediction: **{wound_type}** (conf={wound_conf:.2f})")

        with col3:
            st.subheader("Body Grad-CAM")
            st.image(gradcam_overlay_pil, width='stretch')
            # st.caption(f"Body location: **{body_label}** (conf={body_conf:.2f})")

        st.markdown("---")
                # ---------- Premium numeric card ----------
        # derive label + colours once
        if wound_conf > 0.8:
            wound_level = "high"
            level_color_bg = "#dcfce7"
            level_color_text = "#166534"
        elif wound_conf > 0.5:
            wound_level = "medium"
            level_color_bg = "#fef9c3"
            level_color_text = "#92400e"
        else:
            wound_level = "low"
            level_color_bg = "#fee2e2"
            level_color_text = "#b91c1c"

        wound_pct = int(wound_conf * 100)
        body_pct = int(body_conf * 100)

        html_numeric = f"""
        <div style="
            background-color:white;
            padding:24px;
            border-radius:16px;
            box-shadow:0 14px 35px rgba(15,23,42,0.12);
            margin-top:24px;
            margin-bottom:8px;
            font-size:16px;
        ">
        <!-- header -->
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:20px;">
            <div style="display:flex; align-items:center; gap:10px;">
            <div style="
                width:38px; height:38px;
                border-radius:14px;
                background:linear-gradient(135deg,#1d4ed8,#22c55e);
                display:flex; align-items:center; justify-content:center;
                color:white; font-size:22px;
            ">📊</div>
            <div>
                <div style="font-size:22px; font-weight:700; color:#0f172a;">
                Numeric model outputs
                </div>
                <div style="font-size:13px; color:#6b7280;">
                Snapshot of what the vision models think for this image.
                </div>
            </div>
            </div>
            <div style="
                font-size:12px;
                text-transform:uppercase;
                letter-spacing:0.08em;
                color:#6b7280;
            ">
            YOLOv11 · ResNet-50 · Grad-CAM
            </div>
        </div>

        <!-- wound type -->
        <div style="margin-bottom:14px; display:flex; justify-content:space-between; align-items:center;">
            <div style="font-weight:600; color:#111827;">Wound type (YOLO)</div>
            <span style="
                background:#ecfdf5;
                color:#166534;
                padding:6px 14px;
                border-radius:999px;
                font-weight:600;
                font-family:monospace;
            ">{wound_type}</span>
        </div>

        <!-- wound confidence -->
        <div style="margin-bottom:18px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
            <div style="font-weight:600; color:#111827;">Wound confidence</div>
            <span style="
                background:#eef2ff;
                color:#4338ca;
                padding:4px 10px;
                border-radius:999px;
                font-weight:600;
                font-family:monospace;
            ">{wound_pct}%</span>
            </div>
            <div style="
                margin-top:6px;
                width:100%;
                height:8px;
                border-radius:999px;
                background:#e5e7eb;
                overflow:hidden;
            ">
            <div style="
                width:{wound_pct}%;
                height:100%;
                background:linear-gradient(90deg,#22c55e,#16a34a);
            "></div>
            </div>
        </div>

        <!-- body location -->
        <div style="margin-bottom:14px; display:flex; justify-content:space-between; align-items:center;">
            <div style="font-weight:600; color:#111827;">Body location (ResNet)</div>
            <span style="
                background:#f0fdf4;
                color:#166534;
                padding:6px 14px;
                border-radius:999px;
                font-weight:600;
                font-family:monospace;
            ">{body_label}</span>
        </div>

        <!-- body confidence -->
        <div style="margin-bottom:18px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
            <div style="font-weight:600; color:#111827;">Body-location confidence</div>
            <span style="
                background:#eef2ff;
                color:#4338ca;
                padding:4px 10px;
                border-radius:999px;
                font-weight:600;
                font-family:monospace;
            ">{body_pct}%</span>
            </div>
            <div style="
                margin-top:6px;
                width:100%;
                height:8px;
                border-radius:999px;
                background:#e5e7eb;
                overflow:hidden;
            ">
            <div style="
                width:{body_pct}%;
                height:100%;
                background:linear-gradient(90deg,#38bdf8,#1d4ed8);
            "></div>
            </div>
        </div>

        <!-- overall tag -->
        <div style="margin-top:10px; display:flex; align-items:center; gap:8px;">
            <div style="font-weight:600; color:#111827;">Overall wound confidence</div>
            <span style="
                background:{level_color_bg};
                color:{level_color_text};
                padding:6px 14px;
                border-radius:999px;
                font-weight:700;
                text-transform:capitalize;
            ">{wound_level}</span>
        </div>
        </div>
        """

        st.markdown(html_numeric, unsafe_allow_html=True)


    # ========================= TAB 2 – LLM =========================
    with tab2:
        st.markdown(
            """
            <div style='background-color:white; padding:18px; border-radius:10px;
                        box-shadow:0 2px 8px rgba(0,0,0,0.06); margin-top:15px;'>
                <h3>LLM first-aid assistant</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ---------- Mode A: Model confident – auto first-aid ----------
        if llm_mode == "confident":
            st.info(
                "The system is **reasonably confident** about both wound type and body "
                "location, so it will automatically generate first-aid guidance."
            )
            prompt = build_first_aid_prompt_from_models(
                wound_type=wound_type,
                wound_conf=wound_conf,
                body_location=body_label,
                body_conf=body_conf,
                user_body_override=None,
            )

            with st.expander("Show LLM prompt"):
                st.text(prompt)

            # 1) First-aid generation
            with st.spinner("Generating first-aid guidance from local LLM..."):
                first_aid_text = run_llm_locally(prompt)

            st.markdown("**LLM first-aid guidance:**")
            st.write(first_aid_text)

            # 2) XAI explanation for WHY these instructions make sense
            xai_prompt = build_xai_explanation_prompt(
                wound_type=wound_type,
                wound_conf=wound_conf,
                body_location=body_label,   # model’s body prediction
                body_conf=body_conf,
                first_aid_text=first_aid_text,
            )

            with st.expander("💡 Why these instructions? (LLM explanation)"):
                with st.spinner("Generating explanation from local LLM..."):
                    xai_text = run_llm_locally(xai_prompt)
                st.write(xai_text)


        # ---------- Mode B: Wound OK, body = 'other' ----------
        elif llm_mode == "body_unknown":
            st.warning(
                "The wound type looks reasonably clear, but the body location is "
                "classified as **'other'**. Please specify where on the body it is "
                "so we can generate more precise first-aid."
            )

            user_body = st.text_input(
                "Where on the body is the wound? (e.g., 'left index finger', 'upper right arm')"
            ).strip()

            if st.button("Generate first-aid"):
                if not user_body:
                    st.error("Please enter a short body-location description first.")
                else:
                    prompt = build_first_aid_prompt_from_models(
                        wound_type=wound_type,
                        wound_conf=wound_conf,
                        body_location=body_label,
                        body_conf=body_conf,
                        user_body_override=user_body,  # override used here
                    )
                    with st.expander("Show LLM prompt"):
                        st.text(prompt)

                    # 1) First-aid guidance using user-provided body location
                    with st.spinner("Generating first-aid guidance from local LLM..."):
                        first_aid_text = run_llm_locally(prompt)

                    st.markdown("**LLM first-aid guidance:**")
                    st.write(first_aid_text)

                    # 2) XAI explanation referencing the user body-location
                    xai_prompt = build_xai_explanation_prompt(
                        wound_type=wound_type,
                        wound_conf=wound_conf,
                        body_location=user_body,  # use user override in explanation
                        body_conf=body_conf,
                        first_aid_text=first_aid_text,
                    )

                    with st.expander("💡 Why these instructions? (LLM explanation)"):
                        with st.spinner("Generating explanation from local LLM..."):
                            xai_text = run_llm_locally(xai_prompt)
                        st.write(xai_text)

            else:
                st.info("Enter the body location and click **Generate first-aid**.")

        # ---------- Mode C: Wound unknown or low confidence ----------
        else:  # llm_mode == "uncertain_wound"
            st.error(
                "The model is **uncertain** about the wound (unknown label or low "
                "confidence). Please describe what you see so the LLM can help."
            )

            user_desc = st.text_area(
                "Short description of the wound and where it is "
                "(e.g., 'small cut on left index finger', 'burn on front of lower leg').",
                height=120,
            ).strip()

            if st.button("Generate first-aid from my description"):
                if not user_desc:
                    st.error("Please provide a brief description first.")
                else:
                    prompt = build_first_aid_prompt_from_user_description(user_desc)
                    with st.expander("Show LLM prompt"):
                        st.text(prompt)

                    with st.spinner("Generating first-aid guidance from local LLM..."):
                        llm_output = run_llm_locally(prompt)

                    st.markdown("**LLM first-aid guidance:**")
                    st.write(llm_output)
            else:
                st.info(
                    "Describe the wound in a few words, then click "
                    "**Generate first-aid from my description**."
                )

else:
    st.info("Please upload an image to begin.")

# ============================================================
# FOOTER
# ============================================================
st.markdown(
    """
    <hr/>
    <div style='text-align:center; color:#6b7280; font-size:0.85rem; padding-top:6px;'>
        <h6> ⚠️ This application is for educational / research purpose only and must not be used as professional medical advice. </h6><br/>
        © 2025 Sumit Barua and Ruth Bahre
    </div>
    """,
    unsafe_allow_html=True,
)

