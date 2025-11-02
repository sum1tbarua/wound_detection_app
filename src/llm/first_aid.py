# src/llm/first_aid.py
"""
LLM-facing utilities for generating first-aid instructions.

- takes structured detections from YOLO (label, confidence, bbox)
- turns them into a prompt for the LLM
- if the detection contains 'wound_unknown', the LLM is asked to
  request more details from the user
- returns Markdown that the UI can render directly
"""


import os
import requests
from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

BASE_PROMPT = """
You are a virtual primary care medical assistant that ONLY gives basic, first-aid style guidance.
You are NOT an actual doctor and you should NOT give diagnosis.
Use the wound detection info from the vision model.
If the detection is 'wound_unknown', ask 1-2 clarifying questions (cause? burn vs cut? location?).
Return Markdown.
"""

def generate_first_aid(detections: list):
    # pick first detection
    det = detections[0] if detections else {"label": "wound_unknown", "confidence": 0.0}

    label = det.get("label", "wound_unknown")
    conf = det.get("confidence", 0.0)

    user_prompt = (
        BASE_PROMPT
        + "\nDetected wound info:\n"
        + f"- label: {label}\n"
        + f"- confidence: {conf}\n"
        + "Now generate the response.\n"
    )

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": user_prompt,
        "stream": False,
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "").strip()
    except Exception as e:
        return f"⚠️ Could not reach local LLM (Ollama). Is it running?\nError: {e}"
