# src/llm/first_aid.py
"""
LLM-facing utilities for generating first-aid instructions.

- Consumes structured detections from YOLO (label, confidence, bbox)
- If 'user_clarification' is present, produce final first-aid instructions
- If any detection is 'wound_unknown' and no user clarification yet,
  ask 1–2 concise clarifying questions
- Otherwise (confident case), generate first-aid directly
- Always returns Markdown for Streamlit to render
"""

import os
import requests
from textwrap import dedent
from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")


# ---------- Prompt Builders ----------

def _summarize_detections(detections: list, max_items: int = 3) -> str:
    """Compact bullet list summary of detections for the prompt."""
    if not detections:
        return "- (none)"
    lines = []
    count = 0
    for d in detections:
        if d.get("label") == "user_clarification":
            # leave this out of the auto summary; handled separately
            continue
        lbl = d.get("label", "unknown")
        conf = d.get("confidence", 0)
        # bbox is optional, keeps prompt clean
        lines.append(f"- {lbl} (conf={conf:.2f})")
        count += 1
        if count >= max_items:
            break
    return "\n".join(lines) if lines else "- (none)"

def _build_prompt_clarify(detections: list) -> str:
    """Ask for short clarification while keeping safety and scope."""
    det_summary = _summarize_detections(detections)
    return dedent(f"""
    You are a virtual primary-care assistant that ONLY gives basic, first-aid style guidance.
    You are NOT a doctor and should NOT provide diagnoses. Keep a helpful, calm tone.

    The vision system's current summary:
    {det_summary}

    The wound type/body location is unclear. Ask the user 1–2 short clarifying questions
    (e.g., “Is it a cut or a burn?” and “Where on the body is it?”).
    Respond in **Markdown**, and DO NOT provide instructions yet—just ask the questions.
    """).strip()

def _build_prompt_final(detections: list, user_note: str | None) -> str:
    """Generate first-aid instructions with optional user clarification."""
    det_summary = _summarize_detections(detections)
    user_line = f"\nUser clarification: {user_note}" if user_note else ""
    return dedent(f"""
    You are a virtual primary-care assistant that ONLY gives basic, first-aid style guidance.
    You are NOT a doctor and should NOT provide diagnoses. Keep a helpful, calm tone.

    Vision summary:
    {det_summary}{user_line}

    Task:
    - Identify the likely wound type and body location from the info above.
    - Provide concise, step-by-step first-aid instructions (bulleted or numbered).
    - Include a short **When to seek medical help** section with 3–5 bullet points.
    - Avoid brand names and advanced treatments.
    - Keep it brief and actionable.

    Return your answer in **Markdown**.
    """).strip()


# ---------- LLM Call ----------

def _call_ollama_markdown(prompt: str, model: str | None = None, timeout: int = 60) -> str:
    """Call Ollama and return the 'response' text or a helpful error."""
    payload = {
        "model": model or OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        text = (data.get("response") or "").strip()
        return text if text else "⚠️ LLM returned an empty response."
    except Exception as e:
        return f"⚠️ Could not reach local LLM (Ollama). Is it running?\n\n`{e}`"


# ---------- Public API ----------

def generate_first_aid(detections: list) -> str:
    """
    Decide whether to ask for clarification or produce final instructions,
    and return Markdown.
    """
    detections = detections or [{"label": "wound_unknown", "confidence": 0.0}]

    # Extract optional user clarification
    user_note = None
    has_unknown = False
    filtered = []
    for d in detections:
        lbl = d.get("label", "wound_unknown")
        if lbl == "user_clarification":
            user_note = (d.get("text") or "").strip() or None
        elif lbl == "wound_unknown":
            has_unknown = True
            filtered.append(d)
        else:
            filtered.append(d)

    # If the user clarified, always produce final instructions
    if user_note:
        prompt = _build_prompt_final(filtered, user_note)
        return _call_ollama_markdown(prompt)

    # If anything is unknown and no clarification yet → ask questions
    if has_unknown:
        prompt = _build_prompt_clarify(filtered)
        return _call_ollama_markdown(prompt)

    # Confident path
    prompt = _build_prompt_final(filtered, user_note=None)
    return _call_ollama_markdown(prompt)
