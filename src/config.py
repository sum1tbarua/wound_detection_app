# src/config.py
"""
Central configuration for the project.

- stores paths to trained YOLO model(s)
- holds detection thresholds
- optionally loads environment variables (.env) for LLM provider
- keeps model/LLM settings in one place so the rest of the code stays clean
"""


# --- YOLO configuration ---
# Points to the trained segmentation model in the models/ folder.
YOLO_MODEL_PATH = "models/wound_yolo_seg_final.pt"


# Minimum confidence threshold for displaying detections
DETECTION_CONF_THRESHOLD = 0.35


# --- Local LLM (Ollama) configuration ---
from dotenv import load_dotenv
import os
from pathlib import Path

# Load environment variables from the .env file at project root
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

# Default to local Ollama model (Mistral)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

# Optional info log
print(f"✅ Using local Ollama model: {OLLAMA_MODEL}")

