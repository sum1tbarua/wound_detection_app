# src/prompts.py

"""
Prompt templates and helper utilities for the LLM.

- builds the JSON/Markdown style prompt from detection output
- keeps wording for “ask for clarification” separate from app logic
- makes it easy to switch between OpenAI/Ollama/stub LLMs
"""


BASE_SYSTEM_PROMPT = """Think youself as a visrtual Primary-care physician who provides first-aid instructions.
If the wound type or body location is unknown, you MUST first ask the user for clarification
instead of guessing. Be safe, concise, Markdown only.
"""
def build_user_prompt(detections: list) -> str:
    has_unknown = any(d["label"] == "wound_unknown" for d in detections)

    if has_unknown:
        # user-interactive mode
        return f"""
The vision model could not confidently identify the wound.

Detections:
{detections}

Ask the user a SHORT clarification question in Markdown, for example:
- "Is this a cut, a burn, or something else?"
- "Which body part is injured (leg, hand, arm)?"

Do NOT generate first-aid steps yet.
"""
    else:
        # normal mode
        return f"""
Generate first-aid instructions for the following detections.
Return Markdown with sections: ### Assessment, ### First-Aid Steps, ### When to seek care.

Detections:
{detections}
"""

"""
IMPORTANT:
- You are NOT a doctor.
- You must tell the user to seek emergency care if the wound looks severe.
- Output must be in Markdown.
- Be concise and step-by-step.
"""
