import json
import re

def extract_json_from_text(text):
    if isinstance(text, list):
        text = "\n".join(map(str, text))

    if not isinstance(text, str):
        raise TypeError(f"Expected string, got {type(text)}")

    if not text.strip():
        raise ValueError("Empty LLM response")

    text = re.sub(r"```json|```", "", text, flags=re.IGNORECASE).strip()

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in:\n{text}")

    return json.loads(match.group(0))