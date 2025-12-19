import json
import re

# Extracts and parses JSON data from LLM text responses that may contain markdown or extra text
def extract_json_from_text(text):
    # Convert list input to a newline-separated string
    if isinstance(text, list):
        text = "\n".join(map(str, text))

    # Validate that the input is a string 
    if not isinstance(text, str):
        raise TypeError(f"Expected string, got {type(text)}")

    # Check that the input is not empty
    if not text.strip():
        raise ValueError("Empty LLM response")

    # Remove markdown code block markers (```json or ```) from the text
    text = re.sub(r"```json|```", "", text, flags=re.IGNORECASE).strip()

    # Search for JSON object pattern in the text using regex
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in:\n{text}")

    # Parse the matched JSON string and return the parsed object
    return json.loads(match.group(0))