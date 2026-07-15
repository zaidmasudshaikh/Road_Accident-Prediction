"""
ai_explanation.py
Optional AI-powered explanation feature.

IMPORTANT: This module does NOT perform accident severity prediction.
The locally trained ML model (src/train_model.py, src/evaluate_model.py)
performs the actual prediction. This module only takes an already-made
prediction and the input factors, and asks an AI API to generate a
human-readable explanation of possible contributing risk factors plus
general road safety recommendations.

If the API key is missing or the request fails, generate_explanation()
returns a result dict with success=False and a short reason, so the
calling app can display a graceful message and keep working normally.
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"  # small/fast model, sufficient for a short explanation
REQUEST_TIMEOUT_SECONDS = 15


def _build_prompt(input_factors: dict, predicted_severity: str) -> str:
    factors_text = "\n".join(f"- {key.replace('_', ' ')}: {value}" for key, value in input_factors.items())
    return (
        "A machine learning model has already predicted the following road accident "
        f"severity: {predicted_severity}.\n\n"
        "The prediction was based on these factors:\n"
        f"{factors_text}\n\n"
        "Do NOT re-predict or question the severity classification. Instead, in under "
        "150 words, write:\n"
        "1. A short, plain-language explanation of which of these factors likely "
        "contributed most to this severity level.\n"
        "2. 2-3 brief, general road safety recommendations relevant to these factors.\n"
        "Keep the tone factual and non-alarming."
    )


def generate_explanation(input_factors: dict, predicted_severity: str) -> dict:
    """
    Returns:
        {"success": True, "text": "..."} on success
        {"success": False, "reason": "..."} if unavailable (missing key, API error, etc.)
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return {"success": False, "reason": "No API key configured (ANTHROPIC_API_KEY missing in .env)."}

    prompt = _build_prompt(input_factors, predicted_severity)

    try:
        response = requests.post(
            ANTHROPIC_API_URL,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": ANTHROPIC_MODEL,
                "max_tokens": 300,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()
        text_blocks = [block["text"] for block in data.get("content", []) if block.get("type") == "text"]
        explanation_text = "\n".join(text_blocks).strip()

        if not explanation_text:
            return {"success": False, "reason": "AI API returned an empty response."}

        return {"success": True, "text": explanation_text}

    except requests.exceptions.Timeout:
        return {"success": False, "reason": "AI API request timed out."}
    except requests.exceptions.RequestException as e:
        return {"success": False, "reason": f"AI API request failed: {e}"}
    except (KeyError, ValueError) as e:
        return {"success": False, "reason": f"Unexpected AI API response format: {e}"}


if __name__ == "__main__":
    # Quick manual test
    sample_factors = {
        "Age_band_of_driver": "18-30",
        "Weather_conditions": "Raining",
        "Light_conditions": "Darkness - no lighting",
        "Cause_of_accident": "Overspeed",
    }
    result = generate_explanation(sample_factors, "Serious Injury")
    print(result)
