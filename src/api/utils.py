#!/usr/bin/env python3
"""
WavBench API Utilities

Common utilities for calling LLM APIs for evaluation.
Uses Google genai SDK for Gemini models.
"""

import json
import time
import os
from pathlib import Path
from typing import Tuple, Any, Optional

from google import genai
from google.genai import types


# Global client instance (initialized lazily)
_client: Optional[genai.Client] = None


def get_client(api_key: Optional[str] = None) -> genai.Client:
    """Get or create the genai client."""
    global _client
    if _client is None:
        # API key can be passed directly or via environment variable GOOGLE_API_KEY
        if api_key:
            _client = genai.Client(api_key=api_key)
        else:
            _client = genai.Client()
    return _client


def safe_float(x: Any) -> Optional[float]:
    """Safely convert value to float."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def call_llm_api(
    prompt: str,
    model: str,
    api_key: Optional[str] = None,
    audio_path: Optional[Path] = None,
    max_retry: int = 3,
    sleep_between_retry: int = 2,
    temperature: float = 0.0,
    max_tokens: int = 512,
    **kwargs,  # Accept but ignore legacy parameters like api_base, api_format
) -> Tuple[bool, str, str]:
    """
    Call Gemini API using Google genai SDK.

    Args:
        prompt: Text prompt
        model: Model name (e.g., gemini-3-pro-preview)
        api_key: Optional API key (can also use GOOGLE_API_KEY env var)
        audio_path: Optional path to audio file for multimodal input
        max_retry: Maximum retry attempts
        sleep_between_retry: Sleep time between retries
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate

    Returns:
        Tuple of (success, response_text, error_message)
    """
    client = get_client(api_key)

    last_error = ""
    uploaded_file = None

    for attempt in range(1, max_retry + 1):
        try:
            # Build contents
            contents = []

            # Add text prompt
            contents.append(prompt)

            # Upload and add audio if provided
            if audio_path and audio_path.exists():
                if uploaded_file is None:
                    uploaded_file = client.files.upload(file=str(audio_path))
                contents.append(uploaded_file)

            # Configure generation
            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )

            # Call API
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )

            # Extract response text
            if response.text:
                return True, response.text.strip(), ""
            else:
                last_error = "Empty response from API"

        except Exception as e:
            last_error = str(e)

        if attempt < max_retry:
            time.sleep(sleep_between_retry)

    return False, "", last_error


def parse_json_response(response_text: str) -> dict:
    """
    Parse JSON from LLM response, handling markdown code blocks.

    Args:
        response_text: Raw response text from LLM

    Returns:
        Parsed JSON dict or empty dict on failure
    """
    text = response_text.strip()

    # Handle markdown code blocks
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines (code block markers)
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    return {}
