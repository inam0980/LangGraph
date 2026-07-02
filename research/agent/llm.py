"""Google Gemini integration.

We keep all LLM creation in one place so the rest of the code never has to
know *which* model we use or how it's configured. If you want to swap models
or providers later, you only change this file.
"""
from __future__ import annotations

import json
import os
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI


# Tried in order: if one model is rate-limited or down, the next takes over.
# The GEMINI_MODEL env var (if set) is always tried first.
_FALLBACK_MODELS = [
    "gemini-3.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
]


def _candidate_models() -> list[str]:
    """Return the ordered list of models to try (env override first)."""
    primary = os.getenv("GEMINI_MODEL", "").strip()
    models = [primary] if primary else []
    for name in _FALLBACK_MODELS:
        if name not in models:
            models.append(name)
    return models


def get_llm(temperature: float = 0.2, model: str | None = None) -> ChatGoogleGenerativeAI:
    """Create and return a configured Gemini chat model.

    Args:
        temperature: Creativity of the model. Lower = more deterministic,
            which is what we want for financial analysis.
        model: Model name to use. Defaults to GEMINI_MODEL from the
            environment (or the first fallback model).

    Returns:
        A ready-to-use ``ChatGoogleGenerativeAI`` instance.

    Raises:
        RuntimeError: If the GOOGLE_API_KEY environment variable is missing.
    """
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Copy .env.example to .env and add "
            "your Gemini API key from https://aistudio.google.com/app/apikey"
        )

    return ChatGoogleGenerativeAI(
        model=model or _candidate_models()[0],
        google_api_key=api_key,
        temperature=temperature,
    )


def ask_json(system_prompt: str, user_prompt: str, temperature: float = 0.2) -> dict[str, Any]:
    """Ask Gemini a question and parse the reply as JSON.

    Many of our nodes need *structured* answers (scores, lists, labels). The
    simplest robust pattern is: tell the model to reply with JSON only, then
    parse it. This helper centralizes that logic plus the error handling.

    Args:
        system_prompt: Instructions describing the role and the exact JSON
            shape we expect back.
        user_prompt: The actual data/question for this call.
        temperature: Passed through to the model.

    Returns:
        The parsed JSON object as a Python dict. If parsing fails, returns a
        dict with an "error" key so callers can degrade gracefully.
    """
    import time

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    # Reliability strategy: walk the model fallback chain. A rate-limited
    # (429) or overloaded (503) model is skipped in favor of the next one,
    # since each model has its own quota pool. Transient errors get one
    # short retry on the same model first.
    last_error = ""
    for model_name in _candidate_models():
        llm = get_llm(temperature=temperature, model=model_name)
        for attempt in range(2):
            raw = ""
            try:
                response = llm.invoke(messages)
                raw = _extract_text(response.content).strip()
                cleaned = _strip_code_fences(raw)
                return json.loads(cleaned)
            except json.JSONDecodeError:
                print(f"[llm] {model_name}: JSON parse failed. First 300 chars: {raw[:300]}", flush=True)
                return {"error": "Could not parse model response as JSON", "raw": raw}
            except Exception as exc:  # noqa: BLE001 - we want any failure to be visible but non-fatal
                last_error = str(exc)
                print(f"[llm] {model_name} failed (attempt {attempt + 1}/2): {exc}", flush=True)
                # Permanent failures (bad/blocked/leaked key) won't fix themselves.
                if "PERMISSION_DENIED" in last_error or "API_KEY_INVALID" in last_error:
                    return {"error": f"LLM call failed: {last_error}"}
                # Rate limit / quota: this model is exhausted for now — move
                # to the next model immediately instead of waiting it out.
                if "429" in last_error or "RESOURCE_EXHAUSTED" in last_error or "quota" in last_error.lower():
                    break
                if attempt == 0:
                    time.sleep(3)
        print(f"[llm] switching to next fallback model...", flush=True)

    return {"error": f"LLM call failed: {last_error}"}


def _extract_text(content: Any) -> str:
    """Get plain text from a model response's ``content``.

    Older models return a plain string, but newer Gemini models return a
    *list of content blocks* like ``[{"type": "text", "text": "..."}]``.
    This helper normalizes both shapes into a single string.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content)


def _strip_code_fences(text: str) -> str:
    """Extract JSON from model response — handles code fences and leading text."""
    import re
    # Handle ```json ... ``` or ``` ... ``` anywhere in the response
    match = re.search(r"```(?:json)?\s*\n([\s\S]*?)\n```", text)
    if match:
        return match.group(1).strip()
    # If the text starts directly with a fence
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    # Try to extract a JSON object if there's leading text
    brace = text.find("{")
    if brace > 0:
        return text[brace:]
    return text
