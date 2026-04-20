"""
utils/gemini_client.py
Gemini API client (free tier available)
Get API key at: https://aistudio.google.com/apikey
Free: gemini-2.0-flash — 15 req/min, 1500 req/day
"""
from __future__ import annotations
import json
import re
from typing import Any
import httpx
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential
from config.settings import get_settings

settings = get_settings()

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


class GeminiClient:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or settings.gemini_api_key
        self.model   = model or settings.gemini_model
        self.timeout = settings.gemini_timeout

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def chat(self, prompt: str, system: str | None = None, temperature: float = 0.1) -> str:
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY not set in .env. "
                "Get a free key at https://aistudio.google.com/apikey"
            )
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": settings.gemini_max_tokens,
            }
        }
        if system:
            payload["system_instruction"] = {"parts": [{"text": system}]}

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(
                    GEMINI_URL.format(model=self.model),
                    json=payload,
                    params={"key": self.api_key},
                    headers={"Content-Type": "application/json"},
                )
                resp.raise_for_status()
                data = resp.json()
                return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except httpx.HTTPStatusError as e:
            logger.error(f"Gemini API error {e.response.status_code}: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Gemini chat error: {e}")
            raise

    def chat_json(self, prompt: str, system: str | None = None,
                  temperature: float = 0.0, retries: int = 2) -> dict[str, Any] | list:
        strict = "\n\nReturn ONLY raw JSON. No explanation. No markdown. No backticks."
        for attempt in range(retries + 1):
            raw = self.chat(prompt + (strict if attempt > 0 else ""), system=system, temperature=temperature)
            try:
                return json.loads(_strip_fences(raw))
            except json.JSONDecodeError:
                logger.warning(f"Gemini JSON parse failed attempt {attempt+1}: {raw[:200]}")
                if attempt == retries:
                    logger.error("All Gemini JSON attempts failed.")
                    return {}

    def is_available(self) -> bool:
        return bool(self.api_key)


def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    return match.group(1) if match else text.strip()