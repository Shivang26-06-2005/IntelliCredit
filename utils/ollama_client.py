"""
utils/ollama_client.py
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


def _strip_think(text: str) -> str:
    """Remove qwen3 <think> blocks — handles both closed and unclosed/truncated."""
    # Remove complete <think>...</think> block
    text = re.sub(r"<think>[\s\S]*?</think>", "", text)
    # Remove unclosed <think>... (model cut off before writing </think>)
    text = re.sub(r"<think>[\s\S]*", "", text)
    return text.strip()


def _extract_json(text: str) -> str:
    """Extract and repair JSON from model output."""
    text = _strip_think(text)

    # Strip markdown fences
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```\s*$",       "", text, flags=re.MULTILINE)
    text = text.strip()

    # Try to find a complete JSON object or array
    m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    if m:
        return m.group(1)

    # Repair truncated JSON — model was cut off mid-output
    if text.startswith("{") or text.startswith("["):
        text = re.sub(r',\s*"[^"]*$',  "",       text)  # incomplete key
        text = re.sub(r':\s*"[^"]*$',  ": null", text)  # incomplete string value
        text = re.sub(r':\s*[\d.]*$',  ": null", text)  # incomplete number value
        text = re.sub(r',\s*$',        "",       text)  # trailing comma
        text += "]" * max(text.count("[") - text.count("]"), 0)
        text += "}" * max(text.count("{") - text.count("}"), 0)
        return text

    return text


class OllamaClient:

    def __init__(self, base_url=None, model=None):
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model    = (model or settings.ollama_model).strip()

        _timeout = float(settings.ollama_timeout)
        self.client = httpx.Client(
            timeout=httpx.Timeout(connect=30.0, read=_timeout, write=60.0, pool=60.0),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def chat(self, prompt: str, system: str | None = None,
             temperature: float = 0.1, num_predict: int | None = None) -> str:

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model":   self.model,
            "messages": messages,
            "stream":  True,
            "think":   False,   # disables qwen3 chain-of-thought at API level
            "options": {
                "temperature": temperature,
                "num_predict": num_predict or settings.ollama_max_tokens,
                "num_ctx":     4096,
            },
        }

        try:
            chunks = []
            with self.client.stream("POST", f"{self.base_url}/api/chat", json=payload) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        data  = json.loads(line)
                        token = data.get("message", {}).get("content", "")
                        if token:
                            chunks.append(token)
                        if data.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue

            full = "".join(chunks).strip()
            # Belt-and-suspenders: strip think even if API flag was ignored
            return _strip_think(full)

        except httpx.ConnectError:
            logger.error(f"Cannot connect to Ollama at {self.base_url}. Run: ollama serve")
            raise
        except Exception as e:
            logger.error(f"Ollama chat error: {e}")
            raise

    def chat_json(self, prompt: str, system: str | None = None,
                  temperature: float = 0.0, retries: int = 2) -> dict | list:

        suffix = "\n\nReturn ONLY raw JSON. No explanation. No markdown. No backticks."

        for attempt in range(retries + 1):
            raw = self.chat(
                prompt + (suffix if attempt > 0 else ""),
                system=system,
                temperature=temperature,
                num_predict=768,  # extra headroom for JSON calls
            )

            logger.debug(f"chat_json raw [{len(raw)} chars]: {raw[:200]}")

            try:
                return json.loads(_extract_json(raw))
            except json.JSONDecodeError:
                logger.warning(
                    f"JSON parse failed (attempt {attempt+1}) "
                    f"[{len(raw)} chars]: >>>{raw[:300]}<<<"
                )
                if attempt == retries:
                    logger.error("All JSON attempts failed. Returning {}.")
                    return {}

    def is_available(self) -> bool:
        try:
            return self.client.get(f"{self.base_url}/api/tags").status_code == 200
        except Exception:
            return False

    def list_models(self) -> list[str]:
        try:
            return [m["name"] for m in
                    self.client.get(f"{self.base_url}/api/tags").json().get("models", [])]
        except Exception:
            return []

    def close(self):
        try:
            self.client.close()
        except Exception:
            pass


# Singleton
_client: OllamaClient | None = None

def get_ollama_client() -> OllamaClient:
    global _client
    if _client is None:
        _client = OllamaClient()
    return _client
