"""
utils/llm_client.py
Unified LLM client — switches between Ollama and Gemini
based on LLM_PROVIDER in .env

Usage:
    from utils.llm_client import get_llm_client
    client = get_llm_client()
    response = client.chat("your prompt")
    data     = client.chat_json("your prompt")

Switch provider by changing .env:
    LLM_PROVIDER=ollama    # free, local, needs Ollama running
    LLM_PROVIDER=gemini    # free tier, fast, needs GEMINI_API_KEY
"""
from __future__ import annotations
from config.settings import get_settings

settings = get_settings()

def get_llm_client():
    """
    Returns the appropriate LLM client based on LLM_PROVIDER in .env.
    Defaults to ollama if not set.
    """
    provider = settings.llm_provider.lower().strip()

    if provider == "gemini":
        from utils.gemini_client import GeminiClient
        return GeminiClient()
    else:
        from utils.ollama_client import OllamaClient
        return OllamaClient()