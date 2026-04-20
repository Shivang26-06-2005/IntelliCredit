"""
config/settings.py
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "IntelliCredit"
    app_env: str = "development"
    debug: bool = True
    log_level: str = "INFO"

    # ── LLM Provider Switch ────────────────────────────────────────────────────
    # Set to "ollama" or "gemini" in .env
    llm_provider: str = "ollama"

    # ── Ollama (free, local) ───────────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:8b"
    ollama_max_tokens: int = 512
    ollama_timeout: int = 600

    # ── Gemini (free tier) ─────────────────────────────────────────────────────
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    gemini_max_tokens: int = 4096
    gemini_timeout: int = 60

    # ── News (optional) ────────────────────────────────────────────────────────
    news_api_key: str = ""

    # ── File Storage ───────────────────────────────────────────────────────────
    upload_dir: str = "./uploads"
    reports_dir: str = "./reports"
    max_upload_size_mb: int = 50

    # ── OCR ────────────────────────────────────────────────────────────────────
    tesseract_cmd: str = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

@lru_cache
def get_settings() -> Settings:
    return Settings()