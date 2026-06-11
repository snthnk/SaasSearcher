from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv


class ConfigurationError(RuntimeError):
    """Raised when required environment variables are missing."""


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str
    perplexity_api_key: str
    gemini_model_name: str = "gemini-2.5-flash"
    perplexity_model_name: str = "sonar"
    perplexity_base_url: str = "https://api.perplexity.ai/chat/completions"
    request_timeout_seconds: float = 60.0


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv()

    gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
    perplexity_api_key = os.getenv("PERPLEXITY_API_KEY", "").strip()
    gemini_model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash").strip()
    perplexity_model_name = os.getenv("PERPLEXITY_MODEL_NAME", "sonar").strip()
    timeout_raw = os.getenv("REQUEST_TIMEOUT_SECONDS", "60").strip()

    missing_keys = []
    if not gemini_api_key:
        missing_keys.append("GEMINI_API_KEY")

    if not perplexity_api_key:
        missing_keys.append("PERPLEXITY_API_KEY")

    if missing_keys:
        missing_str = ", ".join(missing_keys)
        raise ConfigurationError(
            f"Не заданы обязательные переменные окружения: {missing_str}. Проверьте файл .env"
        )

    try:
        timeout_seconds = float(timeout_raw)
    except ValueError as error:
        raise ConfigurationError("REQUEST_TIMEOUT_SECONDS должен быть числом.") from error

    return Settings(
        gemini_api_key=gemini_api_key,
        perplexity_api_key=perplexity_api_key,
        gemini_model_name=gemini_model_name,
        perplexity_model_name=perplexity_model_name,
        request_timeout_seconds=timeout_seconds,
    )
