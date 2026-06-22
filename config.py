from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv


class ConfigurationError(RuntimeError):
    """Raised when required environment variables are missing."""


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    perplexity_api_key: str
    gemini_api_key: str = ""
    mistral_api_key: str = ""
    yandex_api_key: str = ""
    yandex_folder_id: str = ""
    gemini_model_name: str = "gemini-2.0-flash"
    mistral_model_name: str = "mistral-large-latest"
    mistral_base_url: str = "https://api.mistral.ai/v1/chat/completions"
    llm_request_timeout_seconds: float = 180.0
    llm_retry_attempts: int = 6
    llm_retry_initial_delay_seconds: float = 10.0
    llm_retry_max_delay_seconds: float = 90.0
    perplexity_model_name: str = "sonar"
    perplexity_base_url: str = "https://api.perplexity.ai/chat/completions"
    request_timeout_seconds: float = 60.0


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv()

    llm_provider = os.getenv("LLM_PROVIDER", "mistral").strip().lower()
    gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
    mistral_api_key = os.getenv("MISTRAL_API_KEY", "").strip()
    perplexity_api_key = os.getenv("PERPLEXITY_API_KEY", "").strip()
    yandex_api_key = os.getenv("YANDEX_API_KEY", "").strip()
    yandex_folder_id = os.getenv("YANDEX_FOLDER_ID", "").strip()
    gemini_model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash").strip()
    mistral_model_name = os.getenv("MISTRAL_MODEL_NAME", "mistral-large-latest").strip()
    mistral_base_url = os.getenv("MISTRAL_BASE_URL", "https://api.mistral.ai/v1/chat/completions").strip()
    llm_timeout_raw = os.getenv(
        "LLM_REQUEST_TIMEOUT_SECONDS",
        os.getenv("GEMINI_REQUEST_TIMEOUT_SECONDS", "180"),
    ).strip()
    llm_retry_attempts_raw = os.getenv(
        "LLM_RETRY_ATTEMPTS",
        os.getenv("GEMINI_RETRY_ATTEMPTS", "6"),
    ).strip()
    llm_retry_initial_delay_raw = os.getenv(
        "LLM_RETRY_INITIAL_DELAY_SECONDS",
        os.getenv("GEMINI_RETRY_INITIAL_DELAY_SECONDS", "10"),
    ).strip()
    llm_retry_max_delay_raw = os.getenv(
        "LLM_RETRY_MAX_DELAY_SECONDS",
        os.getenv("GEMINI_RETRY_MAX_DELAY_SECONDS", "90"),
    ).strip()
    perplexity_model_name = os.getenv("PERPLEXITY_MODEL_NAME", "sonar").strip()
    timeout_raw = os.getenv("REQUEST_TIMEOUT_SECONDS", "60").strip()

    if gemini_api_key == "your_gemini_api_key_here":
        gemini_api_key = ""
    if mistral_api_key == "your_mistral_api_key_here":
        mistral_api_key = ""
    if yandex_api_key == "your_yandex_api_key":
        yandex_api_key = ""
    if yandex_folder_id == "your_yandex_folder_id":
        yandex_folder_id = ""

    if llm_provider not in {"gemini", "mistral"}:
        raise ConfigurationError("LLM_PROVIDER должен быть одним из: gemini, mistral.")

    missing_keys = []
    if not perplexity_api_key:
        missing_keys.append("PERPLEXITY_API_KEY")
    if llm_provider == "gemini" and not gemini_api_key:
        missing_keys.append("GEMINI_API_KEY")
    if llm_provider == "mistral" and not mistral_api_key:
        missing_keys.append("MISTRAL_API_KEY")

    if missing_keys:
        missing_str = ", ".join(missing_keys)
        raise ConfigurationError(
            f"Не заданы обязательные переменные окружения: {missing_str}. Проверьте файл .env"
        )

    try:
        timeout_seconds = float(timeout_raw)
        llm_request_timeout_seconds = float(llm_timeout_raw)
        llm_retry_attempts = int(llm_retry_attempts_raw)
        llm_retry_initial_delay_seconds = float(llm_retry_initial_delay_raw)
        llm_retry_max_delay_seconds = float(llm_retry_max_delay_raw)
    except ValueError as error:
        raise ConfigurationError(
            "Проверьте числовые значения REQUEST_TIMEOUT_SECONDS и LLM_*_SECONDS / LLM_RETRY_ATTEMPTS."
        ) from error

    if llm_retry_attempts < 1:
        raise ConfigurationError("LLM_RETRY_ATTEMPTS должен быть >= 1.")
    if llm_retry_initial_delay_seconds <= 0 or llm_retry_max_delay_seconds <= 0:
        raise ConfigurationError("LLM_RETRY_*_SECONDS должны быть > 0.")

    return Settings(
        llm_provider=llm_provider,
        perplexity_api_key=perplexity_api_key,
        gemini_api_key=gemini_api_key,
        mistral_api_key=mistral_api_key,
        yandex_api_key=yandex_api_key,
        yandex_folder_id=yandex_folder_id,
        gemini_model_name=gemini_model_name,
        mistral_model_name=mistral_model_name,
        mistral_base_url=mistral_base_url,
        llm_request_timeout_seconds=llm_request_timeout_seconds,
        llm_retry_attempts=llm_retry_attempts,
        llm_retry_initial_delay_seconds=llm_retry_initial_delay_seconds,
        llm_retry_max_delay_seconds=llm_retry_max_delay_seconds,
        perplexity_model_name=perplexity_model_name,
        request_timeout_seconds=timeout_seconds,
    )
