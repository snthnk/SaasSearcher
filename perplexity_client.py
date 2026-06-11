from __future__ import annotations

import logging
from typing import Any

import httpx

from config import get_settings

LOGGER = logging.getLogger(__name__)

SEARCH_PROMPT_TEMPLATE = (
    "Проведи детальное исследование в реальном времени по теме: '{niche}'. "
    "Найди недавние (за последние 3-6 месяцев) жалобы пользователей, нерешенные проблемы, "
    "баги или отсутствующие функции в существующих решениях на площадках Reddit, Twitter/X, "
    "Quora и профильных форумах. Опиши конкретные ситуации, где люди выражают недовольство "
    "текущими инструментами или ручными процессами. По возможности укажи ссылки на источники "
    "или названия обсуждаемых продуктов. Ответ структурируй в виде аналитической заметки с "
    "четкими пунктами: боли, контекст, повторяющиеся паттерны, упомянутые продукты и выводы."
)


class PerplexityClient:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._client = httpx.Client(timeout=self.settings.request_timeout_seconds)

    def search_pain_points(self, niche: str) -> str:
        LOGGER.info("Запрос к Perplexity запущен")
        payload = {
            "model": self.settings.perplexity_model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Ты аналитик рынка и исследователь пользовательских болей. "
                        "Проводишь актуальный веб-поиск и возвращаешь фактический отчет."
                    ),
                },
                {
                    "role": "user",
                    "content": SEARCH_PROMPT_TEMPLATE.format(niche=niche),
                },
            ],
            "temperature": 0.2,
        }
        headers = {
            "Authorization": f"Bearer {self.settings.perplexity_api_key}",
            "Content-Type": "application/json",
        }

        response = self._client.post(
            self.settings.perplexity_base_url,
            headers=headers,
            json=payload,
        )

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            raise RuntimeError(
                f"Perplexity API вернул ошибку {response.status_code}: {response.text}"
            ) from error

        data: dict[str, Any] = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("Perplexity API не вернул choices в ответе.")

        message = choices[0].get("message") or {}
        content = message.get("content", "").strip()
        if not content:
            raise RuntimeError("Perplexity API вернул пустой ответ.")

        citations = data.get("citations") or []
        if citations:
            content += "\n\nИсточники:\n" + "\n".join(f"- {item}" for item in citations)

        LOGGER.info("Ответ от Perplexity успешно получен")
        return content
