from __future__ import annotations

import logging
from typing import Any

import httpx

from config import get_settings

LOGGER = logging.getLogger(__name__)

US_SEARCH_TEMPLATE = (
    "Conduct a detailed real-time research on the niche: '{niche}'. "
    "Identify 3-5 emerging or trending micro-SaaS products, tools, or utilities recently launched in the US market "
    "(check Product Hunt, HackerNews, Y Combinator, and indie forums). "
    "For these products, find recent (last 3-6 months) user complaints, unresolved pain points, and bugs discussed "
    "on Reddit, Twitter/X, and G2/Capterra. "
    "Structure your response in English with clear sections: "
    "1. Emerging US Products (names and concepts), "
    "2. Key User Pain Points and Frustrations, "
    "3. Missing Features or Gaps."
)

RU_SEARCH_TEMPLATE = (
    "Проведи исследование российского рынка на тему: '{niche}'.\n\n"
    "КОНТЕКСТ ИЗ США:\n"
    "В США в этой нише популярны следующие тренды и продукты со своими болями:\n"
    "{us_trends}\n\n"
    "СПРОС В РФ (ДАННЫЕ WORDSTAT):\n"
    "{wordstat_data}\n\n"
    "ТВОЯ ЗАДАЧА:\n"
    "Выясни, существуют ли в Рунете (проверь VC.ru, Habr, Яндекс, профильные каналы) "
    "прямые аналоги или клоны указанных американских продуктов, либо другие локальные решения в этой нише. "
    "Опиши текущую конкурентную ситуацию в РФ: "
    "занята ли ниша, какие локальные альтернативы предлагают российские разработчики, "
    "и с какими специфическими проблемами сталкиваются российские пользователи (например, проблемы с оплатой "
    "зарубежных сервисов, отсутствие локализации и т.д.).\n\n"
    "Ответ структурируй в виде аналитической заметки на русском языке."
)


class PerplexityClient:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._client = httpx.Client(timeout=self.settings.request_timeout_seconds)

    def _execute_query(self, system_prompt: str, user_prompt: str) -> str:
        """Базовый метод выполнения запроса к Perplexity API."""
        payload = {
            "model": self.settings.perplexity_model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
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

        return content

    def search_us_trends(self, niche: str) -> str:
        LOGGER.info("Шаг 2: Поиск американских трендов в Perplexity")
        system_prompt = (
            "You are a market analyst and user researcher. "
            "You conduct precise real-time web searches and return factual, structured reports."
        )
        user_prompt = US_SEARCH_TEMPLATE.format(niche=niche)
        return self._execute_query(system_prompt, user_prompt)

    def search_ru_alternatives(self, niche: str, wordstat_data: str, us_trends: str) -> str:
        LOGGER.info("Шаг 3: Поиск российских аналогов и локальной специфики в Perplexity")
        system_prompt = (
            "Ты — аналитик российского рынка IT-решений. Ты исследуешь уровень конкуренции "
            "в Рунете и сопоставляешь отечественные аналоги с зарубежными трендами."
        )
        user_prompt = RU_SEARCH_TEMPLATE.format(
            niche=niche,
            us_trends=us_trends,
            wordstat_data=wordstat_data,
        )
        return self._execute_query(system_prompt, user_prompt)
