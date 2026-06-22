from __future__ import annotations

import logging
from typing import Any

import httpx

from config import get_settings

LOGGER = logging.getLogger(__name__)


class WordstatClient:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._client = httpx.Client(timeout=self.settings.request_timeout_seconds)
        self._url = "https://searchapi.api.cloud.yandex.net/v2/wordstat/topRequests"

    def get_top_phrases(self, phrase: str, num_phrases: int = 20) -> str:
        """
        Запрашивает топ похожих запросов с частотностью за последние 30 дней.
        Возвращает форматированную текстовую сводку.
        """
        LOGGER.info("Запрос к Yandex Wordstat API запущен")

        if not self.settings.yandex_api_key or not self.settings.yandex_folder_id:
            LOGGER.warning("Yandex API Key или Folder ID не настроены. Пропускаем Wordstat.")
            return "Статистика Wordstat недоступна (не настроены ключи API)."

        payload = {
            "phrase": phrase,
            "numPhrases": num_phrases,
            "regions": ["225"],
            "devices": ["DEVICE_ALL"],
            "folderId": self.settings.yandex_folder_id,
        }
        headers = {
            "Authorization": f"Api-key {self.settings.yandex_api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = self._client.post(self._url, json=payload, headers=headers)
            response.raise_for_status()
        except httpx.HTTPStatusError:
            LOGGER.error(
                "Wordstat API вернул ошибку %s: %s",
                response.status_code,
                response.text,
            )
            return f"Ошибка при запросе к Wordstat API: {response.text}"
        except Exception as error:  # noqa: BLE001
            LOGGER.error("Непредвиденная ошибка Wordstat API: %s", error)
            return f"Не удалось собрать статистику Wordstat: {error}"

        data: dict[str, Any] = response.json()
        top_phrases: list[dict[str, Any]] = data.get("topPhrases", [])
        if not top_phrases:
            return f"По запросу '{phrase}' в Wordstat ничего не найдено."

        lines = [f"Статистика поисковых запросов в Яндексе по теме '{phrase}' за 30 дней:"]
        for idx, item in enumerate(top_phrases, start=1):
            text = item.get("phrase", "")
            count = item.get("showCount", "0")
            lines.append(f"{idx}. '{text}' — {count} запросов в месяц")

        LOGGER.info("Статистика от Wordstat успешно получена")
        return "\n".join(lines)
