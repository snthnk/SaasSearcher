from __future__ import annotations

import logging
from dataclasses import dataclass

from google import genai
from google.genai import types
from pydantic import ValidationError

from config import get_settings
from schemas import SearchResultReport

LOGGER = logging.getLogger(__name__)

INNOVATOR_PROMPT = (
    "Ты - креативный SaaS-архитектор и продуктовый дизайнер. Твоя задача - изучить "
    "сырые аналитические данные о проблемах пользователей и предложить 3-5 конкретных "
    "концепций SaaS-продуктов. Каждая идея должна быть направлена на устранение реальной "
    "боли, упомянутой в исследовании. Фокусируйся на микро-SaaS решениях (небольших "
    "утилитах или специализированных сервисах), которые можно запустить силами одного разработчика."
)

CRITIC_PROMPT = (
    "Ты - прагматичный венчурный инвестор, технический директор, продуктовый директор с опытом и скептик. Изучи предложенные "
    "концепции SaaS. Найди в них слабые места: высокую техническую сложность, сложность продвижения, низкую готовность "
    "пользователей платить за это решение, сильных конкурентов, юридические и прочие риски. Будь "
    "конструктивен, но строг. Твоя цель - помочь улучшить идеи или отсеять нежизнеспособные."
)

FORMULATOR_PROMPT = (
    "Ты - системный продуктовый аналитик. Твоя задача - объединить первоначальные концепции "
    "Новатора и критические замечания Критика. Доработай каждую концепцию так, чтобы она "
    "учитывала слабые стороны, выявленные Критиком. Подготовь итоговый отчет строго в "
    "соответствии с заданной JSON-схемой. Не добавляй поясняющий текст вне JSON."
)


@dataclass
class StructuredOutputError(RuntimeError):
    message: str
    raw_response: str = ""

    def __str__(self) -> str:
        return self.message


class GeminiOrchestrator:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.client = genai.Client(api_key=self.settings.gemini_api_key)

    def process_search_results(self, niche: str, search_data: str) -> SearchResultReport:
        LOGGER.info("Шаг 1/3: генерация идей")
        ideas = self._generate_text(
            INNOVATOR_PROMPT, self._build_innovator_input(niche, search_data)
        )

        LOGGER.info("Шаг 2/3: критический анализ")
        criticism = self._generate_text(CRITIC_PROMPT, self._build_critic_input(niche, ideas))

        LOGGER.info("Шаг 3/3: формирование структурированного отчета")
        formulator_input = self._build_formulator_input(niche, search_data, ideas, criticism)

        try:
            return self._generate_structured_report(formulator_input)
        except StructuredOutputError:
            LOGGER.warning("Первая попытка структурированного вывода не удалась, выполняется повтор")
            retry_input = formulator_input + (
                "\n\nПовтори ответ строго как JSON, который валидируется по указанной схеме. "
                "Не добавляй markdown, комментарии или блоки кода."
            )
            return self._generate_structured_report(retry_input)

    def _generate_text(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.settings.gemini_model_name,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.4,
            ),
        )
        text = (response.text or "").strip()
        if not text:
            raise RuntimeError("Gemini вернул пустой текстовый ответ.")
        return text

    def _generate_structured_report(self, user_prompt: str) -> SearchResultReport:
        response = self.client.models.generate_content(
            model=self.settings.gemini_model_name,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=FORMULATOR_PROMPT,
                temperature=0.2,
                response_mime_type="application/json",
                response_schema=SearchResultReport,
            ),
        )
        raw_text = (response.text or "").strip()

        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, SearchResultReport):
            return parsed
        if isinstance(parsed, dict):
            return SearchResultReport.model_validate(parsed)

        if not raw_text:
            raise StructuredOutputError("Gemini не вернул JSON-ответ.", raw_response="")

        try:
            return SearchResultReport.model_validate_json(raw_text)
        except ValidationError as error:
            raise StructuredOutputError(
                f"Gemini вернул JSON, который не прошел валидацию: {error}",
                raw_response=raw_text,
            ) from error

    @staticmethod
    def _build_innovator_input(niche: str, search_data: str) -> str:
        return (
            f"Ниша: {niche}\n\n"
            "Ниже приведен сырой аналитический отчет по жалобам пользователей и рыночным сигналам. "
            "Сформируй 3-5 микро-SaaS концепций. Для каждой укажи: название, аудиторию, какую боль "
            "закрывает, ключевой функционал, конкурентов, модель монетизации и риски.\n\n"
            f"Отчет:\n{search_data}"
        )

    @staticmethod
    def _build_critic_input(niche: str, ideas: str) -> str:
        return (
            f"Ниша: {niche}\n\n"
            "Ниже список предложенных SaaS-концепций. Оцени каждую идею критически: готовность рынка "
            "платить, сложность разработки, конкуренцию, риски внедрения, правовые и интеграционные "
            "проблемы. Предложи, как улучшить идеи.\n\n"
            f"Концепции:\n{ideas}"
        )

    @staticmethod
    def _build_formulator_input(
        niche: str,
        search_data: str,
        ideas: str,
        criticism: str,
    ) -> str:
        return (
            f"Ниша: {niche}\n\n"
            "Собери финальный отчет в JSON по предоставленной схеме. Верни доработанные концепции, "
            "которые учитывают критику.\n\n"
            f"Сырой отчет исследования:\n{search_data}\n\n"
            f"Идеи новатора:\n{ideas}\n\n"
            f"Критика:\n{criticism}"
        )
