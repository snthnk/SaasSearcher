from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from google import genai
from google.genai import errors as genai_errors
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
    "Новатора и замечания Критика, создав новые версии концепций.\n"
    "Выводи результат строго в соответствии с предоставленной целевой JSON-схемой. "
    "Не копируй структуру исходных документов. Ответ должен содержать только поля из схемы "
    "('niche' и список 'concepts' с внутренними полями 'title', 'target_audience', 'source_pains', "
    "'solution_description', 'competitors', 'technical_difficulty', 'monetization_model', 'criticism_addressed').\n"
    "Не добавляй поясняющий текст или разметку markdown вне JSON-объекта."
)


@dataclass
class StructuredOutputError(RuntimeError):
    message: str
    raw_response: str = ""

    def __str__(self) -> str:
        return self.message


@dataclass
class OrchestrationResult:
    report: SearchResultReport
    provider: str
    model_name: str
    innovator_input: str
    innovator_output: str
    critic_input: str
    critic_output: str
    formulator_input: str
    final_raw_response: str


class GeminiOrchestrator:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.provider = self.settings.llm_provider
        self.gemini_client: genai.Client | None = None
        self.mistral_client: httpx.Client | None = None

        if self.provider == "gemini":
            gemini_timeout_ms = max(int(self.settings.llm_request_timeout_seconds * 1000), 10_000)
            self.gemini_client = genai.Client(
                api_key=self.settings.gemini_api_key,
                http_options=types.HttpOptions(
                    timeout=gemini_timeout_ms,
                    retry_options=types.HttpRetryOptions(
                        attempts=self.settings.llm_retry_attempts,
                        initial_delay=self.settings.llm_retry_initial_delay_seconds,
                        max_delay=self.settings.llm_retry_max_delay_seconds,
                        exp_base=2.0,
                        jitter=1.0,
                        http_status_codes=[408, 429, 500, 502, 503, 504],
                    ),
                ),
            )
        else:
            self.mistral_client = httpx.Client(timeout=self.settings.llm_request_timeout_seconds)

    def process_search_results(self, niche: str, search_data: str) -> SearchResultReport:
        return self.process_search_results_with_trace(niche=niche, search_data=search_data).report

    def process_search_results_with_trace(
        self,
        niche: str,
        search_data: str,
        output_dir: Path | None = None,
    ) -> OrchestrationResult:
        LOGGER.info("LLM-провайдер: %s, модель: %s", self.provider, self._get_active_model_name())

        LOGGER.info("Шаг 1/3: генерация идей")
        innovator_input = self._build_innovator_input(niche, search_data)
        self._write_artifact(output_dir, "05_innovator_input.txt", innovator_input)
        ideas = self._generate_text(
            INNOVATOR_PROMPT,
            innovator_input,
            temperature=0.7,
            operation_name="Шаг новатора",
        )
        self._write_artifact(output_dir, "06_innovator_output.txt", ideas)

        LOGGER.info("Шаг 2/3: критический анализ")
        critic_input = self._build_critic_input(niche, ideas)
        self._write_artifact(output_dir, "07_critic_input.txt", critic_input)
        criticism = self._generate_text(
            CRITIC_PROMPT,
            critic_input,
            operation_name="Шаг критика",
        )
        self._write_artifact(output_dir, "08_critic_output.txt", criticism)

        LOGGER.info("Шаг 3/3: формирование структурированного отчета")
        
        # Получаем строковое представление схемы для промпта
        schema_json = json.dumps(SearchResultReport.model_json_schema(), ensure_ascii=False, indent=2)
        
        formulator_input = self._build_formulator_input(
            niche, 
            search_data, 
            ideas, 
            criticism, 
            schema_json
        )
        self._write_artifact(output_dir, "09_formulator_input.txt", formulator_input)
        self._write_artifact(output_dir, "09_formulator_input.txt", formulator_input)

        try:
            report, final_raw_response = self._generate_structured_report(formulator_input)
        except StructuredOutputError as error:
            if error.raw_response:
                self._write_artifact(output_dir, "10_formulator_raw_attempt_1.txt", error.raw_response)
            LOGGER.warning("Первая попытка структурированного вывода не удалась, выполняется повтор")
            retry_input = formulator_input + (
                "\n\nПовтори ответ строго как JSON, который валидируется по указанной схеме. "
                "Не добавляй markdown, комментарии или блоки кода."
            )
            self._write_artifact(output_dir, "10_formulator_retry_input.txt", retry_input)
            report, final_raw_response = self._generate_structured_report(retry_input)

        self._write_artifact(output_dir, "11_formulator_raw_response.json", final_raw_response)
        return OrchestrationResult(
            report=report,
            provider=self.provider,
            model_name=self._get_active_model_name(),
            innovator_input=innovator_input,
            innovator_output=ideas,
            critic_input=critic_input,
            critic_output=criticism,
            formulator_input=formulator_input,
            final_raw_response=final_raw_response,
        )

    def _generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.4,
        operation_name: str = "Текстовый запрос к LLM",
    ) -> str:
        if self.provider == "gemini":
            assert self.gemini_client is not None
            response = self._generate_with_retry(
                operation_name=operation_name,
                request_fn=lambda: self.gemini_client.models.generate_content(
                    model=self.settings.gemini_model_name,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=temperature,
                    ),
                ),
            )
            text = (response.text or "").strip()
        else:
            text = self._generate_with_retry(
                operation_name=operation_name,
                request_fn=lambda: self._mistral_chat_complete(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=temperature,
                ),
            )

        if not text:
            raise RuntimeError(f"{operation_name} вернул пустой текстовый ответ.")
        return text

    def _generate_structured_report(self, user_prompt: str) -> tuple[SearchResultReport, str]:
        if self.provider == "gemini":
            assert self.gemini_client is not None
            response = self._generate_with_retry(
                operation_name="Структурированный JSON-запрос к Gemini",
                request_fn=lambda: self.gemini_client.models.generate_content(
                    model=self.settings.gemini_model_name,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=FORMULATOR_PROMPT,
                        temperature=0.2,
                        response_mime_type="application/json",
                        response_schema=SearchResultReport,
                    ),
                ),
            )
            raw_text = (response.text or "").strip()

            parsed = getattr(response, "parsed", None)
            if isinstance(parsed, SearchResultReport):
                return parsed, raw_text
            if isinstance(parsed, dict):
                return SearchResultReport.model_validate(parsed), raw_text
        else:
            raw_text = self._generate_with_retry(
                operation_name="Структурированный JSON-запрос к Mistral",
                request_fn=lambda: self._mistral_chat_complete(
                    system_prompt=FORMULATOR_PROMPT,
                    user_prompt=(
                        user_prompt
                        + "\n\nВерни только JSON-объект, полностью соответствующий заданной структуре."
                    ),
                    temperature=0.2,
                    response_format={"type": "json_object"},
                ),
            )

        if not raw_text:
            raise StructuredOutputError("LLM не вернул JSON-ответ.", raw_response="")

        try:
            return SearchResultReport.model_validate_json(raw_text), raw_text
        except ValidationError as error:
            raise StructuredOutputError(
                f"LLM вернул JSON, который не прошел валидацию: {error}",
                raw_response=raw_text,
            ) from error

    def _mistral_chat_complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        assert self.mistral_client is not None

        payload: dict[str, Any] = {
            "model": self.settings.mistral_model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        response = self.mistral_client.post(
            self.settings.mistral_base_url,
            headers={
                "Authorization": f"Bearer {self.settings.mistral_api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        response.raise_for_status()

        data: dict[str, Any] = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("Mistral API не вернул choices в ответе.")

        message = choices[0].get("message") or {}
        content = message.get("content", "")
        text = self._normalize_message_content(content).strip()
        if not text:
            raise RuntimeError("Mistral API вернул пустой ответ.")
        return text

    def _generate_with_retry(
        self,
        operation_name: str,
        request_fn: Callable[[], Any],
    ) -> Any:
        delay_seconds = self.settings.llm_retry_initial_delay_seconds

        for attempt in range(1, self.settings.llm_retry_attempts + 1):
            try:
                return request_fn()
            except Exception as error:  # noqa: BLE001
                if not self._is_retryable_error(error):
                    raise
                if attempt >= self.settings.llm_retry_attempts:
                    LOGGER.error(
                        "%s завершился неудачей после %s попыток: %s",
                        operation_name,
                        attempt,
                        error,
                    )
                    raise

                wait_seconds = min(delay_seconds, self.settings.llm_retry_max_delay_seconds)
                LOGGER.warning(
                    "%s временно не выполнен (%s). Жду %.1f сек. перед повтором %s/%s.",
                    operation_name,
                    self._describe_error(error),
                    wait_seconds,
                    attempt + 1,
                    self.settings.llm_retry_attempts,
                )
                time.sleep(wait_seconds)
                delay_seconds = min(
                    delay_seconds * 2,
                    self.settings.llm_retry_max_delay_seconds,
                )

        raise RuntimeError(f"{operation_name} завершился без результата после всех повторов.")

    def _get_active_model_name(self) -> str:
        if self.provider == "gemini":
            return self.settings.gemini_model_name
        return self.settings.mistral_model_name

    @staticmethod
    def _write_artifact(output_dir: Path | None, filename: str, content: str) -> None:
        if output_dir is None:
            return
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / filename).write_text(content, encoding="utf-8")

    @staticmethod
    def _normalize_message_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(parts)
        if isinstance(content, dict):
            return json.dumps(content, ensure_ascii=False)
        return str(content)

    @staticmethod
    def _describe_error(error: Exception) -> str:
        status_code = GeminiOrchestrator._status_code_from_error(error)
        if status_code is not None:
            return f"HTTP {status_code}"
        return str(error)

    @staticmethod
    def _status_code_from_error(error: Exception) -> int | None:
        response = getattr(error, "response", None)
        if response is not None and getattr(response, "status_code", None) is not None:
            return int(response.status_code)

        status_code = getattr(error, "code", None) or getattr(error, "status_code", None)
        if isinstance(status_code, int):
            return status_code
        return None

    @staticmethod
    def _is_retryable_error(error: Exception) -> bool:
        if isinstance(
            error,
            (
                genai_errors.ServerError,
                httpx.TimeoutException,
                httpx.NetworkError,
                TimeoutError,
            ),
        ):
            return True

        if isinstance(error, genai_errors.APIError):
            status_code = GeminiOrchestrator._status_code_from_error(error)
            if status_code in {408, 429, 500, 502, 503, 504}:
                return True

        if isinstance(error, httpx.HTTPStatusError):
            status_code = GeminiOrchestrator._status_code_from_error(error)
            if status_code in {408, 429, 500, 502, 503, 504}:
                return True

        message = str(error).upper()
        retryable_markers = (
            "408",
            "429",
            "500",
            "502",
            "503",
            "504",
            "UNAVAILABLE",
            "DEADLINE_EXCEEDED",
            "TIMEOUT",
            "TIMED OUT",
            "CONNECTION RESET",
        )
        return any(marker in message for marker in retryable_markers)

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
        schema_json: str,  # Передаем схему в виде строки
    ) -> str:
        return (
            f"Ниша: {niche}\n\n"
            "Твоя задача — извлечь доработанные концепции и сформировать итоговый отчет строго по целевой JSON-схеме.\n"
            "ВАЖНО: Полностью игнорируй структуру исходного JSON (поля 'market_analysis', 'refined_concepts', 'recommendations' и т.д.). "
            "Тебе нужно пересобрать информацию и разложить её исключительно по полям, указанным в целевой схеме.\n\n"
            f"Целевая JSON-схема, которой должен соответствовать твой ответ:\n{schema_json}\n\n"
            f"Исходный отчет исследования (используй только как источник информации):\n{search_data}\n\n"
            f"Идеи новатора:\n{ideas}\n\n"
            f"Критика:\n{criticism}"
        )
