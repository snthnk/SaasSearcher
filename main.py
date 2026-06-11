from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path

from config import ConfigurationError
from gemini_orchestrator import GeminiOrchestrator, StructuredOutputError
from perplexity_client import PerplexityClient
from schemas import SearchResultReport

LOGGER = logging.getLogger("saas_finder")
PROJECT_ROOT = Path(__file__).resolve().parent


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def prompt_for_niche() -> str:
    niche = input("Введите нишу или тему для поиска SaaS-идей: ").strip()
    if not niche:
        raise ValueError("Ниша не может быть пустой.")
    return niche


def slugify(value: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", value, flags=re.UNICODE).strip().lower()
    slug = re.sub(r"[-\s]+", "_", slug, flags=re.UNICODE)
    return slug or datetime.now().strftime("%Y%m%d_%H%M%S")


def write_json_report(report: SearchResultReport, file_path: Path) -> None:
    file_path.write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_markdown_report(report: SearchResultReport) -> str:
    lines = [
        f"# SaaS Idea Finder Report: {report.niche}",
        "",
    ]

    for index, concept in enumerate(report.concepts, start=1):
        lines.extend(
            [
                f"## {index}. {concept.title}",
                "",
                f"- **Целевая аудитория:** {concept.target_audience}",
                f"- **Боли из исследования:** {', '.join(concept.source_pains)}",
                f"- **Конкуренты / альтернативы:** {', '.join(concept.competitors)}",
                f"- **Сложность реализации:** {concept.technical_difficulty}",
                f"- **Монетизация:** {concept.monetization_model}",
                "",
                "### Описание решения",
                concept.solution_description,
                "",
                "### Как учтена критика",
                concept.criticism_addressed,
                "",
            ]
        )

    return "\n".join(lines)


def write_markdown_report(report: SearchResultReport, file_path: Path) -> None:
    file_path.write_text(build_markdown_report(report), encoding="utf-8")


def write_text_file(content: str, file_path: Path) -> None:
    file_path.write_text(content, encoding="utf-8")


def main() -> int:
    configure_logging()

    try:
        niche = prompt_for_niche()
    except ValueError as error:
        LOGGER.error("Ошибка ввода: %s", error)
        return 1

    output_slug = slugify(niche)
    search_output_path = PROJECT_ROOT / f"output_{output_slug}_search.txt"
    json_output_path = PROJECT_ROOT / f"output_{output_slug}.json"
    markdown_output_path = PROJECT_ROOT / f"output_{output_slug}.md"
    raw_output_path = PROJECT_ROOT / f"output_{output_slug}_formulator_raw.txt"

    try:
        LOGGER.info("Старт поиска болей по нише: %s", niche)
        search_client = PerplexityClient()
        search_data = search_client.search_pain_points(niche)
        write_text_file(search_data, search_output_path)
        LOGGER.info("Поиск завершен. Сырой отчет сохранен в %s", search_output_path.name)

        LOGGER.info("Старт оркестрации Gemini")
        orchestrator = GeminiOrchestrator()
        report = orchestrator.process_search_results(niche=niche, search_data=search_data)

        write_json_report(report, json_output_path)
        write_markdown_report(report, markdown_output_path)
        LOGGER.info("JSON-отчет сохранен в %s", json_output_path.name)
        LOGGER.info("Markdown-отчет сохранен в %s", markdown_output_path.name)
        return 0
    except StructuredOutputError as error:
        if error.raw_response:
            write_text_file(error.raw_response, raw_output_path)
            LOGGER.error(
                "Не удалось получить валидный JSON от Gemini. Сырой ответ сохранен в %s",
                raw_output_path.name,
            )
        LOGGER.exception("Сбой на этапе структурированного вывода: %s", error)
        return 1
    except ConfigurationError as error:
        LOGGER.error("Ошибка конфигурации: %s", error)
        return 1
    except Exception as error:  # noqa: BLE001
        LOGGER.exception("Необработанная ошибка: %s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
