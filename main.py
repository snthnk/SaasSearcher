from __future__ import annotations

import json
import logging
import re
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from config import ConfigurationError, get_settings
from gemini_orchestrator import GeminiOrchestrator, OrchestrationResult, StructuredOutputError
from perplexity_client import PerplexityClient
from schemas import SearchResultReport
from wordstat_client import WordstatClient

LOGGER = logging.getLogger("saas_finder")
PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_ROOT = PROJECT_ROOT / "runs"


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
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")


@dataclass
class PipelineResult:
    report: SearchResultReport
    wordstat_data: str
    us_trends: str
    ru_market: str
    combined_search_data: str
    orchestration: OrchestrationResult


def build_combined_search_data(wordstat_data: str, us_trends: str, ru_market: str) -> str:
    return (
        f"=== ДАННЫЕ О ПОИСКОВОМ СПРОСЕ В РФ (WORDSTAT) ===\n{wordstat_data}\n\n"
        f"=== АНАЛИЗ РЫНКА И ТРЕНДОВ В США ===\n{us_trends}\n\n"
        f"=== АНАЛИЗ АНАЛОГОВ И СПЕЦИФИКИ В РФ ===\n{ru_market}"
    )


def create_run_directory(niche: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_ROOT / f"{timestamp}_{slugify(niche)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def attach_run_file_logger(run_dir: Path) -> logging.Handler:
    handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logging.getLogger().addHandler(handler)
    return handler


def write_run_metadata(run_dir: Path, niche: str) -> None:
    settings = get_settings()
    metadata = {
        "niche": niche,
        "run_started_at": datetime.now().isoformat(),
        "llm_provider": settings.llm_provider,
        "llm_model": (
            settings.gemini_model_name
            if settings.llm_provider == "gemini"
            else settings.mistral_model_name
        ),
        "perplexity_model": settings.perplexity_model_name,
        "wordstat_enabled": bool(settings.yandex_api_key and settings.yandex_folder_id),
    }
    write_text_file(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        run_dir / "00_run_metadata.json",
    )


def write_error_details(run_dir: Path, error: Exception) -> None:
    write_text_file(
        "".join(traceback.format_exception(type(error), error, error.__traceback__)),
        run_dir / "99_error.txt",
    )


def run_pipeline(niche: str, run_dir: Path) -> PipelineResult:
    wordstat = WordstatClient()
    perplexity = PerplexityClient()
    orchestrator = GeminiOrchestrator()

    LOGGER.info("Шаг 1/4: сбор спроса и семантики через Wordstat")
    wordstat_data = wordstat.get_top_phrases(niche)
    write_text_file(wordstat_data, run_dir / "01_wordstat.txt")

    LOGGER.info("Шаг 2/4: исследование трендов и болей в США")
    us_trends = perplexity.search_us_trends(niche)
    write_text_file(us_trends, run_dir / "02_perplexity_us.txt")

    LOGGER.info("Шаг 3/4: исследование аналогов и локальной специфики в РФ")
    ru_market = perplexity.search_ru_alternatives(niche, wordstat_data, us_trends)
    write_text_file(ru_market, run_dir / "03_perplexity_ru.txt")

    LOGGER.info("Шаг 4/4: передача объединенного контекста в LLM-оркестратор")
    combined_search_data = build_combined_search_data(wordstat_data, us_trends, ru_market)
    write_text_file(combined_search_data, run_dir / "04_combined_search_data.txt")

    orchestration = orchestrator.process_search_results_with_trace(
        niche=niche,
        search_data=combined_search_data,
        output_dir=run_dir,
    )
    return PipelineResult(
        report=orchestration.report,
        wordstat_data=wordstat_data,
        us_trends=us_trends,
        ru_market=ru_market,
        combined_search_data=combined_search_data,
        orchestration=orchestration,
    )


def main() -> int:
    configure_logging()

    try:
        niche = prompt_for_niche()
    except ValueError as error:
        LOGGER.error("Ошибка ввода: %s", error)
        return 1

    run_dir = create_run_directory(niche)
    log_handler = attach_run_file_logger(run_dir)

    try:
        write_run_metadata(run_dir, niche)
        LOGGER.info("Старт пайплайна исследования по нише: %s", niche)
        pipeline_result = run_pipeline(niche, run_dir)
        LOGGER.info("Поиск завершен. Артефакты поиска сохранены в %s", run_dir)

        json_output_path = run_dir / "12_report.json"
        markdown_output_path = run_dir / "13_report.md"
        write_json_report(pipeline_result.report, json_output_path)
        write_markdown_report(pipeline_result.report, markdown_output_path)
        LOGGER.info("JSON-отчет сохранен в %s", json_output_path)
        LOGGER.info("Markdown-отчет сохранен в %s", markdown_output_path)
        return 0
    except StructuredOutputError as error:
        if error.raw_response:
            write_text_file(error.raw_response, run_dir / "11_formulator_raw_response_error.txt")
            LOGGER.error(
                "Не удалось получить валидный JSON от LLM. Сырой ответ сохранен в %s",
                run_dir / "11_formulator_raw_response_error.txt",
            )
        write_error_details(run_dir, error)
        LOGGER.exception("Сбой на этапе структурированного вывода: %s", error)
        return 1
    except ConfigurationError as error:
        write_error_details(run_dir, error)
        LOGGER.error("Ошибка конфигурации: %s", error)
        return 1
    except Exception as error:  # noqa: BLE001
        write_error_details(run_dir, error)
        LOGGER.exception("Необработанная ошибка: %s", error)
        return 1
    finally:
        logging.getLogger().removeHandler(log_handler)
        log_handler.close()


if __name__ == "__main__":
    raise SystemExit(main())
