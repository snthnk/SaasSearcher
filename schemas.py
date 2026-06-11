from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class SaaSConcept(BaseModel):
    title: str = Field(description="Краткое и запоминающееся название концепта")
    target_audience: str = Field(description="Целевая аудитория и её сегмент")
    source_pains: List[str] = Field(
        description="Список болей из отчета Perplexity, которые решает данный концепт"
    )
    solution_description: str = Field(
        description="Подробное описание решения и его ключевого функционала"
    )
    competitors: List[str] = Field(
        description="Потенциальные конкуренты или альтернативные способы решения"
    )
    technical_difficulty: str = Field(
        description="Оценка сложности реализации (Low/Medium/High) с кратким пояснением"
    )
    monetization_model: str = Field(
        description="Предлагаемая модель монетизации (SaaS, Pay-as-you-go, Freemium и т.д.)"
    )
    criticism_addressed: str = Field(
        description="Как данная концепция учитывает критику (слабые стороны, барьеры)"
    )


class SearchResultReport(BaseModel):
    niche: str = Field(description="Исходная ниша или тема")
    concepts: List[SaaSConcept] = Field(description="Список разработанных концепций")
