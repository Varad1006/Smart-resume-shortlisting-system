"""HTTP response models (the public API contract)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from resume_shortlister.application.dto import (
    RunStatus,
    RunSummary,
    ShortlistOptions,
    ShortlistRun,
    Stage,
)
from resume_shortlister.domain.models import (
    ExtractedDocument,
    ExtractionMethod,
    ProfileHandles,
    ShortlistResult,
)


def run_ui_url(run_id: str) -> str:
    return f"/shortlists/{run_id}"


def run_api_url(run_id: str) -> str:
    return f"/api/v1/shortlists/{run_id}"


class RunCreatedResponse(BaseModel):
    id: str
    status: RunStatus
    status_url: str = Field(description="Poll this URL for progress and results.")
    ui_url: str

    @classmethod
    def of(cls, run: ShortlistRun) -> RunCreatedResponse:
        return cls(
            id=run.id, status=run.status, status_url=run_api_url(run.id), ui_url=run_ui_url(run.id)
        )


class ProgressOut(BaseModel):
    stage: Stage
    done: int
    total: int


class RunDetailResponse(BaseModel):
    id: str
    status: RunStatus
    progress: ProgressOut
    created_at: datetime
    updated_at: datetime
    title: str
    job_description: str
    options: ShortlistOptions
    result: ShortlistResult | None = None
    error: str | None = None

    @classmethod
    def of(cls, run: ShortlistRun) -> RunDetailResponse:
        return cls(
            id=run.id,
            status=run.status,
            progress=ProgressOut(stage=run.stage, done=run.progress_done, total=run.progress_total),
            created_at=run.created_at,
            updated_at=run.updated_at,
            title=run.title,
            job_description=run.job_description,
            options=run.options,
            result=run.result,
            error=run.error,
        )


class RunSummaryOut(BaseModel):
    id: str
    status: RunStatus
    stage: Stage
    created_at: datetime
    title: str
    document_count: int
    shortlisted_count: int | None
    top_candidate: str | None
    top_score: float | None
    ui_url: str

    @classmethod
    def of(cls, summary: RunSummary) -> RunSummaryOut:
        return cls(
            id=summary.id,
            status=summary.status,
            stage=summary.stage,
            created_at=summary.created_at,
            title=summary.title,
            document_count=summary.document_count,
            shortlisted_count=summary.shortlisted_count,
            top_candidate=summary.top_candidate,
            top_score=summary.top_score,
            ui_url=run_ui_url(summary.id),
        )


class RunListResponse(BaseModel):
    items: list[RunSummaryOut]
    total: int
    limit: int
    offset: int


class ExtractionResponse(BaseModel):
    filename: str
    methods: list[ExtractionMethod]
    pages: int
    language: str | None
    profiles: ProfileHandles
    links: list[str]
    warnings: list[str]
    characters: int
    text: str

    @classmethod
    def of(
        cls, document: ExtractedDocument, language: str | None, profiles: ProfileHandles
    ) -> ExtractionResponse:
        return cls(
            filename=document.filename,
            methods=list(document.methods),
            pages=document.pages,
            language=language,
            profiles=profiles,
            links=list(document.links),
            warnings=list(document.warnings),
            characters=len(document.text),
            text=document.text,
        )


class StatusResponse(BaseModel):
    status: str = "ok"
    version: str
    ocr: dict[str, Any]
    models: dict[str, Any]
    llm: dict[str, Any] = Field(default_factory=lambda: {"configured": False})
