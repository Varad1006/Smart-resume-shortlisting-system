"""Data carried across the application boundary (requests, runs, summaries)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from resume_shortlister.domain.models import Requirement, RequirementMatch, ShortlistResult


class RunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

    @property
    def finished(self) -> bool:
        return self in (RunStatus.COMPLETED, RunStatus.FAILED)


class Stage(StrEnum):
    QUEUED = "queued"
    EXTRACTING = "extracting"
    RETRIEVING = "retrieving"
    MATCHING = "matching"
    PROFILES = "profiles"
    INSIGHTS = "insights"
    DONE = "done"


@dataclass(frozen=True, slots=True)
class UploadedDocument:
    filename: str
    content: bytes = field(repr=False)


@dataclass(frozen=True, slots=True)
class ShortlistOptions:
    shortlist_size: int | None = None  # None -> policy default
    include_social: bool = False
    include_insights: bool = False  # AI summaries for the top shortlisted candidates


@dataclass(frozen=True, slots=True)
class InsightRequest:
    """What an LLM sees about one candidate: redacted resume text plus our evidence."""

    job_title: str
    requirements: tuple[Requirement, ...]
    matches: tuple[RequirementMatch, ...]
    resume_text: str  # contact details removed, truncated
    language: str | None
    final_score: float


@dataclass(frozen=True, slots=True)
class UploadLimits:
    max_files: int = 50
    max_file_bytes: int = 10 * 1024 * 1024
    max_total_bytes: int = 100 * 1024 * 1024
    max_job_description_chars: int = 20_000
    max_shortlist_size: int = 50


@dataclass(slots=True)
class ShortlistRun:
    id: str
    status: RunStatus
    stage: Stage
    created_at: datetime
    updated_at: datetime
    job_description: str
    options: ShortlistOptions
    document_count: int
    progress_done: int = 0
    progress_total: int = 0
    result: ShortlistResult | None = None
    error: str | None = None

    @property
    def title(self) -> str:
        return job_title(self.job_description)


@dataclass(frozen=True, slots=True)
class RunSummary:
    id: str
    status: RunStatus
    stage: Stage
    created_at: datetime
    title: str
    document_count: int
    shortlisted_count: int | None = None
    top_candidate: str | None = None
    top_score: float | None = None


def job_title(job_description: str, max_chars: int = 80) -> str:
    """First non-empty line of a job description, shortened for listings."""
    for line in job_description.splitlines():
        line = line.strip(" #*-•\t")
        if line:
            return line if len(line) <= max_chars else line[: max_chars - 1].rstrip() + "…"
    return "Untitled job"
