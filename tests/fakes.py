"""In-memory test doubles for the application ports (no models, no network, no disk)."""

from __future__ import annotations

import re
import zlib
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime

import numpy as np

from resume_shortlister.application.dto import (
    InsightRequest,
    RunStatus,
    RunSummary,
    ShortlistRun,
    Stage,
)
from resume_shortlister.domain.errors import ExtractionError, InsightError
from resume_shortlister.domain.models import (
    CandidateInsight,
    ExtractedDocument,
    ExtractionMethod,
    GitHubStats,
    ProfileHandles,
    ShortlistResult,
    SocialSnapshot,
)
from resume_shortlister.domain.scoring import social_score

_TOKEN = re.compile(r"\w+", re.UNICODE)


def tokens(text: str) -> set[str]:
    return {t for t in _TOKEN.findall(text.lower()) if len(t) > 2}


class FakeExtractor:
    """Treats file content as UTF-8 text; b"FAIL..." raises an ExtractionError."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def extract(self, filename: str, content: bytes) -> ExtractedDocument:
        self.calls.append(filename)
        if content.startswith(b"FAIL"):
            raise ExtractionError("Could not open this PDF (corrupt file).")
        return ExtractedDocument(
            filename=filename,
            text=content.decode(),
            methods=(ExtractionMethod.PLAIN_TEXT,),
        )


class FakeEmbedder:
    """Hashed bag-of-words vectors: texts sharing words have higher cosine similarity."""

    dim = 512

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        for token in tokens(text):
            vector[zlib.crc32(token.encode()) % self.dim] += 1.0
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack([self._vector(t) for t in texts]) if texts else np.zeros((0, self.dim))

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self.embed_queries(texts)


class FakeReranker:
    """Probability = share of the query's words found in the passage."""

    def __init__(self) -> None:
        self.pairs_scored = 0

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        self.pairs_scored += len(pairs)
        scores = []
        for query, passage in pairs:
            wanted = tokens(query)
            scores.append(len(wanted & tokens(passage)) / len(wanted) if wanted else 0.0)
        return scores


class FakeLanguageDetector:
    def detect(self, text: str) -> str | None:
        return "hi" if re.search(r"[ऀ-ॿ]", text) else "en"


class FakeSocial:
    """Every GitHub handle gets fixed, strong stats; records which handles were queried."""

    def __init__(self) -> None:
        self.queried: list[ProfileHandles] = []

    async def fetch(self, handles: ProfileHandles) -> SocialSnapshot:
        self.queried.append(handles)
        github = (
            GitHubStats(handles.github, public_repos=30, followers=50, total_stars=200)
            if handles.github
            else None
        )
        return SocialSnapshot(github=github, score=social_score(github, None, None))


class FakeSummarizer:
    """Deterministic 'LLM': lists met requirements; resumes containing FAIL-LLM error out."""

    name = "Fake LLM"

    def __init__(self) -> None:
        self.requests: list[InsightRequest] = []

    async def summarize(self, request: InsightRequest) -> CandidateInsight:
        self.requests.append(request)
        if "FAIL-LLM" in request.resume_text:
            raise InsightError("api.example.com rate limit reached; try again in a minute.")
        met = [
            r.text
            for r, m in zip(request.requirements, request.matches, strict=True)
            if m.score >= 0.5
        ]
        return CandidateInsight(
            summary=f"Scored {request.final_score:.0f} for {request.job_title}.",
            strengths=tuple(met[:2]),
            model="fake-model",
        )


class InMemoryRepository:
    def __init__(self) -> None:
        self.runs: dict[str, ShortlistRun] = {}
        self.stages: list[Stage] = []

    async def add(self, run: ShortlistRun) -> None:
        self.runs[run.id] = replace(run)

    async def update_progress(self, run_id: str, stage: Stage, done: int, total: int) -> None:
        self.stages.append(stage)
        run = self.runs.get(run_id)
        if run and not run.status.finished:
            run.status, run.stage = RunStatus.RUNNING, stage
            run.progress_done, run.progress_total = done, total

    async def complete(self, run_id: str, result: ShortlistResult) -> None:
        run = self.runs.get(run_id)
        if run and not run.status.finished:
            run.status, run.stage, run.result = RunStatus.COMPLETED, Stage.DONE, result

    async def fail(self, run_id: str, error: str) -> None:
        run = self.runs.get(run_id)
        if run and not run.status.finished:
            run.status, run.error = RunStatus.FAILED, error

    async def get(self, run_id: str) -> ShortlistRun | None:
        return self.runs.get(run_id)

    async def list(self, limit: int, offset: int) -> tuple[list[RunSummary], int]:
        ordered = sorted(self.runs.values(), key=lambda r: r.created_at, reverse=True)
        items = [
            RunSummary(
                id=r.id,
                status=r.status,
                stage=r.stage,
                created_at=r.created_at,
                title=r.title,
                document_count=r.document_count,
            )
            for r in ordered[offset : offset + limit]
        ]
        return items, len(ordered)

    async def delete(self, run_id: str) -> bool:
        return self.runs.pop(run_id, None) is not None

    async def fail_unfinished(self, error: str) -> int:
        count = 0
        for run in self.runs.values():
            if not run.status.finished:
                run.status, run.error = RunStatus.FAILED, error
                count += 1
        return count


def utc(year: int = 2026, month: int = 1, day: int = 1) -> datetime:
    return datetime(year, month, day, tzinfo=UTC)
