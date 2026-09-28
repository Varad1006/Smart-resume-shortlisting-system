"""Ports: the interfaces the application needs from the outside world.

Infrastructure adapters implement these; tests substitute in-memory fakes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from resume_shortlister.application.dto import InsightRequest, RunSummary, ShortlistRun, Stage
from resume_shortlister.domain.models import (
    CandidateInsight,
    ExtractedDocument,
    ProfileHandles,
    ShortlistResult,
    SocialSnapshot,
)


class DocumentExtractor(Protocol):
    """Turns an uploaded file into text (text layer, DOCX, or OCR). Blocking."""

    def extract(self, filename: str, content: bytes) -> ExtractedDocument:
        """Raise UnsupportedDocumentError / ExtractionError when no text can be produced."""
        ...


class Embedder(Protocol):
    """Bi-encoder producing L2-normalised embeddings. Blocking (CPU/GPU bound)."""

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...


class Reranker(Protocol):
    """Cross-encoder scoring (query, passage) pairs as match probabilities in [0, 1]."""

    def score(self, pairs: Sequence[tuple[str, str]]) -> Sequence[float]: ...


class LanguageDetector(Protocol):
    def detect(self, text: str) -> str | None:
        """ISO 639-1 code (e.g. "en", "hi") or None when unsure."""
        ...


class SocialStatsProvider(Protocol):
    async def fetch(self, handles: ProfileHandles) -> SocialSnapshot: ...


class CandidateSummarizer(Protocol):
    """LLM that writes a recruiter-facing assessment. Raises InsightError on failure."""

    @property
    def name(self) -> str: ...

    async def summarize(self, request: InsightRequest) -> CandidateInsight: ...


class ShortlistRepository(Protocol):
    async def add(self, run: ShortlistRun) -> None: ...

    async def update_progress(self, run_id: str, stage: Stage, done: int, total: int) -> None: ...

    async def complete(self, run_id: str, result: ShortlistResult) -> None: ...

    async def fail(self, run_id: str, error: str) -> None: ...

    async def get(self, run_id: str) -> ShortlistRun | None: ...

    async def list(self, limit: int, offset: int) -> tuple[list[RunSummary], int]: ...

    async def delete(self, run_id: str) -> bool: ...

    async def fail_unfinished(self, error: str) -> int:
        """Mark queued/running runs as failed (e.g. after a restart). Returns the count."""
        ...
