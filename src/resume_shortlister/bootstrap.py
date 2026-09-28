"""Composition root: the only place that knows which adapter implements which port."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from typing import Any
from urllib.parse import urlparse

from resume_shortlister.application.shortlist_service import ShortlistService
from resume_shortlister.config import Settings
from resume_shortlister.infrastructure.extraction.document_extractor import DocumentExtractor
from resume_shortlister.infrastructure.llm.openai_compatible import OpenAICompatibleSummarizer
from resume_shortlister.infrastructure.nlp.embedder import SentenceTransformerEmbedder
from resume_shortlister.infrastructure.nlp.language import LangDetectLanguageDetector
from resume_shortlister.infrastructure.nlp.reranker import CrossEncoderReranker
from resume_shortlister.infrastructure.ocr.base import OcrEngine
from resume_shortlister.infrastructure.ocr.chandra import ChandraOcrEngine
from resume_shortlister.infrastructure.ocr.fallback import FallbackOcrEngine
from resume_shortlister.infrastructure.ocr.tesseract import TesseractOcrEngine
from resume_shortlister.infrastructure.persistence.sqlite_repository import (
    SqliteShortlistRepository,
)
from resume_shortlister.infrastructure.social.codechef import CodeChefClient
from resume_shortlister.infrastructure.social.github import GitHubClient
from resume_shortlister.infrastructure.social.leetcode import LeetCodeClient
from resume_shortlister.infrastructure.social.service import (
    SocialProfileService,
    build_http_client,
)

logger = logging.getLogger(__name__)

Hook = Callable[[], Awaitable[None]]


@dataclass
class Container:
    settings: Settings
    service: ShortlistService
    status: Callable[[], dict[str, Any]]  # blocking system status (OCR reachability, models)
    startup_hooks: list[Hook] = field(default_factory=list)
    shutdown_hooks: list[Hook] = field(default_factory=list)
    background: set[asyncio.Task[Any]] = field(default_factory=set)

    async def startup(self) -> None:
        for hook in self.startup_hooks:
            await hook()

    async def shutdown(self) -> None:
        await self.service.shutdown()
        for task in self.background:
            task.cancel()
        for hook in reversed(self.shutdown_hooks):
            await hook()


def build_ocr_engine(settings: Settings) -> OcrEngine | None:
    def chandra() -> ChandraOcrEngine:
        return ChandraOcrEngine(
            settings.chandra_api_base,
            model_name=settings.chandra_model_name,
            api_key=settings.chandra_api_key.get_secret_value(),
            prompt_type=settings.chandra_prompt_type,
            max_retries=settings.chandra_max_retries,
        )

    def tesseract() -> TesseractOcrEngine:
        return TesseractOcrEngine(settings.tesseract_langs)

    if settings.ocr_engine == "none":
        return None
    if settings.ocr_engine == "chandra":
        return FallbackOcrEngine([chandra()])
    if settings.ocr_engine == "tesseract":
        return FallbackOcrEngine([tesseract()])
    return FallbackOcrEngine([chandra(), tesseract()])


def build_container(settings: Settings) -> Container:
    ocr = build_ocr_engine(settings)
    extractor = DocumentExtractor(
        ocr,
        max_pages=settings.max_pdf_pages,
        min_chars_per_page=settings.pdf_min_chars_per_page,
        ocr_dpi=settings.ocr_dpi,
    )
    embedder = SentenceTransformerEmbedder(
        settings.embedding_model,
        device=settings.model_device,
        query_prompt=settings.embedding_query_prompt,
        document_prompt=settings.embedding_document_prompt,
    )
    reranker = CrossEncoderReranker(settings.reranker_model, device=settings.model_device)
    http = build_http_client(settings.social_timeout_seconds)
    token = settings.github_token.get_secret_value() if settings.github_token else None
    social = SocialProfileService(
        GitHubClient(http, token),
        LeetCodeClient(http),
        CodeChefClient(http),
        cache_ttl_seconds=settings.social_cache_ttl_seconds,
    )
    repository = SqliteShortlistRepository(settings.database_path)
    summarizer = (
        OpenAICompatibleSummarizer(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key.get_secret_value(),
            model=settings.llm_model,
            display_name=settings.llm_display_name,
            max_tokens=settings.llm_max_tokens,
            reasoning_effort=settings.llm_reasoning_effort or None,
            temperature=settings.llm_temperature,
            timeout_seconds=settings.llm_timeout_seconds,
        )
        if settings.llm_api_key
        else None
    )
    service = ShortlistService(
        extractor=extractor,
        embedder=embedder,
        reranker=reranker,
        language_detector=LangDetectLanguageDetector(),
        social=social,
        repository=repository,
        policy=settings.scoring_policy(),
        limits=settings.upload_limits(),
        max_concurrent_runs=settings.max_concurrent_runs,
        summarizer=summarizer,
        max_insights=settings.llm_max_candidates,
        max_insight_chars=settings.llm_max_resume_chars,
    )

    def status() -> dict[str, Any]:
        return {
            "ocr": {
                "mode": settings.ocr_engine,
                "engines": [asdict(s) for s in ocr.status()] if ocr else [],
            },
            "models": {
                "embedding": {"name": embedder.model_name, "loaded": embedder.loaded},
                "reranker": {"name": reranker.model_name, "loaded": reranker.loaded},
                "device": settings.model_device,
            },
            "llm": llm_status(settings, service.summarizer_name),
        }

    container = Container(settings=settings, service=service, status=status)

    async def init_database() -> None:
        await repository.initialize()
        interrupted = await repository.fail_unfinished("Interrupted by a server restart.")
        if interrupted:
            logger.warning("Marked %d unfinished run(s) as failed after restart", interrupted)

    async def warmup_models() -> None:
        if not settings.warmup_models:
            return

        def load() -> None:
            embedder.warmup()
            reranker.warmup()
            logger.info("Models loaded")

        task = asyncio.create_task(asyncio.to_thread(load), name="warmup-models")
        container.background.add(task)
        task.add_done_callback(container.background.discard)

    container.startup_hooks += [init_database, warmup_models]
    container.shutdown_hooks.append(http.aclose)
    if summarizer is not None:
        container.shutdown_hooks.append(summarizer.aclose)
    return container


def llm_status(settings: Settings, summarizer_name: str | None) -> dict[str, Any]:
    """Public LLM status. Never includes the API key."""
    if summarizer_name is None:
        return {"configured": False}
    return {
        "configured": True,
        "name": summarizer_name,
        "model": settings.llm_model,
        "host": urlparse(settings.llm_base_url).hostname,
        "max_candidates": settings.llm_max_candidates,
    }
