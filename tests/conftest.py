from __future__ import annotations

import pytest

from resume_shortlister.application.shortlist_service import ShortlistService
from resume_shortlister.domain.scoring import ScoringPolicy
from tests.fakes import (
    FakeEmbedder,
    FakeExtractor,
    FakeLanguageDetector,
    FakeReranker,
    FakeSocial,
    InMemoryRepository,
)

ANDROID_JD = """Android Developer (Kotlin)

Requirements:
- 2+ years building Android apps with Kotlin
- Jetpack Compose, MVVM architecture and Coroutines
- Firebase Auth, Firestore and Cloud Messaging
- Publishing apps on the Google Play Store

Nice to have:
- CI/CD with GitHub Actions

Benefits:
- Competitive salary and health insurance
"""

ANDROID_RESUME = """Priya Sharma - Android Developer. github.com/priya-dev
Three years building Android apps with Kotlin. Jetpack Compose, MVVM architecture, Coroutines.
Firebase Auth, Firestore, Cloud Messaging. Published four apps on the Google Play Store.
CI/CD with GitHub Actions."""

WEB_RESUME = """Rahul Verma - Web Developer. Two years building web apps with React and Node.js.
REST APIs with Express, MongoDB. Some Firebase hosting experience."""

ACCOUNTANT_RESUME = """Anita Desai - Accountant. Six years of financial accounting, GST filing,
payroll processing, Tally ERP and Excel reporting. Managed audits for manufacturing clients."""


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def repository() -> InMemoryRepository:
    return InMemoryRepository()


@pytest.fixture
def social() -> FakeSocial:
    return FakeSocial()


@pytest.fixture
def reranker() -> FakeReranker:
    return FakeReranker()


@pytest.fixture
def make_service(repository: InMemoryRepository, social: FakeSocial, reranker: FakeReranker):
    def factory(**overrides) -> ShortlistService:
        kwargs = {
            "extractor": FakeExtractor(),
            "embedder": FakeEmbedder(),
            "reranker": reranker,
            "language_detector": FakeLanguageDetector(),
            "social": social,
            "repository": repository,
            # Fake bag-of-words cosines are much lower than real model cosines.
            "policy": ScoringPolicy(semantic_floor=0.0, semantic_ceiling=0.8),
        }
        kwargs.update(overrides)
        return ShortlistService(**kwargs)

    return factory
