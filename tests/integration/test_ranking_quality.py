"""End-to-end ranking quality (downloads ~1 GB on first run).

Run with:  uv run pytest -m slow
"""

import pytest

from resume_shortlister.application.dto import UploadedDocument
from resume_shortlister.application.shortlist_service import ShortlistService
from resume_shortlister.config import Settings
from resume_shortlister.domain.scoring import MatchBand, match_band
from resume_shortlister.infrastructure.extraction.document_extractor import DocumentExtractor
from resume_shortlister.infrastructure.nlp.embedder import SentenceTransformerEmbedder
from resume_shortlister.infrastructure.nlp.language import LangDetectLanguageDetector
from resume_shortlister.infrastructure.nlp.reranker import CrossEncoderReranker
from tests.fakes import FakeSocial, InMemoryRepository

pytestmark = [pytest.mark.slow, pytest.mark.anyio]

JOB = """Android Developer (Kotlin)

We are hiring an Android engineer to build our consumer mobile apps.

Requirements:
- 2+ years of Android development with Kotlin
- Jetpack Compose and MVVM architecture
- Coroutines, Room and Retrofit
- Firebase (Auth, Firestore, Cloud Messaging)
- Publishing apps on the Google Play Store and writing unit tests

Nice to have:
- CI/CD with GitHub Actions and Hilt dependency injection

Benefits:
- Health insurance and a competitive salary
"""

RESUMES = {
    "strong_en.txt": """Priya Sharma, Android Developer, Pune. github.com/priya-dev
Summary: Android engineer with three years of experience building consumer apps in Kotlin.
Experience: Mobile Engineer at ShopKart (2022-2025). Rebuilt the checkout flow in Jetpack
Compose with an MVVM architecture, Kotlin Coroutines and Flow. Local caching with Room,
networking with Retrofit. Integrated Firebase Auth, Firestore and Cloud Messaging for order
notifications. Published four apps on the Google Play Store with over one million downloads.
Wrote unit tests with JUnit and MockK, set up CI/CD with GitHub Actions, used Hilt for
dependency injection. Education: B.Tech in Computer Engineering.""",
    "strong_hi.txt": """प्रिया शर्मा - एंड्रॉइड डेवलपर, पुणे।
सारांश: कोटलिन (Kotlin) में तीन वर्षों का एंड्रॉइड ऐप विकास अनुभव।
अनुभव: ShopKart में मोबाइल इंजीनियर। Jetpack Compose और MVVM आर्किटेक्चर के साथ चेकआउट
फिर से बनाया, Coroutines, Room और Retrofit का उपयोग किया। Firebase Auth, Firestore और
Cloud Messaging जोड़ा। Google Play Store पर चार ऐप प्रकाशित किए। JUnit से यूनिट टेस्ट लिखे।""",
    "strong_fr.txt": """Pierre Martin - Développeur Android, Lyon.
Trois ans d'expérience en développement d'applications Android avec Kotlin. Interfaces en
Jetpack Compose, architecture MVVM, Coroutines, Room et Retrofit. Intégration de Firebase
(Auth, Firestore, Cloud Messaging). A publié quatre applications sur le Google Play Store.
Tests unitaires avec JUnit, intégration continue avec GitHub Actions, injection avec Hilt.""",
    "web_dev.txt": """Rahul Verma - Full-stack Web Developer. Two years building web applications
with React, Node.js and MongoDB. Designed REST APIs with Express, deployed on Firebase Hosting.
Some Java from university. Education: B.E. Information Technology.""",
    "ios_dev.txt": """Karan Mehta - iOS Developer. Four years with Swift, SwiftUI, Combine and Core
Data. Published three apps on the App Store. Unit tests with XCTest, CI with Fastlane.""",
    "accountant_en.txt": """Anita Desai - Chartered Accountant. Six years of experience in
financial accounting, GST filing, payroll processing, Tally ERP and Excel reporting. Managed
statutory audits for mid-size manufacturing clients. Education: B.Com, CA.""",
    "accountant_hi.txt": """अनीता देसाई - लेखाकार। वित्तीय लेखांकन, जीएसटी फाइलिंग, पेरोल और
टैली ईआरपी में छह वर्षों का अनुभव। विनिर्माण ग्राहकों के लिए ऑडिट प्रबंधित किए।""",
}


@pytest.fixture(scope="module")
def service() -> ShortlistService:
    settings = Settings()
    return ShortlistService(
        extractor=DocumentExtractor(None),
        embedder=SentenceTransformerEmbedder(
            settings.embedding_model,
            query_prompt=settings.embedding_query_prompt,
            document_prompt=settings.embedding_document_prompt,
        ),
        reranker=CrossEncoderReranker(settings.reranker_model),
        language_detector=LangDetectLanguageDetector(),
        social=FakeSocial(),
        repository=InMemoryRepository(),
        policy=settings.scoring_policy(),
    )


async def test_real_models_rank_by_relevance_across_languages(service):
    documents = [UploadedDocument(name, text.encode()) for name, text in RESUMES.items()]
    result = await service.rank(JOB, documents)
    ranked = [c.filename for c in result.shortlisted]

    assert set(ranked[:3]) == {"strong_en.txt", "strong_hi.txt", "strong_fr.txt"}
    assert set(ranked[-2:]) == {"accountant_en.txt", "accountant_hi.txt"}

    by_name = {c.filename: c for c in result.candidates}
    assert by_name["strong_hi.txt"].language == "hi"
    assert by_name["strong_fr.txt"].language == "fr"
    for name in ("strong_en.txt", "strong_hi.txt", "strong_fr.txt"):
        assert match_band(by_name[name].final_score) is MatchBand.STRONG, name
    for name in ("accountant_en.txt", "accountant_hi.txt"):
        assert match_band(by_name[name].final_score) is MatchBand.WEAK, name
