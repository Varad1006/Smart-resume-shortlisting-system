"""End-to-end HTTP tests: real FastAPI app, real SQLite, fake ML adapters."""

import time

import pytest
from fastapi.testclient import TestClient

from resume_shortlister.api.app import create_app
from resume_shortlister.api.routes.ui import asset_version
from resume_shortlister.application.shortlist_service import ShortlistService
from resume_shortlister.bootstrap import Container
from resume_shortlister.config import Settings
from resume_shortlister.domain.scoring import ScoringPolicy
from resume_shortlister.infrastructure.extraction.document_extractor import DocumentExtractor
from resume_shortlister.infrastructure.persistence.sqlite_repository import (
    SqliteShortlistRepository,
)
from tests.conftest import ACCOUNTANT_RESUME, ANDROID_JD, ANDROID_RESUME, WEB_RESUME
from tests.fakes import (
    FakeEmbedder,
    FakeLanguageDetector,
    FakeReranker,
    FakeSocial,
    FakeSummarizer,
)


@pytest.fixture
def client(tmp_path):
    settings = Settings(database_path=tmp_path / "api.db", max_file_mb=1, max_total_mb=2)
    repository = SqliteShortlistRepository(settings.database_path)
    service = ShortlistService(
        extractor=DocumentExtractor(None),  # real extractor, OCR disabled
        embedder=FakeEmbedder(),
        reranker=FakeReranker(),
        language_detector=FakeLanguageDetector(),
        social=FakeSocial(),
        repository=repository,
        policy=ScoringPolicy(semantic_floor=0.0, semantic_ceiling=0.8),
        limits=settings.upload_limits(),
        summarizer=FakeSummarizer(),
    )
    container = Container(
        settings=settings,
        service=service,
        status=lambda: {"ocr": {"mode": "none", "engines": []}, "models": {}},
        startup_hooks=[repository.initialize],
    )
    with TestClient(create_app(container=container)) as test_client:
        yield test_client


def upload(client, files, job=ANDROID_JD, **fields):
    data = {"job_description": job, **fields}
    return client.post("/api/v1/shortlists", data=data, files=[("files", f) for f in files])


def wait_until_finished(client, run_id, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        run = client.get(f"/api/v1/shortlists/{run_id}").json()
        if run["status"] in ("completed", "failed"):
            return run
        time.sleep(0.05)
    raise AssertionError("run did not finish in time")


RESUME_FILES = [
    ("android.txt", ANDROID_RESUME.encode(), "text/plain"),
    ("web.txt", WEB_RESUME.encode(), "text/plain"),
    ("accountant.txt", ACCOUNTANT_RESUME.encode(), "text/plain"),
    ("scan.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 64, "image/png"),
]


def test_full_shortlist_flow(client):
    response = upload(
        client, RESUME_FILES, shortlist_size="2", include_social="true", include_insights="true"
    )
    assert response.status_code == 202
    created = response.json()
    assert created["status"] == "queued"
    assert created["ui_url"] == f"/shortlists/{created['id']}"

    run = wait_until_finished(client, created["id"])
    assert run["status"] == "completed", run["error"]
    result = run["result"]
    shortlisted = [c for c in result["candidates"] if c["shortlisted"]]
    assert [c["filename"] for c in shortlisted] == ["android.txt", "web.txt"]
    assert shortlisted[0]["social_bonus"] > 0
    assert shortlisted[0]["insight"]["model"] == "fake-model"
    assert run["options"]["include_insights"] is True
    assert result["skipped"][0]["filename"] == "scan.png"  # image but OCR disabled
    assert run["progress"]["stage"] == "done"

    page = client.get(created["ui_url"])
    assert page.status_code == 200
    assert "android.txt" in page.text and "Evidence and details" in page.text
    assert "AI summary" in page.text and "Scored " in page.text

    history = client.get("/api/v1/shortlists").json()
    assert history["total"] == 1
    assert history["items"][0]["top_candidate"] == "android.txt"
    assert "Android Developer (Kotlin)" in client.get("/shortlists").text


def test_delete_run(client):
    run_id = upload(client, RESUME_FILES[:1]).json()["id"]
    wait_until_finished(client, run_id)
    assert client.delete(f"/api/v1/shortlists/{run_id}").status_code == 204
    assert client.get(f"/api/v1/shortlists/{run_id}").status_code == 404
    assert client.delete(f"/api/v1/shortlists/{run_id}").status_code == 404
    assert client.get(f"/shortlists/{run_id}").status_code == 404


@pytest.mark.parametrize(
    ("job", "files", "status", "detail"),
    [
        ("   ", RESUME_FILES[:1], 422, "must not be empty"),
        (ANDROID_JD, [("big.txt", b"x" * (1024 * 1024 + 1), "text/plain")], 413, "larger than"),
    ],
)
def test_invalid_submissions(client, job, files, status, detail):
    response = upload(client, files, job=job)
    assert response.status_code == status
    assert detail in response.json()["detail"]


def test_missing_files_is_a_validation_error(client):
    response = client.post("/api/v1/shortlists", data={"job_description": ANDROID_JD})
    assert response.status_code == 422


def test_oversized_request_is_rejected_before_parsing(client):
    response = client.post(
        "/api/v1/shortlists", content=b"x", headers={"content-length": str(50 * 1024 * 1024)}
    )
    assert response.status_code == 413


def test_extract_endpoint(client):
    response = client.post(
        "/api/v1/extract", files={"file": ("cv.txt", ANDROID_RESUME.encode(), "text/plain")}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["methods"] == ["plain_text"]
    assert body["profiles"]["github"] == "priya-dev"
    assert body["language"] == "en"
    unsupported = client.post(
        "/api/v1/extract", files={"file": ("cv.bin", bytes(range(256)) * 4, "application/pdf")}
    )
    assert unsupported.status_code == 415


def test_pages_and_system_endpoints(client):
    home = client.get("/")
    assert home.status_code == 200
    assert "Job description" in home.text and "70% requirement coverage" in home.text
    assert "Write AI summaries for the top 5 candidates with" in home.text
    assert "Fake LLM" in home.text and "api.groq.com" in home.text
    assert client.get("/health").json() == {"status": "ok"}
    status = client.get("/api/v1/status").json()
    assert status["status"] == "ok" and status["ocr"]["mode"] == "none"
    assert client.get("/static/css/app.css").status_code == 200
    assert f"/static/css/app.css?v={asset_version()}" in home.text  # content-hash cache busting
    assert client.get("/static/js/app.js").status_code == 200
    assert client.get("/openapi.json").json()["info"]["title"] == "Smart Resume Shortlisting System"
