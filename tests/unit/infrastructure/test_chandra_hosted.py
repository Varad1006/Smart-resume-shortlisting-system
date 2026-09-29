import httpx
import pypdfium2 as pdfium
import pytest
from PIL import Image

from resume_shortlister.domain.errors import ExtractionError
from resume_shortlister.domain.models import ExtractionMethod
from resume_shortlister.infrastructure.ocr.chandra_hosted import (
    HostedChandraOcrEngine,
    images_to_pdf,
    split_pages,
)

PAGED_HTML = (
    '<!DOCTYPE html><html><body><div class="page" data-page-id="0"><p>Jane Doe</p>'
    "<p>Kotlin developer</p></div>"
    '<div class="page" data-page-id="1"><table><tr><td>Skills</td><td>Compose</td></tr>'
    "</table></div></body></html>"
)


def pages(count: int = 2) -> list[Image.Image]:
    return [Image.new("RGB", (300, 400), "white") for _ in range(count)]


def test_split_pages_maps_page_divs_to_pages():
    assert split_pages(PAGED_HTML, 2) == ["Jane Doe\nKotlin developer", "Skills | Compose |"]
    assert split_pages(PAGED_HTML, 1) == ["Jane Doe\nKotlin developer"]  # extra pages ignored
    assert split_pages("<p>Only text</p>", 2) == ["Only text", ""]


def test_images_become_one_pdf_with_one_page_each():
    document = pdfium.PdfDocument(images_to_pdf(pages(3)))
    assert len(document) == 3
    document.close()


class FakeService:
    """In-memory stand-in for the hosted conversion service."""

    def __init__(self, *, polls_before_done=1, result=None, submit_statuses=(200,)):
        self.requests: list[httpx.Request] = []
        self.polls_before_done = polls_before_done
        self.result = result or {"status": "complete", "success": True, "html": PAGED_HTML}
        self.submit_statuses = list(submit_statuses)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        assert request.url.host == "ocr.example"  # never follows URLs from responses
        assert request.headers["x-api-key"] == "secret-key"
        path = request.url.path
        if request.method == "GET" and path == "/api/v1/user_health":
            return httpx.Response(200, json={"status": "ok"})
        if request.method == "POST" and path == "/api/v1/convert":
            status = self.submit_statuses.pop(0) if self.submit_statuses else 200
            if status != 200:
                return httpx.Response(status, headers={"retry-after": "3"}, json={"detail": "x"})
            body = request.content
            assert b'name="mode"\r\n\r\nbalanced' in body
            assert b'name="output_format"\r\n\r\nhtml' in body
            assert b'name="paginate"\r\n\r\ntrue' in body
            return httpx.Response(
                200,
                json={
                    "success": True,
                    "request_id": "req1",
                    "request_check_url": "https://evil.example/steal",
                },
            )
        if request.method == "GET" and path == "/api/v1/convert/req1":
            if self.polls_before_done:
                self.polls_before_done -= 1
                return httpx.Response(200, json={"status": "processing"})
            return httpx.Response(200, json=self.result)
        if request.method == "DELETE" and path == "/api/v1/results/req1":
            return httpx.Response(200, json={"success": True})
        raise AssertionError(f"unexpected {request.method} {request.url}")

    def calls(self, method: str, path: str) -> int:
        return sum(1 for r in self.requests if r.method == method and r.url.path == path)


def engine_for(service: FakeService, **kwargs) -> tuple[HostedChandraOcrEngine, list[float]]:
    sleeps: list[float] = []
    engine = HostedChandraOcrEngine(
        "secret-key",
        base_url="https://ocr.example",
        transport=httpx.MockTransport(service),
        sleep=sleeps.append,
        **kwargs,
    )
    return engine, sleeps


def test_recognize_submits_polls_and_deletes_the_result():
    service = FakeService(polls_before_done=2)
    engine, sleeps = engine_for(service, poll_interval_seconds=1.5)
    output = engine.recognize(pages(2))

    assert output.method is ExtractionMethod.CHANDRA
    assert output.texts == ["Jane Doe\nKotlin developer", "Skills | Compose |"]
    assert service.calls("POST", "/api/v1/convert") == 1  # one request per document
    assert service.calls("GET", "/api/v1/convert/req1") == 3
    assert service.calls("DELETE", "/api/v1/results/req1") == 1  # nothing kept remotely
    assert sleeps == [1.5, 1.5]


def test_rate_limited_submissions_are_retried_after_retry_after():
    service = FakeService(polls_before_done=0, submit_statuses=(429, 503, 200))
    engine, sleeps = engine_for(service)
    assert engine.recognize(pages(1)).texts == ["Jane Doe\nKotlin developer"]
    assert service.calls("POST", "/api/v1/convert") == 3
    assert sleeps == [3.0, 3.0]


def test_failed_conversions_raise_and_are_still_deleted():
    service = FakeService(
        polls_before_done=0, result={"status": "failed", "success": False, "error": "corrupt"}
    )
    engine, _ = engine_for(service)
    with pytest.raises(ExtractionError, match="corrupt"):
        engine.recognize(pages(1))
    assert service.calls("DELETE", "/api/v1/results/req1") == 1


def test_rejected_key_marks_the_engine_unavailable():
    service = FakeService(submit_statuses=(401,))
    engine, _ = engine_for(service)
    with pytest.raises(ExtractionError, match="key rejected"):
        engine.recognize(pages(1))
    assert not engine.is_available()  # cached; no health request needed
    assert service.calls("GET", "/api/v1/user_health") == 0


def test_availability_is_probed_once_and_cached():
    service = FakeService()
    engine, _ = engine_for(service)
    assert engine.is_available() and engine.is_available()
    assert service.calls("GET", "/api/v1/user_health") == 1
    (status,) = engine.status()
    assert (status.name, status.available, status.detail) == (
        "chandra-hosted",
        True,
        "connected (balanced mode)",
    )


def test_slow_conversions_time_out():
    service = FakeService(polls_before_done=10**6)
    engine, _ = engine_for(service, timeout_seconds=0.0)
    with pytest.raises(ExtractionError, match="timed out"):
        engine.recognize(pages(1))
    assert service.calls("DELETE", "/api/v1/results/req1") == 1
