from types import SimpleNamespace

import httpx
import pytest
from PIL import Image

from resume_shortlister.domain.errors import ExtractionError, OcrUnavailableError
from resume_shortlister.domain.models import ExtractionMethod
from resume_shortlister.infrastructure.ocr.base import OcrEngineStatus, OcrOutput
from resume_shortlister.infrastructure.ocr.chandra import ChandraOcrEngine, layout_html_to_text
from resume_shortlister.infrastructure.ocr.fallback import FallbackOcrEngine

LAYOUT_HTML = (
    '<div data-bbox="0 0 1000 80" data-label="Page-Header">Jane Doe</div>'
    '<div data-bbox="0 80 1000 120" data-label="Section-Header"><h2>Experience</h2></div>'
    '<div data-bbox="0 120 1000 400" data-label="Text"><p>Built <b>Android</b> apps.</p>'
    "<p>Kotlin &amp; Compose</p></div>"
    '<div data-bbox="0 400 300 600" data-label="Image"><img alt="portrait photo"/></div>'
    '<div data-bbox="0 600 1000 800" data-label="Table"><table><tr><td>Skills</td>'
    "<td>Firebase</td></tr></table></div>"
)


def test_layout_html_keeps_blocks_apart_and_skips_images():
    assert layout_html_to_text(LAYOUT_HTML).splitlines() == [
        "Jane Doe",
        "Experience",
        "Built Android apps.",
        "Kotlin & Compose",
        "Skills | Firebase |",
    ]


def test_plain_html_without_layout_blocks():
    assert layout_html_to_text("<p>Hello</p><p>World</p>") == "Hello\nWorld"


def models_transport(served: list[str], calls: list[str]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        assert request.headers["Authorization"] == "Bearer secret"
        return httpx.Response(200, json={"data": [{"id": name} for name in served]})

    return httpx.MockTransport(handler)


def test_chandra_available_when_server_serves_the_model():
    calls: list[str] = []
    engine = ChandraOcrEngine(
        "http://gpu:8000/v1/", api_key="secret", transport=models_transport(["chandra"], calls)
    )
    assert engine.is_available()
    assert engine.is_available()  # cached within the TTL
    assert calls == ["http://gpu:8000/v1/models"]
    assert engine.status()[0].detail.startswith("connected to http://gpu:8000/v1")


def test_chandra_unavailable_when_model_missing_or_server_down():
    engine = ChandraOcrEngine(
        "http://gpu/v1", api_key="secret", transport=models_transport(["x"], [])
    )
    assert not engine.is_available()
    assert "does not serve model 'chandra'" in engine.status()[0].detail

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    down = ChandraOcrEngine("http://gpu/v1", transport=httpx.MockTransport(refuse))
    assert not down.is_available()
    assert "unreachable" in down.status()[0].detail


class FakeManager:
    def __init__(self, outputs):
        self.outputs = outputs
        self.calls = []

    def generate(self, batch, **kwargs):
        self.calls.append((batch, kwargs))
        return self.outputs


def test_chandra_recognize_uses_layout_prompt_and_parses_output():
    engine = ChandraOcrEngine("http://gpu/v1", max_retries=1)
    engine._manager = FakeManager([SimpleNamespace(raw=LAYOUT_HTML, error=False)])
    output = engine.recognize([Image.new("L", (50, 50), "white")])

    assert output.method is ExtractionMethod.CHANDRA
    assert output.texts[0].startswith("Jane Doe\nExperience")
    batch, kwargs = engine._manager.calls[0]
    assert batch[0].prompt_type == "ocr_layout"
    assert batch[0].image.mode == "RGB"
    assert kwargs["vllm_api_base"] == "http://gpu/v1"
    assert kwargs["max_retries"] == 1


def test_chandra_page_errors_raise():
    engine = ChandraOcrEngine("http://gpu/v1")
    engine._manager = FakeManager([SimpleNamespace(raw="", error=True)])
    with pytest.raises(ExtractionError, match="failed on 1 of 1"):
        engine.recognize([Image.new("RGB", (10, 10))])


class StubEngine:
    def __init__(self, name, available=True, fail=False):
        self.name, self.available, self.fail, self.used = name, available, fail, False

    def is_available(self):
        return self.available

    def status(self):
        return [OcrEngineStatus(self.name, self.available, "stub")]

    def recognize(self, images):
        self.used = True
        if self.fail:
            raise ExtractionError(f"{self.name} failed")
        return OcrOutput([self.name], ExtractionMethod.TESSERACT)


def test_fallback_skips_unavailable_engines():
    chandra, tesseract = StubEngine("chandra", available=False), StubEngine("tesseract")
    engine = FallbackOcrEngine([chandra, tesseract])
    assert engine.recognize([]).texts == ["tesseract"]
    assert not chandra.used
    assert [s.name for s in engine.status()] == ["chandra", "tesseract"]


def test_fallback_tries_next_engine_after_failure():
    engine = FallbackOcrEngine([StubEngine("chandra", fail=True), StubEngine("tesseract")])
    assert engine.recognize([]).texts == ["tesseract"]


def test_fallback_errors():
    with pytest.raises(OcrUnavailableError):
        FallbackOcrEngine([StubEngine("a", available=False)]).recognize([])
    with pytest.raises(ExtractionError, match="All OCR engines failed"):
        FallbackOcrEngine([StubEngine("a", fail=True)]).recognize([])
