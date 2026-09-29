"""Chandra OCR adapter (https://github.com/datalab-to/chandra).

Chandra is a 5B-parameter vision-language OCR model with strong handwriting, layout and
multilingual (90+ languages) support. It needs an NVIDIA GPU, so we talk to it over HTTP:
any OpenAI-compatible server hosting ``datalab-to/chandra-ocr-2`` (vLLM, as started by the
``chandra`` service in docker-compose, or a remote GPU box). The official ``chandra-ocr``
client handles prompting, retries and HTML parsing.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Sequence

import httpx
from bs4 import BeautifulSoup
from PIL import Image

from resume_shortlister.domain.errors import ExtractionError
from resume_shortlister.domain.models import ExtractionMethod
from resume_shortlister.infrastructure.ocr.base import OcrEngineStatus, OcrOutput

logger = logging.getLogger(__name__)


_BLOCK_TAGS = ["br", "p", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "pre", "caption"]
_SKIPPED_LABELS = {"Blank-Page", "Image", "Figure"}  # photos/decoration are not resume text


def layout_html_to_text(raw: str) -> str:
    """Flatten Chandra's raw layout HTML (top-level ``<div data-label>`` blocks) to text.

    Every layout block (header, section title, list, table...) becomes its own line(s), so
    separate blocks never run together; table cells are separated with " | ".
    """
    soup = BeautifulSoup(raw, "html.parser")
    blocks = soup.find_all("div", recursive=False) or [soup]
    parts: list[str] = []
    for block in blocks:
        if block.get("data-label") in _SKIPPED_LABELS:
            continue
        for tag in block.find_all(_BLOCK_TAGS):
            tag.insert_after("\n")
        for cell in block.find_all(["td", "th"]):
            cell.insert_after(" | ")
        lines = (" ".join(line.split()) for line in block.get_text().splitlines())
        parts.extend(line for line in lines if line)
    return "\n".join(parts)


class ChandraOcrEngine:
    method = ExtractionMethod.CHANDRA

    def __init__(
        self,
        api_base: str,
        *,
        model_name: str = "chandra",
        api_key: str = "EMPTY",
        prompt_type: str = "ocr_layout",
        max_retries: int = 2,
        max_output_tokens: int | None = None,
        health_ttl_seconds: float = 30.0,
        health_timeout_seconds: float = 3.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.model_name = model_name
        self._api_key = api_key
        self._prompt_type = prompt_type
        self._max_retries = max_retries
        self._max_output_tokens = max_output_tokens
        self._health_ttl = health_ttl_seconds
        self._health_timeout = health_timeout_seconds
        self._transport = transport
        self._lock = threading.Lock()
        self._checked_at = float("-inf")
        self._available = False
        self._detail = "not checked yet"
        self._manager = None

    # ------------------------------------------------------------ availability
    def is_available(self) -> bool:
        with self._lock:
            if time.monotonic() - self._checked_at < self._health_ttl:
                return self._available
            self._available, self._detail = self._probe()
            self._checked_at = time.monotonic()
            return self._available

    def _probe(self) -> tuple[bool, str]:
        try:
            with httpx.Client(timeout=self._health_timeout, transport=self._transport) as client:
                response = client.get(
                    f"{self.api_base}/models",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                )
            response.raise_for_status()
            served = [m.get("id") for m in response.json().get("data", [])]
        except (httpx.HTTPError, ValueError) as exc:
            return False, f"unreachable at {self.api_base} ({type(exc).__name__})"
        if self.model_name not in served:
            return False, f"server is up but does not serve model {self.model_name!r}: {served}"
        return True, f"connected to {self.api_base} (model {self.model_name!r})"

    def status(self) -> list[OcrEngineStatus]:
        available = self.is_available()
        return [OcrEngineStatus(name="chandra-server", available=available, detail=self._detail)]

    # ------------------------------------------------------------- recognition
    def _inference_manager(self):  # type: ignore[no-untyped-def]
        if self._manager is None:
            from chandra.model import InferenceManager
            from chandra.settings import settings as chandra_settings

            # generate_vllm() reads the model name and key from chandra's global settings.
            chandra_settings.VLLM_MODEL_NAME = self.model_name
            chandra_settings.VLLM_API_KEY = self._api_key
            self._manager = InferenceManager(method="vllm")
        return self._manager

    def recognize(self, images: Sequence[Image.Image]) -> OcrOutput:
        if not images:
            return OcrOutput(texts=[], method=self.method)
        from chandra.model.schema import BatchInputItem

        batch = [
            BatchInputItem(image=image.convert("RGB"), prompt_type=self._prompt_type)
            for image in images
        ]
        try:
            outputs = self._inference_manager().generate(
                batch,
                max_output_tokens=self._max_output_tokens,
                vllm_api_base=self.api_base,
                max_retries=self._max_retries,
            )
        except Exception as exc:  # network/protocol errors from the OpenAI client
            self._checked_at = float("-inf")  # re-probe on next use
            raise ExtractionError(f"Chandra OCR request failed: {exc}") from exc

        failed = sum(1 for output in outputs if output.error)
        if failed:
            self._checked_at = float("-inf")
            raise ExtractionError(f"Chandra OCR failed on {failed} of {len(outputs)} page(s).")
        return OcrOutput(texts=[layout_html_to_text(o.raw) for o in outputs], method=self.method)
