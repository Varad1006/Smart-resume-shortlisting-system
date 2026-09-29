"""Hosted Chandra OCR adapter (Datalab's document conversion service).

For machines without an NVIDIA GPU: scanned pages are sent as one PDF per document,
the result is polled until complete, and the stored result is deleted immediately
afterwards so no resume data is kept on the provider's side.

Protocol: POST /api/v1/convert (multipart) -> request_id; GET /api/v1/convert/{id}
until status is "complete"/"failed"; DELETE /api/v1/results/{id}. Auth: X-API-Key.
"""

from __future__ import annotations

import io
import logging
import threading
import time
from collections.abc import Callable, Sequence
from typing import Any

import httpx
from bs4 import BeautifulSoup
from PIL import Image

from resume_shortlister.domain.errors import ExtractionError
from resume_shortlister.domain.models import ExtractionMethod
from resume_shortlister.infrastructure.ocr.base import OcrEngineStatus, OcrOutput
from resume_shortlister.infrastructure.ocr.chandra import layout_html_to_text

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://www.datalab.to"
_RETRY_STATUSES = {429, 500, 502, 503, 504}


def images_to_pdf(images: Sequence[Image.Image], dpi: int = 200) -> bytes:
    """Bundle page images into one PDF so a whole document costs a single request."""
    pages = [image.convert("RGB") for image in images]
    buffer = io.BytesIO()
    pages[0].save(buffer, "PDF", save_all=True, append_images=pages[1:], resolution=dpi)
    return buffer.getvalue()


def split_pages(html: str, page_count: int) -> list[str]:
    """Map paginated HTML (``<div class="page" data-page-id="N">``) back to page texts."""
    soup = BeautifulSoup(html, "html.parser")
    texts = [""] * page_count
    pages = soup.select("div.page[data-page-id]")
    if not pages:
        if page_count:
            body = soup.body or soup
            texts[0] = layout_html_to_text(body.decode_contents())
        return texts
    for page in pages:
        try:
            index = int(str(page["data-page-id"]))
        except ValueError:
            continue
        if 0 <= index < page_count:
            texts[index] = layout_html_to_text(page.decode_contents())
    return texts


def _retry_after(response: httpx.Response, attempt: int) -> float:
    header = response.headers.get("retry-after", "")
    try:
        return min(60.0, max(0.5, float(header)))
    except ValueError:
        return min(30.0, 2.0 * 2**attempt)


class HostedChandraOcrEngine:
    method = ExtractionMethod.CHANDRA

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        mode: str = "balanced",  # Chandra runs in "balanced" and "accurate" modes
        poll_interval_seconds: float = 2.0,
        timeout_seconds: float = 180.0,
        max_retries: int = 4,
        health_ttl_seconds: float = 300.0,
        failed_health_ttl_seconds: float = 30.0,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._mode = mode
        self._poll_interval = poll_interval_seconds
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._health_ttl = health_ttl_seconds
        self._failed_health_ttl = failed_health_ttl_seconds
        self._transport = transport
        self._sleep = sleep
        self._lock = threading.Lock()
        self._checked_at = float("-inf")
        self._available = False
        self._detail = "not checked yet"

    def _client(self) -> httpx.Client:
        return httpx.Client(
            base_url=self._base_url,
            headers={"X-API-Key": self._api_key},
            timeout=httpx.Timeout(60.0),
            transport=self._transport,
        )

    # ------------------------------------------------------------ availability
    def is_available(self) -> bool:
        with self._lock:
            ttl = self._health_ttl if self._available else self._failed_health_ttl
            if time.monotonic() - self._checked_at < ttl:
                return self._available
            self._available, self._detail = self._probe()
            self._checked_at = time.monotonic()
            return self._available

    def _probe(self) -> tuple[bool, str]:
        try:
            with self._client() as client:
                response = client.get("/api/v1/user_health", timeout=5.0)
        except httpx.HTTPError as exc:
            return False, f"unreachable ({type(exc).__name__})"
        if response.status_code == 401:
            return False, "key rejected (HTTP 401)"
        if response.status_code == 402:
            return False, "spending limit reached (HTTP 402)"
        if response.status_code != 200:
            return False, f"health check returned HTTP {response.status_code}"
        return True, f"connected ({self._mode} mode)"

    def status(self) -> list[OcrEngineStatus]:
        available = self.is_available()
        return [OcrEngineStatus(name="chandra-hosted", available=available, detail=self._detail)]

    def _mark_unavailable(self, detail: str) -> None:
        with self._lock:
            self._available, self._detail = False, detail
            self._checked_at = time.monotonic()

    # ------------------------------------------------------------- recognition
    def recognize(self, images: Sequence[Image.Image]) -> OcrOutput:
        if not images:
            return OcrOutput(texts=[], method=self.method)
        document = images_to_pdf(images)
        with self._client() as client:
            request_id = self._submit(client, document)
            try:
                result = self._wait(client, request_id)
            finally:
                self._delete(client, request_id)
        if not result.get("success"):
            raise ExtractionError(f"Hosted OCR failed: {result.get('error') or 'unknown error'}")
        return OcrOutput(
            texts=split_pages(str(result.get("html") or ""), len(images)), method=self.method
        )

    def _request(self, send: Callable[[], httpx.Response], action: str) -> httpx.Response:
        """Send with retries on rate limits, server errors and network failures."""
        for attempt in range(self._max_retries + 1):
            last_attempt = attempt == self._max_retries
            try:
                response = send()
            except httpx.HTTPError as exc:
                if last_attempt:
                    raise ExtractionError(f"Hosted OCR {action} failed: {exc!r}") from exc
                self._sleep(min(30.0, 2.0 * 2**attempt))
                continue
            if response.status_code in _RETRY_STATUSES and not last_attempt:
                self._sleep(_retry_after(response, attempt))
                continue
            if response.status_code in (401, 402):
                detail = "key rejected" if response.status_code == 401 else "spending limit reached"
                self._mark_unavailable(f"{detail} (HTTP {response.status_code})")
                raise ExtractionError(f"Hosted OCR {action}: {detail}.")
            if response.status_code >= 400:
                raise ExtractionError(
                    f"Hosted OCR {action} returned HTTP {response.status_code}: "
                    f"{response.text[:200]}"
                )
            return response
        raise AssertionError("unreachable")  # pragma: no cover

    def _submit(self, client: httpx.Client, document: bytes) -> str:
        response = self._request(
            lambda: client.post(
                "/api/v1/convert",
                files={"file": ("resume.pdf", document, "application/pdf")},
                data={
                    "output_format": "html",
                    "mode": self._mode,
                    "paginate": "true",
                    "disable_image_extraction": "true",
                    "disable_image_captions": "true",
                },
            ),
            "submission",
        )
        body: dict[str, Any] = response.json()
        request_id = body.get("request_id")
        if not body.get("success") or not request_id:
            raise ExtractionError(f"Hosted OCR rejected the document: {body.get('error')}")
        return str(request_id)

    def _wait(self, client: httpx.Client, request_id: str) -> dict[str, Any]:
        # Poll a URL we build ourselves, so the key is only ever sent to the configured host.
        deadline = time.monotonic() + self._timeout
        while True:
            response = self._request(lambda: client.get(f"/api/v1/convert/{request_id}"), "poll")
            body: dict[str, Any] = response.json()
            if body.get("status") in ("complete", "failed"):
                return body
            if time.monotonic() > deadline:
                raise ExtractionError("Hosted OCR timed out.")
            self._sleep(self._poll_interval)

    def _delete(self, client: httpx.Client, request_id: str) -> None:
        try:
            client.delete(f"/api/v1/results/{request_id}", timeout=10.0)
        except httpx.HTTPError:
            logger.warning("Could not delete hosted OCR result %s", request_id)
