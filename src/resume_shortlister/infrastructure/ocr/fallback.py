"""Try OCR engines in order of preference (e.g. Chandra first, Tesseract as fallback)."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from PIL import Image

from resume_shortlister.domain.errors import ExtractionError, OcrUnavailableError
from resume_shortlister.infrastructure.ocr.base import OcrEngine, OcrEngineStatus, OcrOutput

logger = logging.getLogger(__name__)


class FallbackOcrEngine:
    def __init__(self, engines: Sequence[OcrEngine]) -> None:
        if not engines:
            raise ValueError("at least one OCR engine is required")
        self._engines = list(engines)

    def is_available(self) -> bool:
        return any(engine.is_available() for engine in self._engines)

    def status(self) -> list[OcrEngineStatus]:
        return [status for engine in self._engines for status in engine.status()]

    def recognize(self, images: Sequence[Image.Image]) -> OcrOutput:
        failures: list[str] = []
        for engine in self._engines:
            if not engine.is_available():
                continue
            try:
                return engine.recognize(images)
            except ExtractionError as exc:
                logger.warning("OCR engine %s failed, trying the next one: %s", engine, exc)
                failures.append(str(exc))
        if failures:
            raise ExtractionError("All OCR engines failed: " + " | ".join(failures))
        raise OcrUnavailableError(
            "This file needs OCR, but no OCR engine is available. Start the Chandra server "
            "(docker compose --profile gpu up) or install Tesseract."
        )
