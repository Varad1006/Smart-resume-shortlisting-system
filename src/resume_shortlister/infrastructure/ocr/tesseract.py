"""Tesseract OCR adapter: lightweight CPU fallback when Chandra is not reachable."""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence

import pytesseract
from PIL import Image, ImageOps

from resume_shortlister.domain.errors import ExtractionError
from resume_shortlister.domain.models import ExtractionMethod
from resume_shortlister.infrastructure.ocr.base import OcrEngineStatus, OcrOutput

logger = logging.getLogger(__name__)

MIN_SIDE_PX = 1500  # upscale small scans; Tesseract works best around 300 DPI


class TesseractOcrEngine:
    method = ExtractionMethod.TESSERACT

    def __init__(self, languages: str = "eng", command: str | None = None) -> None:
        if command:
            pytesseract.pytesseract.tesseract_cmd = command
        self._requested = [lang for lang in languages.replace(",", "+").split("+") if lang]
        self._lock = threading.Lock()
        self._checked = False
        self._available = False
        self._languages = "eng"
        self._detail = "not checked yet"

    def _check(self) -> None:
        with self._lock:
            if self._checked:
                return
            self._checked = True
            try:
                version = pytesseract.get_tesseract_version()
                installed = set(pytesseract.get_languages(config=""))
            except (pytesseract.TesseractNotFoundError, OSError) as exc:
                self._detail = f"tesseract binary not found ({type(exc).__name__})"
                return
            usable = [lang for lang in self._requested if lang in installed]
            missing = sorted(set(self._requested) - installed)
            if not usable:
                self._detail = f"none of the language packs {self._requested} are installed"
                return
            if missing:
                logger.warning("Tesseract language packs not installed: %s", missing)
            self._languages = "+".join(usable)
            self._available = True
            self._detail = f"tesseract {version}, languages {self._languages}"

    def is_available(self) -> bool:
        self._check()
        return self._available

    def status(self) -> list[OcrEngineStatus]:
        self._check()
        return [OcrEngineStatus(name="tesseract", available=self._available, detail=self._detail)]

    @staticmethod
    def _prepare(image: Image.Image) -> Image.Image:
        image = ImageOps.exif_transpose(image).convert("L")
        shortest = min(image.size)
        if 0 < shortest < MIN_SIDE_PX:
            scale = MIN_SIDE_PX / shortest
            image = image.resize(
                (round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS
            )
        return image

    def recognize(self, images: Sequence[Image.Image]) -> OcrOutput:
        if not self.is_available():
            raise ExtractionError(f"Tesseract is not available: {self._detail}")
        try:
            texts = [
                pytesseract.image_to_string(self._prepare(image), lang=self._languages)
                for image in images
            ]
        except (pytesseract.TesseractError, RuntimeError, OSError) as exc:
            raise ExtractionError(f"Tesseract OCR failed: {exc}") from exc
        return OcrOutput(texts=texts, method=self.method)
