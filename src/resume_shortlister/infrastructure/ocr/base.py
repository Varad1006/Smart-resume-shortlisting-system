"""OCR engine interface used by the document extractor."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from PIL import Image

from resume_shortlister.domain.models import ExtractionMethod


@dataclass(frozen=True, slots=True)
class OcrOutput:
    texts: list[str]  # one entry per input image, same order
    method: ExtractionMethod  # engine that actually produced the text


@dataclass(frozen=True, slots=True)
class OcrEngineStatus:
    name: str
    available: bool
    detail: str


class OcrEngine(Protocol):
    def is_available(self) -> bool: ...

    def recognize(self, images: Sequence[Image.Image]) -> OcrOutput:
        """Raise ExtractionError when recognition fails."""
        ...

    def status(self) -> list[OcrEngineStatus]: ...
