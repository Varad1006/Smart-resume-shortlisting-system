"""DocumentExtractor adapter: PDF, DOCX, images and plain text -> ExtractedDocument."""

from __future__ import annotations

import io
import itertools
import logging
import unicodedata
from enum import Enum

from PIL import Image, ImageOps, ImageSequence, UnidentifiedImageError

from resume_shortlister.domain.errors import (
    ExtractionError,
    OcrUnavailableError,
    UnsupportedDocumentError,
)
from resume_shortlister.domain.models import ExtractedDocument, ExtractionMethod
from resume_shortlister.infrastructure.extraction.docx import is_docx, read_docx
from resume_shortlister.infrastructure.extraction.pdf import read_pdf
from resume_shortlister.infrastructure.ocr.base import OcrEngine

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = "PDF, DOCX, PNG, JPG, WEBP, TIFF, BMP or TXT"
# Control, surrogate, unassigned and private-use characters (icon fonts in PDFs emit
# private-use glyphs for things like the GitHub or phone icons).
_DROPPED_CATEGORIES = frozenset({"Cc", "Cs", "Cn", "Co"})


class DocumentKind(Enum):
    PDF = "pdf"
    DOCX = "docx"
    IMAGE = "image"
    TEXT = "text"


def _looks_like_text(content: bytes) -> bool:
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        return False
    sample = text[:4000]
    printable = sum(ch.isprintable() or ch in "\n\r\t" for ch in sample)
    return bool(sample) and printable / len(sample) > 0.95


def detect_kind(content: bytes) -> DocumentKind:
    """Identify the format from the file bytes, never from the (untrusted) extension."""
    if b"%PDF-" in content[:1024]:
        return DocumentKind.PDF
    if is_docx(content):
        return DocumentKind.DOCX
    try:
        with Image.open(io.BytesIO(content)):
            return DocumentKind.IMAGE
    except (UnidentifiedImageError, OSError):
        pass
    if _looks_like_text(content):
        return DocumentKind.TEXT
    raise UnsupportedDocumentError(f"Unsupported file format. Upload {SUPPORTED_FORMATS}.")


def normalize_text(text: str) -> str:
    """NFKC (fixes PDF ligatures like "ﬁ"), drop control characters, tidy whitespace."""
    text = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(
        ch for ch in text if ch in "\n\t" or unicodedata.category(ch) not in _DROPPED_CATEGORIES
    )
    lines: list[str] = []
    blank_run = 0
    for raw_line in text.split("\n"):
        line = " ".join(raw_line.split())
        blank_run = blank_run + 1 if not line else 0
        if blank_run <= 1:
            lines.append(line)
    return "\n".join(lines).strip()


def _ordered(methods: set[ExtractionMethod]) -> tuple[ExtractionMethod, ...]:
    return tuple(m for m in ExtractionMethod if m in methods)


def _unique(items: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(items))


class DocumentExtractor:
    def __init__(
        self,
        ocr: OcrEngine | None,
        *,
        max_pages: int = 10,
        min_chars_per_page: int = 40,
        ocr_dpi: int = 200,
    ) -> None:
        self._ocr = ocr
        self._max_pages = max_pages
        self._min_chars_per_page = min_chars_per_page
        self._ocr_dpi = ocr_dpi

    def extract(self, filename: str, content: bytes) -> ExtractedDocument:
        kind = detect_kind(content)
        if kind is DocumentKind.PDF:
            return self._from_pdf(filename, content)
        if kind is DocumentKind.DOCX:
            text, links = read_docx(content)
            return ExtractedDocument(
                filename=filename,
                text=normalize_text(text),
                methods=(ExtractionMethod.DOCX,),
                links=_unique(links),
            )
        if kind is DocumentKind.IMAGE:
            return self._from_image(filename, content)
        return ExtractedDocument(
            filename=filename,
            text=normalize_text(content.decode("utf-8-sig", errors="replace")),
            methods=(ExtractionMethod.PLAIN_TEXT,),
        )

    def _require_ocr(self) -> OcrEngine:
        if self._ocr is None:
            raise OcrUnavailableError("This file is a scan, and reading scans is turned off.")
        return self._ocr

    def _from_pdf(self, filename: str, content: bytes) -> ExtractedDocument:
        pdf = read_pdf(
            content,
            max_pages=self._max_pages,
            min_chars_per_page=self._min_chars_per_page,
            dpi=self._ocr_dpi,
        )
        warnings: list[str] = []
        if pdf.total_pages > self._max_pages:
            warnings.append(
                f"Only the first {self._max_pages} of {pdf.total_pages} pages were read."
            )

        methods: set[ExtractionMethod] = set()
        scanned = [page for page in pdf.pages if page.text is None]
        if len(scanned) < len(pdf.pages):
            methods.add(ExtractionMethod.TEXT_LAYER)
        if scanned:
            try:
                output = self._require_ocr().recognize([page.image for page in scanned])  # type: ignore[misc]
            except ExtractionError as exc:
                if not methods:  # nothing readable without OCR
                    raise
                logger.warning("Scanned pages of a PDF could not be read: %s", exc)
                warnings.append(f"{len(scanned)} scanned page(s) could not be read.")
            else:
                for page, text in zip(scanned, output.texts, strict=True):
                    page.text = text
                methods.add(output.method)
            finally:
                for page in scanned:
                    page.image = None  # release rendered bitmaps early

        text = "\n\n".join(page.text for page in pdf.pages if page.text)
        return ExtractedDocument(
            filename=filename,
            text=normalize_text(text),
            methods=_ordered(methods),
            pages=len(pdf.pages),
            links=_unique(pdf.links),
            warnings=tuple(warnings),
        )

    def _from_image(self, filename: str, content: bytes) -> ExtractedDocument:
        try:
            with Image.open(io.BytesIO(content)) as image:
                frames = [
                    ImageOps.exif_transpose(frame.copy()).convert("RGB")
                    for frame in itertools.islice(ImageSequence.Iterator(image), self._max_pages)
                ]
        except (OSError, Image.DecompressionBombError) as exc:
            raise ExtractionError(f"Could not read this image ({type(exc).__name__}).") from exc

        output = self._require_ocr().recognize(frames)
        return ExtractedDocument(
            filename=filename,
            text=normalize_text("\n\n".join(output.texts)),
            methods=(output.method,),
            pages=len(frames),
        )
