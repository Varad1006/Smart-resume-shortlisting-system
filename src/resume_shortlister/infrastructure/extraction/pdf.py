"""PDF reading with pypdfium2 (permissive licence; replaces AGPL PyMuPDF and PyPDF2).

Each page uses its embedded text layer when it has one; pages without text (scans,
photos of handwritten resumes) are rendered to images for OCR. Hyperlink annotations are
collected too: resumes often hide profile URLs behind words like "GitHub".

PDFium is not thread-safe: two worker threads inside it at once crash the whole process
(SIGSEGV/abort). Every PDFium call in this module therefore runs under ``_PDFIUM_LOCK``;
the slow part, OCR of the rendered pages, happens after the lock is released.
"""

from __future__ import annotations

import ctypes
import logging
import threading
from dataclasses import dataclass

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
from PIL import Image

from resume_shortlister.domain.errors import ExtractionError

logger = logging.getLogger(__name__)

MIN_RENDER_SIDE_PX = 1024
_PDFIUM_LOCK = threading.Lock()


@dataclass(slots=True)
class PdfPage:
    index: int
    text: str | None  # None until OCR fills it in
    image: Image.Image | None  # rendered only for pages that need OCR


@dataclass(slots=True)
class PdfContent:
    pages: list[PdfPage]
    total_pages: int
    links: list[str]


def _render(page: pdfium.PdfPage, dpi: int) -> Image.Image:
    width, height = page.get_size()
    scale = dpi / 72
    shortest = min(width, height)
    if shortest > 0 and shortest * scale < MIN_RENDER_SIDE_PX:
        scale = MIN_RENDER_SIDE_PX / shortest
    bitmap = page.render(scale=scale)
    try:
        return bitmap.to_pil().convert("RGB")  # convert() copies out of the bitmap buffer
    finally:
        bitmap.close()


def _page_links(pdf: pdfium.PdfDocument, page: pdfium.PdfPage) -> list[str]:
    urls: list[str] = []
    position = ctypes.c_int(0)
    link = pdfium_c.FPDF_LINK()
    while pdfium_c.FPDFLink_Enumerate(page.raw, ctypes.byref(position), ctypes.byref(link)):
        action = pdfium_c.FPDFLink_GetAction(link)
        if not action or pdfium_c.FPDFAction_GetType(action) != pdfium_c.PDFACTION_URI:
            continue
        size = pdfium_c.FPDFAction_GetURIPath(pdf.raw, action, None, 0)
        if size <= 0:
            continue
        buffer = ctypes.create_string_buffer(size)
        pdfium_c.FPDFAction_GetURIPath(pdf.raw, action, buffer, size)
        url = buffer.value.decode("utf-8", errors="replace").strip()
        if url:
            urls.append(url)
    return urls


def read_pdf(content: bytes, *, max_pages: int, min_chars_per_page: int, dpi: int) -> PdfContent:
    with _PDFIUM_LOCK:
        return _read_pdf(
            content, max_pages=max_pages, min_chars_per_page=min_chars_per_page, dpi=dpi
        )


def _read_pdf(content: bytes, *, max_pages: int, min_chars_per_page: int, dpi: int) -> PdfContent:
    try:
        pdf = pdfium.PdfDocument(content)
    except pdfium.PdfiumError as exc:
        message = str(exc)
        if "password" in message.lower():
            raise ExtractionError("This PDF is password-protected.") from exc
        raise ExtractionError(f"Could not open this PDF ({message}).") from exc

    try:
        total_pages = len(pdf)
        pages: list[PdfPage] = []
        links: list[str] = []
        for index in range(min(total_pages, max_pages)):
            page = pdf[index]
            try:
                textpage = page.get_textpage()
                try:
                    text = textpage.get_text_range()
                finally:
                    textpage.close()
                if len(text.strip()) >= min_chars_per_page:
                    pages.append(PdfPage(index=index, text=text, image=None))
                else:
                    pages.append(PdfPage(index=index, text=None, image=_render(page, dpi)))
                try:
                    links.extend(_page_links(pdf, page))
                except (OSError, ValueError, ctypes.ArgumentError):  # pragma: no cover - defensive
                    logger.debug("Could not read link annotations on page %d", index)
            finally:
                page.close()
        return PdfContent(pages=pages, total_pages=total_pages, links=links)
    finally:
        pdf.close()
