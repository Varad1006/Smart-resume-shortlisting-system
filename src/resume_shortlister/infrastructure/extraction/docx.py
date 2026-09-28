"""DOCX reading with python-docx, including text boxes, tables, headers and hyperlinks."""

from __future__ import annotations

import io
import zipfile

from docx import Document
from docx.opc.constants import RELATIONSHIP_TYPE
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph

from resume_shortlister.domain.errors import ExtractionError

# Word stores text boxes twice (modern + legacy VML fallback); read only the modern copy.
_FALLBACK_TAG = "{http://schemas.openxmlformats.org/markup-compatibility/2006}Fallback"


def is_docx(content: bytes) -> bool:
    if not content.startswith(b"PK\x03\x04"):
        return False
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            return "word/document.xml" in archive.namelist()
    except zipfile.BadZipFile:
        return False


def _paragraph_texts(element, parent) -> list[str]:  # type: ignore[no-untyped-def]
    texts: list[str] = []
    for paragraph in element.iter(qn("w:p")):
        if any(True for _ in paragraph.iterancestors(_FALLBACK_TAG)):
            continue
        text = Paragraph(paragraph, parent).text.strip()
        if text:
            texts.append(text)
    return texts


def _hyperlinks(part) -> list[str]:  # type: ignore[no-untyped-def]
    return [
        rel.target_ref
        for rel in part.rels.values()
        if rel.reltype == RELATIONSHIP_TYPE.HYPERLINK and rel.is_external
    ]


def read_docx(content: bytes) -> tuple[str, list[str]]:
    """Return (text, hyperlink targets)."""
    try:
        document = Document(io.BytesIO(content))
    except Exception as exc:  # python-docx raises a variety of parser errors
        raise ExtractionError(f"Could not open this DOCX file ({type(exc).__name__}).") from exc

    lines: list[str] = []
    links = _hyperlinks(document.part)
    seen_headers: set[str] = set()
    for section in document.sections:
        for header in (section.first_page_header, section.header):
            if header.is_linked_to_previous:
                continue
            for text in _paragraph_texts(header._element, header):
                if text not in seen_headers:
                    seen_headers.add(text)
                    lines.append(text)
            links.extend(_hyperlinks(header.part))

    lines.extend(_paragraph_texts(document.element.body, document))
    return "\n".join(lines), links
