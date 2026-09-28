"""Read multipart uploads defensively (size limits, sanitised filenames)."""

from __future__ import annotations

from pathlib import PurePosixPath

from fastapi import UploadFile

from resume_shortlister.application.dto import UploadedDocument, UploadLimits
from resume_shortlister.domain.errors import InvalidInputError, PayloadTooLargeError

MB = 1024 * 1024


def safe_filename(name: str | None) -> str:
    """Strip client-side paths (e.g. C:\\fakepath\\cv.pdf) and control characters."""
    base = PurePosixPath((name or "").replace("\\", "/")).name
    cleaned = "".join(ch for ch in base if ch.isprintable()).strip()
    return cleaned[:200] or "resume"


async def read_uploads(files: list[UploadFile], limits: UploadLimits) -> list[UploadedDocument]:
    if len(files) > limits.max_files:
        raise InvalidInputError(f"Too many files ({len(files)}); the limit is {limits.max_files}.")
    documents: list[UploadedDocument] = []
    total = 0
    for upload in files:
        name = safe_filename(upload.filename)
        content = await upload.read(limits.max_file_bytes + 1)
        if len(content) > limits.max_file_bytes:
            raise PayloadTooLargeError(f"{name} is larger than {limits.max_file_bytes // MB} MB.")
        total += len(content)
        if total > limits.max_total_bytes:
            raise PayloadTooLargeError(
                f"Uploads exceed the total limit of {limits.max_total_bytes // MB} MB."
            )
        documents.append(UploadedDocument(filename=name, content=content))
    return documents
