"""Map domain/application errors to HTTP responses."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from resume_shortlister.domain.errors import (
    ExtractionError,
    InvalidInputError,
    NotFoundError,
    PayloadTooLargeError,
    ShortlisterError,
    UnsupportedDocumentError,
)

_STATUS: list[tuple[type[ShortlisterError], int]] = [
    (PayloadTooLargeError, 413),
    (InvalidInputError, 422),
    (NotFoundError, 404),
    (UnsupportedDocumentError, 415),
    (ExtractionError, 422),
]


def status_for(exc: ShortlisterError) -> int:
    for error_type, status in _STATUS:
        if isinstance(exc, error_type):
            return status
    return 400


def install_error_handlers(app: FastAPI) -> None:
    async def handle(request: Request, exc: Exception) -> JSONResponse:
        assert isinstance(exc, ShortlisterError)
        return JSONResponse(status_code=status_for(exc), content={"detail": str(exc)})

    app.add_exception_handler(ShortlisterError, handle)
