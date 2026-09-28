"""FastAPI application factory (``uvicorn resume_shortlister.api.app:create_app --factory``)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from resume_shortlister import __version__
from resume_shortlister.api.errors import install_error_handlers
from resume_shortlister.api.routes import shortlists, system, ui
from resume_shortlister.bootstrap import Container, build_container
from resume_shortlister.config import Settings
from resume_shortlister.logging_config import configure_logging

STATIC_DIR = Path(__file__).resolve().parent / "static"
MULTIPART_OVERHEAD = 1024 * 1024  # form fields + boundaries on top of the file bytes

DESCRIPTION = """
Rank resumes against a job description with explainable, requirement-level scores.

* **Extraction**: PDF text layer, DOCX, or OCR (Chandra, falling back to Tesseract).
* **Stage 1**: multilingual semantic retrieval over resume passages.
* **Stage 2**: cross-encoder matching of every job requirement, with evidence.
* **Optional**: bonus for public GitHub / LeetCode / CodeChef activity.
"""


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    if container is None:
        settings = settings or Settings()
        configure_logging(settings.log_level)
        container = build_container(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await container.startup()
        try:
            yield
        finally:
            await container.shutdown()

    app = FastAPI(
        title="Smart Resume Shortlisting System",
        version=__version__,
        description=DESCRIPTION,
        lifespan=lifespan,
    )
    app.state.container = container

    max_body = container.service.limits.max_total_bytes + MULTIPART_OVERHEAD

    @app.middleware("http")
    async def reject_oversized_bodies(request: Request, call_next):  # type: ignore[no-untyped-def]
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > max_body:
            return JSONResponse(status_code=413, content={"detail": "Request body too large."})
        response: Response = await call_next(request)
        return response

    install_error_handlers(app)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    app.include_router(system.router)
    app.include_router(shortlists.router)
    app.include_router(ui.router)
    return app
