"""Liveness and status endpoints."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter

from resume_shortlister import __version__
from resume_shortlister.api.dependencies import ContainerDep
from resume_shortlister.api.schemas import StatusResponse

router = APIRouter(tags=["system"])


@router.get("/health", summary="Liveness probe")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/api/v1/status", response_model=StatusResponse, summary="OCR and model status")
async def status(container: ContainerDep) -> StatusResponse:
    details = await asyncio.to_thread(container.status)  # may probe the Chandra server
    return StatusResponse(version=__version__, **details)
