"""REST API for shortlist runs."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, Form, Query, Response, UploadFile, status

from resume_shortlister.api.dependencies import ServiceDep
from resume_shortlister.api.schemas import (
    ExtractionResponse,
    RunCreatedResponse,
    RunDetailResponse,
    RunListResponse,
    RunSummaryOut,
)
from resume_shortlister.api.uploads import read_uploads
from resume_shortlister.application.dto import ShortlistOptions
from resume_shortlister.domain.profiles import extract_profile_handles

router = APIRouter(prefix="/api/v1", tags=["shortlists"])


@router.post(
    "/shortlists",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=RunCreatedResponse,
    summary="Start a shortlisting run",
    description=(
        "Upload a job description and resumes (PDF, DOCX, images or TXT). Processing runs "
        "in the background; poll `status_url` until `status` is `completed` or `failed`."
    ),
)
async def create_shortlist(
    service: ServiceDep,
    job_description: Annotated[str, Form(description="Full job description text.")],
    files: Annotated[list[UploadFile], File(description="Resume files.")],
    shortlist_size: Annotated[int | None, Form(ge=1, le=50)] = None,
    include_social: Annotated[
        bool, Form(description="Add a bonus for public GitHub/LeetCode/CodeChef activity.")
    ] = False,
    include_insights: Annotated[
        bool,
        Form(
            description="Ask the configured LLM for summaries of the top shortlisted "
            "candidates (sends their redacted resume text to the LLM provider)."
        ),
    ] = False,
) -> RunCreatedResponse:
    documents = await read_uploads(files, service.limits)
    run = await service.submit(
        job_description,
        documents,
        ShortlistOptions(
            shortlist_size=shortlist_size,
            include_social=include_social,
            include_insights=include_insights,
        ),
    )
    return RunCreatedResponse.of(run)


@router.get("/shortlists", response_model=RunListResponse, summary="List past runs")
async def list_shortlists(
    service: ServiceDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> RunListResponse:
    items, total = await service.list_runs(limit=limit, offset=offset)
    return RunListResponse(
        items=[RunSummaryOut.of(item) for item in items], total=total, limit=limit, offset=offset
    )


@router.get(
    "/shortlists/{run_id}", response_model=RunDetailResponse, summary="Run status and results"
)
async def get_shortlist(run_id: str, service: ServiceDep) -> RunDetailResponse:
    return RunDetailResponse.of(await service.get_run(run_id))


@router.delete(
    "/shortlists/{run_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a run (cancels it if still processing)",
)
async def delete_shortlist(run_id: str, service: ServiceDep) -> Response:
    await service.delete_run(run_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/extract",
    response_model=ExtractionResponse,
    summary="Extract text from one resume (check OCR quality)",
)
async def extract_document(
    service: ServiceDep, file: Annotated[UploadFile, File(description="One resume file.")]
) -> ExtractionResponse:
    (document,) = await read_uploads([file], service.limits)
    extracted, language = await service.extract(document)
    profiles = extract_profile_handles(extracted.text, extracted.links)
    return ExtractionResponse.of(extracted, language, profiles)
