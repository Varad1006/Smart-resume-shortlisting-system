"""Server-rendered pages (Jinja2). All dynamic text is auto-escaped."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from resume_shortlister.api.dependencies import ContainerDep, ServiceDep
from resume_shortlister.api.presenters import register
from resume_shortlister.bootstrap import llm_status
from resume_shortlister.domain.errors import NotFoundError

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=TEMPLATES_DIR)
register(templates.env)

router = APIRouter(include_in_schema=False)
HISTORY_PAGE_SIZE = 20


@router.get("/", response_class=HTMLResponse)
async def index(request: Request, container: ContainerDep) -> HTMLResponse:
    settings = container.settings
    weight_sum = settings.semantic_weight + settings.coverage_weight
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "limits": container.service.limits,
            "default_shortlist_size": settings.default_shortlist_size,
            "max_file_mb": settings.max_file_mb,
            "social_max_bonus": settings.social_max_bonus,
            "coverage_pct": round(100 * settings.coverage_weight / weight_sum),
            "semantic_pct": round(100 * settings.semantic_weight / weight_sum),
            "llm": llm_status(settings, container.service.summarizer_name),
        },
    )


@router.get("/shortlists", response_class=HTMLResponse)
async def history(
    request: Request, service: ServiceDep, page: Annotated[int, Query(ge=1)] = 1
) -> HTMLResponse:
    items, total = await service.list_runs(
        limit=HISTORY_PAGE_SIZE, offset=(page - 1) * HISTORY_PAGE_SIZE
    )
    return templates.TemplateResponse(
        request,
        "history.html",
        {
            "items": items,
            "total": total,
            "page": page,
            "has_next": page * HISTORY_PAGE_SIZE < total,
        },
    )


@router.get("/shortlists/{run_id}", response_class=HTMLResponse)
async def run_page(request: Request, run_id: str, service: ServiceDep) -> HTMLResponse:
    try:
        run = await service.get_run(run_id)
    except NotFoundError:
        return templates.TemplateResponse(
            request,
            "error.html",
            {
                "title": "Shortlist not found",
                "message": "It may have been deleted. Start a new one from the home page.",
            },
            status_code=404,
        )
    requirements = {r.id: r for r in run.result.requirements} if run.result else {}
    return templates.TemplateResponse(
        request, "run.html", {"run": run, "result": run.result, "requirements": requirements}
    )
