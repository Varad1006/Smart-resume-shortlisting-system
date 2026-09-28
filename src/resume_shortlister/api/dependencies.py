"""FastAPI dependencies: access to the composition root without globals."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from resume_shortlister.application.shortlist_service import ShortlistService
from resume_shortlister.bootstrap import Container


def get_container(request: Request) -> Container:
    return request.app.state.container


def get_service(container: Annotated[Container, Depends(get_container)]) -> ShortlistService:
    return container.service


ContainerDep = Annotated[Container, Depends(get_container)]
ServiceDep = Annotated[ShortlistService, Depends(get_service)]
