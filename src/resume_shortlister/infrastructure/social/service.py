"""SocialStatsProvider adapter: concurrent, cached lookups across all platforms."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from resume_shortlister.domain.errors import SocialLookupError
from resume_shortlister.domain.models import ProfileHandles, SocialSnapshot
from resume_shortlister.domain.scoring import social_score
from resume_shortlister.infrastructure.social.codechef import CodeChefClient
from resume_shortlister.infrastructure.social.github import GitHubClient
from resume_shortlister.infrastructure.social.leetcode import LeetCodeClient

logger = logging.getLogger(__name__)

PLATFORM_LABELS = {"github": "GitHub", "leetcode": "LeetCode", "codechef": "CodeChef"}
USER_AGENT = (
    "resume-shortlister/1.0 (+https://github.com/Varad1006/Smart-resume-shortlisting-system)"
)


def build_http_client(timeout_seconds: float) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=httpx.Timeout(timeout_seconds),
        headers={"User-Agent": USER_AGENT},
        limits=httpx.Limits(max_connections=20),
    )


class SocialProfileService:
    """Fetches GitHub/LeetCode/CodeChef stats concurrently, caching results for a while."""

    def __init__(
        self,
        github: GitHubClient,
        leetcode: LeetCodeClient,
        codechef: CodeChefClient,
        *,
        cache_ttl_seconds: float = 3600.0,
    ) -> None:
        self._fetchers: dict[str, Callable[[str], Awaitable[Any]]] = {
            "github": github.fetch,
            "leetcode": leetcode.fetch,
            "codechef": codechef.fetch,
        }
        self._ttl = cache_ttl_seconds
        self._cache: dict[tuple[str, str], tuple[float, Any]] = {}

    async def _lookup(self, platform: str, username: str) -> Any:
        key = (platform, username.lower())
        cached = self._cache.get(key)
        if cached and time.monotonic() - cached[0] < self._ttl:
            return cached[1]
        value = await self._fetchers[platform](username)
        self._cache[key] = (time.monotonic(), value)
        return value

    async def fetch(self, handles: ProfileHandles) -> SocialSnapshot:
        wanted = {
            platform: username
            for platform, username in (
                ("github", handles.github),
                ("leetcode", handles.leetcode),
                ("codechef", handles.codechef),
            )
            if username
        }
        outcomes = await asyncio.gather(
            *(self._lookup(p, u) for p, u in wanted.items()), return_exceptions=True
        )

        stats: dict[str, Any] = {}
        errors: list[str] = []
        for (platform, username), outcome in zip(wanted.items(), outcomes, strict=True):
            label = PLATFORM_LABELS[platform]
            if isinstance(outcome, SocialLookupError):
                errors.append(f"{label}: {outcome}")
            elif isinstance(outcome, httpx.HTTPError):
                errors.append(f"{label}: request failed ({type(outcome).__name__}).")
            elif isinstance(outcome, BaseException):
                logger.warning("%s lookup for %s failed: %r", platform, username, outcome)
                errors.append(f"{label}: lookup failed.")
            elif outcome is None:
                errors.append(f"{label}: profile '{username}' not found.")
            else:
                stats[platform] = outcome

        github, leetcode, codechef = (stats.get(k) for k in ("github", "leetcode", "codechef"))
        return SocialSnapshot(
            github=github,
            leetcode=leetcode,
            codechef=codechef,
            score=social_score(github, leetcode, codechef),
            errors=tuple(errors),
        )
