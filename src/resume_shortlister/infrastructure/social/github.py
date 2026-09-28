"""GitHub REST API client (public profile statistics)."""

from __future__ import annotations

import httpx

from resume_shortlister.domain.errors import SocialLookupError
from resume_shortlister.domain.models import GitHubStats

API_BASE = "https://api.github.com"


class GitHubClient:
    def __init__(self, http: httpx.AsyncClient, token: str | None = None) -> None:
        self._http = http
        self._headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if token:
            self._headers["Authorization"] = f"Bearer {token}"

    async def fetch(self, username: str) -> GitHubStats | None:
        """Return stats, or None when the user does not exist."""
        user = await self._http.get(f"{API_BASE}/users/{username}", headers=self._headers)
        if user.status_code == 404:
            return None
        if user.status_code in (403, 429):
            raise SocialLookupError("GitHub rate limit reached (set GITHUB_TOKEN to raise it).")
        user.raise_for_status()
        profile = user.json()

        repos = await self._http.get(
            f"{API_BASE}/users/{username}/repos",
            params={"per_page": 100, "type": "owner", "sort": "pushed"},
            headers=self._headers,
        )
        stars = 0
        if repos.status_code == 200:
            stars = sum(
                int(repo.get("stargazers_count", 0))
                for repo in repos.json()
                if not repo.get("fork")
            )
        return GitHubStats(
            username=profile.get("login", username),
            public_repos=int(profile.get("public_repos", 0)),
            followers=int(profile.get("followers", 0)),
            total_stars=stars,
        )
