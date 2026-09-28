"""CodeChef profile scraper (best effort: CodeChef has no public API)."""

from __future__ import annotations

import re

import httpx

from resume_shortlister.domain.errors import SocialLookupError
from resume_shortlister.domain.models import CodeChefStats

PROFILE_URL = "https://www.codechef.com/users/{username}"

_RATING = re.compile(r'class="rating-number"[^>]*>\s*(\d{1,4})')
_STARS = re.compile(r"(\d)\s*(?:★|&#9733;|&starf;)")
_SOLVED = re.compile(r"Total\s+Problems\s+Solved\s*:?\s*(\d+)", re.IGNORECASE)


def _int(pattern: re.Pattern[str], html: str) -> int | None:
    match = pattern.search(html)
    return int(match.group(1)) if match else None


def parse_codechef_profile(html: str, username: str) -> CodeChefStats | None:
    rating = _int(_RATING, html)
    solved = _int(_SOLVED, html)
    if rating is None and solved is None:
        return None
    return CodeChefStats(
        username=username, rating=rating, stars=_int(_STARS, html), problems_solved=solved
    )


class CodeChefClient:
    def __init__(self, http: httpx.AsyncClient) -> None:
        self._http = http

    async def fetch(self, username: str) -> CodeChefStats | None:
        response = await self._http.get(
            PROFILE_URL.format(username=username), follow_redirects=False
        )
        if response.status_code in (301, 302, 303, 307, 308, 404):
            return None  # CodeChef redirects unknown users to the home page
        if response.status_code in (403, 429):
            raise SocialLookupError("CodeChef refused the request (rate limited or blocked).")
        response.raise_for_status()
        return parse_codechef_profile(response.text, username)
