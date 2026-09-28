"""LeetCode public GraphQL client."""

from __future__ import annotations

from typing import Any

import httpx

from resume_shortlister.domain.errors import SocialLookupError
from resume_shortlister.domain.models import LeetCodeStats

GRAPHQL_URL = "https://leetcode.com/graphql"
QUERY = """
query userProblemsSolved($username: String!) {
  matchedUser(username: $username) {
    submitStatsGlobal { acSubmissionNum { difficulty count } }
  }
}
"""


def parse_leetcode_response(payload: dict[str, Any], username: str) -> LeetCodeStats | None:
    """Parse the GraphQL response.

    ``acSubmissionNum`` holds an "All" entry *plus* one per difficulty; summing every entry
    (as the original project did) double-counts. Use "All" for the total.
    """
    user = (payload.get("data") or {}).get("matchedUser")
    if not user:
        return None
    stats = user.get("submitStatsGlobal") or user.get("submitStats") or {}
    counts = {
        str(item.get("difficulty", "")).lower(): int(item.get("count", 0))
        for item in stats.get("acSubmissionNum", [])
    }
    easy, medium, hard = counts.get("easy", 0), counts.get("medium", 0), counts.get("hard", 0)
    return LeetCodeStats(
        username=username,
        solved_total=counts.get("all", easy + medium + hard),
        easy=easy,
        medium=medium,
        hard=hard,
    )


class LeetCodeClient:
    def __init__(self, http: httpx.AsyncClient) -> None:
        self._http = http

    async def fetch(self, username: str) -> LeetCodeStats | None:
        response = await self._http.post(
            GRAPHQL_URL,
            json={
                "query": QUERY,
                "variables": {"username": username},
                "operationName": "userProblemsSolved",
            },
            headers={"Referer": f"https://leetcode.com/u/{username}/"},
        )
        if response.status_code in (403, 429):
            raise SocialLookupError("LeetCode refused the request (rate limited or blocked).")
        response.raise_for_status()
        return parse_leetcode_response(response.json(), username)
