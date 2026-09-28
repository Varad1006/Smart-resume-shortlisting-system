import json

import httpx
import pytest

from resume_shortlister.domain.errors import SocialLookupError
from resume_shortlister.domain.models import ProfileHandles
from resume_shortlister.infrastructure.social.codechef import CodeChefClient, parse_codechef_profile
from resume_shortlister.infrastructure.social.github import GitHubClient
from resume_shortlister.infrastructure.social.leetcode import (
    LeetCodeClient,
    parse_leetcode_response,
)
from resume_shortlister.infrastructure.social.service import SocialProfileService

pytestmark = pytest.mark.anyio

LEETCODE_PAYLOAD = {
    "data": {
        "matchedUser": {
            "submitStatsGlobal": {
                "acSubmissionNum": [
                    {"difficulty": "All", "count": 150},
                    {"difficulty": "Easy", "count": 80},
                    {"difficulty": "Medium", "count": 60},
                    {"difficulty": "Hard", "count": 10},
                ]
            }
        }
    }
}
CODECHEF_HTML = """
<div class="rating-header"><div class="rating-number">1843</div>
<span class="rating">4&#9733;</span></div>
<h3>Total Problems Solved: 212</h3>
"""


def test_leetcode_total_uses_all_entry_without_double_counting():
    stats = parse_leetcode_response(LEETCODE_PAYLOAD, "coder")
    assert (stats.solved_total, stats.easy, stats.medium, stats.hard) == (150, 80, 60, 10)


def test_leetcode_unknown_user():
    assert parse_leetcode_response({"data": {"matchedUser": None}}, "nobody") is None


def test_codechef_profile_parsing():
    stats = parse_codechef_profile(CODECHEF_HTML, "chef")
    assert (stats.rating, stats.stars, stats.problems_solved) == (1843, 4, 212)
    assert parse_codechef_profile("<html>maintenance</html>", "chef") is None


def github_handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/users/ghost":
        return httpx.Response(404)
    if path == "/users/limited":
        return httpx.Response(403, json={"message": "API rate limit exceeded"})
    if path == "/users/jane":
        assert request.headers["Authorization"] == "Bearer tkn"
        return httpx.Response(200, json={"login": "Jane", "public_repos": 12, "followers": 40})
    if path == "/users/jane/repos":
        return httpx.Response(
            200,
            json=[
                {"stargazers_count": 30, "fork": False},
                {"stargazers_count": 5000, "fork": True},  # forks don't count
                {"stargazers_count": 12, "fork": False},
            ],
        )
    raise AssertionError(f"unexpected request {request.url}")


async def test_github_stats_exclude_forked_stars():
    async with httpx.AsyncClient(transport=httpx.MockTransport(github_handler)) as http:
        client = GitHubClient(http, token="tkn")
        stats = await client.fetch("jane")
        assert (stats.username, stats.public_repos, stats.followers, stats.total_stars) == (
            "Jane",
            12,
            40,
            42,
        )
        assert await client.fetch("ghost") is None
        with pytest.raises(SocialLookupError, match="rate limit"):
            await client.fetch("limited")


async def test_service_combines_platforms_caches_and_reports_errors():
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.host + request.url.path)
        if request.url.host == "api.github.com":
            return github_handler(request)
        if request.url.host == "leetcode.com":
            assert json.loads(request.content)["variables"] == {"username": "coder"}
            return httpx.Response(200, json=LEETCODE_PAYLOAD)
        return httpx.Response(302, headers={"Location": "https://www.codechef.com/"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        service = SocialProfileService(
            GitHubClient(http, token="tkn"), LeetCodeClient(http), CodeChefClient(http)
        )
        handles = ProfileHandles(github="jane", leetcode="coder", codechef="missing_chef")
        snapshot = await service.fetch(handles)
        again = await service.fetch(handles)

    assert snapshot.github.total_stars == 42
    assert snapshot.leetcode.solved_total == 150
    assert snapshot.codechef is None
    assert snapshot.errors == ("CodeChef: profile 'missing_chef' not found.",)
    assert 0 < snapshot.score <= 100
    assert again == snapshot
    assert len(calls) == 4  # second fetch served from cache


async def test_service_without_any_data_has_no_score():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timeout", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        service = SocialProfileService(
            GitHubClient(http), LeetCodeClient(http), CodeChefClient(http)
        )
        snapshot = await service.fetch(ProfileHandles(github="jane"))
    assert snapshot.score is None
    assert snapshot.errors == ("GitHub: request failed (ConnectTimeout).",)
