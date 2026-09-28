"""Find GitHub / LeetCode / CodeChef usernames in resume text and embedded hyperlinks."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable

from resume_shortlister.domain.models import ProfileHandles

_GITHUB_NAME = r"[A-Za-z0-9](?:[A-Za-z0-9]|-(?=[A-Za-z0-9])){0,38}"
# The name must end here (no partial matches via backtracking) and must not be a domain,
# so "GitHub: github.com/x" never yields the username "github" (or "githu").
_NOT_A_DOMAIN = r"(?![A-Za-z0-9_-])(?!\.\w)(?!/)"

_GITHUB_URL = re.compile(
    rf"github\.com/\s?(?P<owner>{_GITHUB_NAME})(?P<repo>/[A-Za-z0-9._-]+)?", re.IGNORECASE
)
_GITHUB_PAGES = re.compile(rf"(?P<owner>{_GITHUB_NAME})\.github\.io", re.IGNORECASE)
_GITHUB_LABEL = re.compile(
    rf"\bgit\s?hub\s*(?:id|username|handle|profile)?\s*[:|\-–—]\s*@?(?P<user>{_GITHUB_NAME})"
    rf"{_NOT_A_DOMAIN}",
    re.IGNORECASE,
)
_LEETCODE_URL = re.compile(
    r"leetcode\.com/(?:u/|profile/)?\s?(?P<user>[A-Za-z0-9_-]{2,40})", re.IGNORECASE
)
_LEETCODE_LABEL = re.compile(
    rf"\bleet\s?code\s*(?:id|username|handle|profile)?\s*[:|\-–—]\s*@?"
    rf"(?P<user>[A-Za-z0-9_-]{{2,40}}){_NOT_A_DOMAIN}",
    re.IGNORECASE,
)
_CODECHEF_URL = re.compile(r"codechef\.com/users/\s?(?P<user>[A-Za-z0-9_]{2,40})", re.IGNORECASE)
_CODECHEF_LABEL = re.compile(
    rf"\bcode\s?chef\s*(?:id|username|handle|profile)?\s*[:|\-–—]\s*@?"
    rf"(?P<user>[A-Za-z0-9_]{{2,40}}){_NOT_A_DOMAIN}",
    re.IGNORECASE,
)

GITHUB_RESERVED = frozenset({
    "about", "account", "apps", "blog", "collections", "contact", "customer-stories",
    "dashboard", "discussions", "enterprise", "events", "explore", "features", "git-guides",
    "github", "home", "issues", "join", "login", "logout", "marketplace", "new",
    "notifications", "orgs", "organizations", "packages", "pricing", "pulls", "readme",
    "search", "security", "settings", "signup", "site", "solutions", "sponsors", "stars",
    "team", "topics", "trending", "users",
})  # fmt: skip
LEETCODE_RESERVED = frozenset({
    "accounts", "assessment", "circle", "company", "contest", "discuss", "explore",
    "interview", "jobs", "list", "playground", "points", "privacy", "problems", "problemset",
    "profile", "progress", "store", "studyplan", "submissions", "subscribe", "support", "tag",
    "tags", "terms", "u",
})  # fmt: skip


def _github(text: str) -> str | None:
    profiles: list[str] = []
    repo_owners: list[str] = []
    for match in _GITHUB_URL.finditer(text):
        owner = match.group("owner")
        if owner.lower() in GITHUB_RESERVED:
            continue
        (repo_owners if match.group("repo") else profiles).append(owner)
    profiles.extend(m.group("owner") for m in _GITHUB_PAGES.finditer(text))
    if profiles:
        return profiles[0]
    if repo_owners:
        # Repo links may point at other people's projects; the most frequent owner wins.
        return Counter(repo_owners).most_common(1)[0][0]
    return _first(_GITHUB_LABEL, text, GITHUB_RESERVED)


def _first(
    pattern: re.Pattern[str], text: str, reserved: frozenset[str] = frozenset()
) -> str | None:
    for match in pattern.finditer(text):
        user = match.group("user")
        if user.lower() not in reserved:
            return user
    return None


def extract_profile_handles(text: str, links: Iterable[str] = ()) -> ProfileHandles:
    """Extract usernames, preferring embedded hyperlinks (most reliable) over visible text."""
    haystack = "\n".join([*links, text])
    return ProfileHandles(
        github=_github(haystack),
        leetcode=_first(_LEETCODE_URL, haystack, LEETCODE_RESERVED)
        or _first(_LEETCODE_LABEL, haystack, LEETCODE_RESERVED),
        codechef=_first(_CODECHEF_URL, haystack) or _first(_CODECHEF_LABEL, haystack),
    )
