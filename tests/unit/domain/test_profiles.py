import pytest

from resume_shortlister.domain.models import ProfileHandles
from resume_shortlister.domain.profiles import extract_profile_handles


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Code: https://github.com/jane-doe", "jane-doe"),
        ("GITHUB.COM/JaneDoe/", "JaneDoe"),
        ("GitHub: janedoe", "janedoe"),
        ("GitHub | jane_x", None),  # underscores are not valid in GitHub usernames
        ("/gtbgithub.com/AnimeshN", "AnimeshN"),  # icon-font residue glued to the URL
        ("Portfolio: janedoe.github.io", "janedoe"),
        ("github.com/orgs/acme and github.com/features", None),
        ("GitHub: github.com/realname", "realname"),  # label must not yield "github"/"githu"
    ],
)
def test_github_handles(text, expected):
    assert extract_profile_handles(text).github == expected


def test_github_prefers_profile_url_over_repository_links():
    text = "Contributed to github.com/apache/kafka. Profile: github.com/janedoe"
    assert extract_profile_handles(text).github == "janedoe"


def test_github_repo_links_fall_back_to_most_frequent_owner():
    text = "github.com/apache/kafka github.com/jane/app github.com/jane/api"
    assert extract_profile_handles(text).github == "jane"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("leetcode.com/u/coder_42/", "coder_42"),  # regression: new URL format parsed as "u"
        ("https://leetcode.com/coder-42", "coder-42"),
        ("leetcode.com/problems/two-sum", None),
        ("LeetCode ID: coder42", "coder42"),
    ],
)
def test_leetcode_handles(text, expected):
    assert extract_profile_handles(text).leetcode == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("codechef.com/users/chef_99", "chef_99"),
        ("CodeChef: chef99", "chef99"),
        ("codechef.com/practice", None),
    ],
)
def test_codechef_handles(text, expected):
    assert extract_profile_handles(text).codechef == expected


def test_embedded_hyperlinks_take_precedence_over_visible_text():
    handles = extract_profile_handles(
        "GitHub  LeetCode", links=["https://github.com/linked", "https://leetcode.com/u/lc/"]
    )
    assert handles == ProfileHandles(github="linked", leetcode="lc", codechef=None)
    assert handles.any()
    assert not ProfileHandles().any()
