import pytest

from resume_shortlister.domain.privacy import redact_contact_details


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Mail priya.sharma+jobs@example.co.in now", "Mail [email] now"),
        ("Call +91-7745871792 today", "Call [phone] today"),
        ("Phone: 98765 43210", "Phone: [phone]"),
        ("Office (020) 2555-1234", "Office [phone]"),
    ],
)
def test_contact_details_are_redacted(text, expected):
    assert redact_contact_details(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "ShopKart (2022 - 2025)",
        "B.Tech 2019-2023",
        "80% unit test coverage, 1M+ downloads",
        "Joined 01/2020, CGPA 8.62",
        "github.com/octocat",
    ],
)
def test_ordinary_resume_content_is_kept(text):
    assert redact_contact_details(text) == text
