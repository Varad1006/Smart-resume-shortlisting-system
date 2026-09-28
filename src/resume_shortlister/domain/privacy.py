"""Minimise personal data before resume text is sent to a third-party service."""

from __future__ import annotations

import re

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PHONE_CANDIDATE = re.compile(r"\+?\(?\d[\d\s().-]{7,}\d")
_YEAR_RANGE = re.compile(r"(?:19|20)\d{2}\s*[-–]\s*(?:19|20)\d{2}")


def _redact_phone(match: re.Match[str]) -> str:
    text = match.group(0)
    digits = sum(ch.isdigit() for ch in text)
    if not 9 <= digits <= 15 or _YEAR_RANGE.fullmatch(text.strip()):
        return text  # dates, year ranges, short numbers stay
    return "[phone]"


def redact_contact_details(text: str) -> str:
    """Replace email addresses and phone numbers; an assessment never needs them."""
    return _PHONE_CANDIDATE.sub(_redact_phone, _EMAIL.sub("[email]", text))
