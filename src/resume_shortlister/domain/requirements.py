"""Turn a free-text job description into individually matchable requirements.

Cross-encoders are trained on short query -> passage pairs. Feeding them a whole job
description produces saturated, meaningless scores, so we split the description into
bullets/sentences and match each one separately. Sections such as "Nice to have" are
weighted lower, and boilerplate sections ("Benefits", "About us") are ignored.
"""

from __future__ import annotations

import re
from enum import Enum

from resume_shortlister.domain.models import Requirement

MAX_REQUIREMENT_CHARS = 400

_BULLET = re.compile(r"^\s*(?:[-*•●▪◦‣·–—>+]+|\(?\d{1,2}[.)]|\(?[a-zA-Z][.)])\s+")
_MARKDOWN_DECORATION = re.compile(r"^[#*_\s]+|[#*_\s]+$")
_SENTENCE_END = re.compile(r"(?<=[.!?।;])\s+")
_BOILERPLATE = re.compile(
    r"\b(salary|compensation|perks|equal[- ]opportunity|we offer|apply (?:now|today|here)|"
    r"send (?:us )?(?:your )?(?:cv|resume)|health insurance|paid (?:time off|leave)|"
    r"visa sponsorship|about us)\b",
    re.IGNORECASE,
)
_ABBREVIATIONS = (
    "e.g.", "i.e.", "etc.", "vs.", "approx.", "incl.", "min.", "max.", "yrs.", "exp.",
    "no.", "sr.", "jr.", "b.tech.", "m.tech.", "b.e.", "m.e.", "b.sc.", "m.sc.", "ph.d.",
)  # fmt: skip
_ABBREVIATION_RE = re.compile(
    "|".join(re.escape(a) for a in sorted(_ABBREVIATIONS, key=len, reverse=True)), re.IGNORECASE
)
_DOT_PLACEHOLDER = "․"  # one-dot leader: protects abbreviation periods while splitting


class _Section(Enum):
    REQUIRED = "required"
    OPTIONAL = "optional"
    IGNORED = "ignored"


_HEADINGS: dict[_Section, tuple[str, ...]] = {
    _Section.OPTIONAL: (
        "nice to have", "nice-to-have", "good to have", "good-to-have", "preferred",
        "preferred qualifications", "preferred skills", "bonus", "bonus points", "plus",
        "desirable", "optional",
    ),
    _Section.IGNORED: (
        "about us", "about the company", "who we are", "company overview", "benefits",
        "perks", "what we offer", "why join us", "why join", "compensation", "salary",
        "how to apply", "equal opportunity", "our culture", "location", "work location",
    ),
    _Section.REQUIRED: (
        "requirements", "required", "required skills", "must have", "must-have",
        "qualifications", "minimum qualifications", "responsibilities", "key responsibilities",
        "what you'll do", "what you will do", "skills", "key skills", "technical skills",
        "what we're looking for", "what we are looking for", "who you are", "experience",
        "role", "the role", "about the role", "job description", "duties",
    ),
}  # fmt: skip


def _heading_section(text: str) -> _Section | None:
    """Return the section a heading introduces, or None if ``text`` is not a known heading."""
    normalized = _MARKDOWN_DECORATION.sub("", text).rstrip(":").strip().lower()
    for section, phrases in _HEADINGS.items():
        if normalized in phrases:
            return section
    return None


def _split_sentences(line: str) -> list[str]:
    masked = _ABBREVIATION_RE.sub(lambda m: m.group(0).replace(".", _DOT_PLACEHOLDER), line)
    return [s.replace(_DOT_PLACEHOLDER, ".") for s in _SENTENCE_END.split(masked)]


def _clean(sentence: str) -> str:
    sentence = _MARKDOWN_DECORATION.sub("", sentence)
    sentence = sentence.strip(" \t-–—•;,").strip()
    if len(sentence) > MAX_REQUIREMENT_CHARS:
        sentence = sentence[:MAX_REQUIREMENT_CHARS].rsplit(" ", 1)[0]
    return sentence


def _is_meaningful(sentence: str) -> bool:
    alnum = sum(ch.isalnum() for ch in sentence)
    return alnum >= 2 and not sentence.replace(" ", "").isdigit()


def extract_requirements(job_description: str, max_requirements: int = 15) -> list[Requirement]:
    """Split a job description into at most ``max_requirements`` requirements.

    Required items are kept in preference to optional ones when the cap is hit. If nothing
    usable is found (e.g. a one-line description), the whole text becomes one requirement.
    """
    if max_requirements < 1:
        raise ValueError("max_requirements must be >= 1")

    section = _Section.REQUIRED
    candidates: list[tuple[str, bool]] = []
    seen: set[str] = set()

    for raw_line in job_description.splitlines():
        line = _BULLET.sub("", raw_line.strip()).strip()
        if not line:
            continue

        heading = _heading_section(line)
        if heading is not None:
            section = heading
            continue

        # Inline heading, e.g. "Nice to have: CI/CD with GitHub Actions".
        if ":" in line:
            prefix, rest = line.split(":", 1)
            inline = _heading_section(prefix)
            if inline is not None:
                section = inline
                line = rest.strip()
            elif not rest.strip() and len(prefix.split()) <= 8:
                continue  # an unknown short heading such as "Tech stack:"

        if section is _Section.IGNORED or not line:
            continue

        for sentence in _split_sentences(line):
            cleaned = _clean(sentence)
            key = cleaned.lower()
            if not _is_meaningful(cleaned) or _BOILERPLATE.search(cleaned) or key in seen:
                continue
            seen.add(key)
            candidates.append((cleaned, section is _Section.OPTIONAL))

    if not candidates:
        fallback = _clean(" ".join(job_description.split()))
        return [Requirement(id=1, text=fallback)] if fallback else []

    # Keep required items first when capping, then restore the original order.
    ranked = sorted(range(len(candidates)), key=lambda i: candidates[i][1])[:max_requirements]
    kept = sorted(ranked)
    return [
        Requirement(id=n, text=candidates[i][0], optional=candidates[i][1])
        for n, i in enumerate(kept, start=1)
    ]
