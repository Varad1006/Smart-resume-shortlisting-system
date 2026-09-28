"""View helpers (Jinja2 filters/globals) for the server-rendered UI."""

from __future__ import annotations

from datetime import UTC, datetime

from jinja2 import Environment

from resume_shortlister import __version__
from resume_shortlister.application.dto import Stage
from resume_shortlister.domain.models import ExtractionMethod
from resume_shortlister.domain.scoring import match_band

METHOD_LABELS = {
    ExtractionMethod.TEXT_LAYER: "PDF text",
    ExtractionMethod.DOCX: "DOCX",
    ExtractionMethod.PLAIN_TEXT: "Plain text",
    ExtractionMethod.CHANDRA: "Chandra OCR",
    ExtractionMethod.TESSERACT: "Tesseract OCR",
}

LANGUAGE_NAMES = {
    "ar": "Arabic", "bn": "Bengali", "cs": "Czech", "da": "Danish", "de": "German",
    "el": "Greek", "en": "English", "es": "Spanish", "fa": "Persian", "fi": "Finnish",
    "fr": "French", "gu": "Gujarati", "he": "Hebrew", "hi": "Hindi", "hu": "Hungarian",
    "id": "Indonesian", "it": "Italian", "ja": "Japanese", "kn": "Kannada", "ko": "Korean",
    "ml": "Malayalam", "mr": "Marathi", "ne": "Nepali", "nl": "Dutch", "no": "Norwegian",
    "pa": "Punjabi", "pl": "Polish", "pt": "Portuguese", "ro": "Romanian", "ru": "Russian",
    "sv": "Swedish", "ta": "Tamil", "te": "Telugu", "th": "Thai", "tr": "Turkish",
    "uk": "Ukrainian", "ur": "Urdu", "vi": "Vietnamese", "zh-cn": "Chinese",
    "zh-tw": "Chinese (Traditional)",
}  # fmt: skip

STAGES: list[tuple[str, str]] = [
    (Stage.QUEUED.value, "Waiting in queue"),
    (Stage.EXTRACTING.value, "Reading resumes"),
    (Stage.RETRIEVING.value, "Semantic retrieval"),
    (Stage.MATCHING.value, "Matching requirements"),
    (Stage.PROFILES.value, "Checking public profiles"),
    (Stage.INSIGHTS.value, "Writing AI summaries"),
    (Stage.DONE.value, "Done"),
]

# Share of the overall progress bar covered by each stage: (start %, end %).
_STAGE_SPAN = {
    Stage.QUEUED: (0, 0),
    Stage.EXTRACTING: (0, 50),
    Stage.RETRIEVING: (50, 58),
    Stage.MATCHING: (58, 80),
    Stage.PROFILES: (80, 88),
    Stage.INSIGHTS: (88, 99),
    Stage.DONE: (100, 100),
}


def overall_progress(stage: Stage, done: int, total: int) -> int:
    start, end = _STAGE_SPAN[stage]
    fraction = done / total if total else 0.0
    return round(start + (end - start) * min(1.0, max(0.0, fraction)))


def fmt_score(value: float | None) -> str:
    return "—" if value is None else f"{value:.1f}"


def band(value: float | None) -> str:
    result = match_band(value)
    return result.value if result else "none"


def requirement_level(probability: float) -> str:
    if probability >= 0.5:
        return "met"
    if probability >= 0.15:
        return "partial"
    return "missing"


def pct(probability: float) -> int:
    return round(probability * 100)


def when(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%d %b %Y, %H:%M UTC")


def language_name(code: str | None) -> str:
    if not code:
        return "Unknown"
    return LANGUAGE_NAMES.get(code, code.upper())


def method_label(method: ExtractionMethod | str) -> str:
    return METHOD_LABELS.get(ExtractionMethod(method), str(method))


def register(env: Environment) -> None:
    env.filters.update(
        score=fmt_score,
        band=band,
        level=requirement_level,
        pct=pct,
        when=when,
        language_name=language_name,
        method_label=method_label,
    )
    env.globals.update(STAGES=STAGES, overall_progress=overall_progress, version=__version__)
