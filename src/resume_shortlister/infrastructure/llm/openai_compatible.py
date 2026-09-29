"""CandidateSummarizer adapter for any OpenAI-compatible chat API (Groq, OpenAI, vLLM, Ollama).

Default configuration: GPT-OSS 120B on Groq. The model only *explains* a ranking that was
computed locally; it never changes scores. Requests are small (redacted resume excerpt +
requirement evidence) so they fit free-tier token-per-minute limits.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any
from urllib.parse import urlparse

import httpx
import openai
from openai import AsyncOpenAI

from resume_shortlister.application.dto import InsightRequest
from resume_shortlister.domain.errors import InsightError
from resume_shortlister.domain.models import CandidateInsight

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an experienced technical recruiter. Assess ONE candidate against a \
job using only the resume text and the automated requirement-match evidence you are given.
Rules:
- Be specific, factual and concise. Never invent skills, employers, dates or numbers.
- The match scores come from an automated matcher and can be wrong: trust the resume text.
- The resume may be OCR output or in another language; answer in the language of the job \
requirements.
- Reply with a single JSON object and nothing else."""

RESPONSE_SHAPE = """Return JSON with exactly these keys:
{"summary": "2-3 sentences on overall fit for this job",
 "strengths": ["up to 4 concrete strengths backed by the resume"],
 "gaps": ["up to 4 job requirements the resume does NOT demonstrate"],
 "interview_questions": ["up to 3 questions to verify fit or probe the gaps"]}
List only real gaps: never list a requirement the resume shows, and use [] when there
are none. Keep every list item under 25 words."""

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
_LIMITS = {"strengths": 4, "gaps": 4, "interview_questions": 3}


def _error_detail(exc: openai.APIStatusError) -> str:
    """The provider's own error message (the SDK's default text embeds the raw body)."""
    body = exc.body
    if isinstance(body, dict):
        error = body.get("error", body)
        if isinstance(error, dict) and error.get("message"):
            return " ".join(str(error["message"]).split())[:200]
    return " ".join(str(exc.message).split())[:200]


def build_messages(request: InsightRequest) -> list[dict[str, str]]:
    scores = {m.requirement_id: m.score for m in request.matches}
    lines = [
        f"R{r.id} ({round(scores.get(r.id, 0.0) * 100)}"
        f"{', nice to have' if r.optional else ''}): {r.text}"
        for r in request.requirements
    ]
    user = (
        f"Job: {request.job_title}\n\n"
        "Requirements with automated match scores (0-100 = confidence that the resume "
        "demonstrates it):\n" + "\n".join(lines) + "\n\n"
        f"Overall ranking score: {request.final_score:.1f}/100\n\n"
        f"Resume (detected language: {request.language or 'unknown'}; contact details "
        f'removed):\n"""\n{request.resume_text}\n"""\n\n{RESPONSE_SHAPE}'
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def _clean_text(value: Any, max_chars: int) -> str:
    text = " ".join(str(value).split()) if isinstance(value, str | int | float) else ""
    return text if len(text) <= max_chars else text[: max_chars - 1].rstrip() + "…"


def _clean_list(value: Any, max_items: int) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    items = (_clean_text(item, 300) for item in value)
    return tuple(item for item in items if item)[:max_items]


def parse_insight(content: str) -> CandidateInsight:
    """Parse the model's JSON (tolerating code fences or stray prose around it)."""
    text = _FENCE.sub("", content.strip())
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = _OBJECT.search(text)
        try:
            data = json.loads(match.group(0)) if match else None
        except json.JSONDecodeError:
            data = None
    if not isinstance(data, dict):
        raise InsightError("The AI service returned an unreadable answer.")
    summary = _clean_text(data.get("summary"), 800)
    if not summary:
        raise InsightError("The AI service returned an empty summary.")
    return CandidateInsight(
        summary=summary,
        strengths=_clean_list(data.get("strengths"), _LIMITS["strengths"]),
        gaps=_clean_list(data.get("gaps"), _LIMITS["gaps"]),
        interview_questions=_clean_list(
            data.get("interview_questions"), _LIMITS["interview_questions"]
        ),
    )


class OpenAICompatibleSummarizer:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        display_name: str | None = None,
        max_tokens: int = 1200,
        reasoning_effort: str | None = "low",
        temperature: float = 0.3,
        timeout_seconds: float = 60.0,
        max_retries: int = 6,  # the SDK waits for `retry-after` on HTTP 429
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._model = model
        self._name = display_name or model
        self._host = urlparse(base_url).hostname or base_url
        self._max_tokens = max_tokens
        self._reasoning_effort = reasoning_effort
        self._temperature = temperature
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout_seconds,
            max_retries=max_retries,
            http_client=http_client,
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def host(self) -> str:
        return self._host

    async def aclose(self) -> None:
        await self._client.close()

    async def summarize(self, request: InsightRequest) -> CandidateInsight:
        options: dict[str, Any] = {
            "model": self._model,
            "messages": build_messages(request),
            "max_completion_tokens": self._max_tokens,
            "temperature": self._temperature,
            "response_format": {"type": "json_object"},
        }
        if self._reasoning_effort:
            options["reasoning_effort"] = self._reasoning_effort
        try:
            completion = await self._client.chat.completions.create(**options)
        except openai.AuthenticationError as exc:
            logger.warning("AI summary rejected by %s: %s", self._host, _error_detail(exc))
            raise InsightError("The AI service rejected the configured key.") from exc
        except openai.RateLimitError as exc:
            logger.warning("AI summary rate limited by %s", self._host)
            raise InsightError("The AI service is busy; try again in a minute.") from exc
        except openai.APIStatusError as exc:
            logger.warning(
                "AI summary failed at %s (HTTP %s): %s",
                self._host,
                exc.status_code,
                _error_detail(exc),
            )
            raise InsightError(
                f"The AI service returned an error (HTTP {exc.status_code})."
            ) from exc
        except (openai.APIConnectionError, openai.APITimeoutError) as exc:
            logger.warning("AI summary could not reach %s: %r", self._host, exc)
            raise InsightError("The AI service could not be reached.") from exc

        choice = completion.choices[0]
        content = choice.message.content or ""
        if not content.strip() and choice.finish_reason == "length":
            logger.warning("AI summary truncated; raise LLM_MAX_TOKENS")
            raise InsightError("The AI summary was cut off.")
        return parse_insight(content)
