import json

import httpx
import pytest

from resume_shortlister.application.dto import InsightRequest
from resume_shortlister.domain.errors import InsightError
from resume_shortlister.domain.models import CandidateInsight, Requirement, RequirementMatch
from resume_shortlister.infrastructure.llm.openai_compatible import (
    OpenAICompatibleSummarizer,
    build_messages,
    parse_insight,
)

pytestmark = pytest.mark.anyio

REQUEST = InsightRequest(
    job_title="Android Developer (Kotlin)",
    requirements=(Requirement(1, "Kotlin"), Requirement(2, "Hilt", optional=True)),
    matches=(RequirementMatch(1, 0.95, "Kotlin for 3 years"), RequirementMatch(2, 0.02)),
    resume_text="Three years of Kotlin. Contact: [email]",
    language="en",
    final_score=81.25,
)
GOOD_JSON = (
    '{"summary": "Strong fit.", "strengths": ["Kotlin"], "gaps": ["No Hilt"],'
    ' "interview_questions": ["Why Compose?"]}'
)


def completion(content: str | None, finish_reason: str = "stop") -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "openai/gpt-oss-120b",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def summarizer_for(handler, **kwargs) -> OpenAICompatibleSummarizer:
    return OpenAICompatibleSummarizer(
        base_url="https://api.groq.com/openai/v1",
        api_key="test-key",
        model="openai/gpt-oss-120b",
        display_name="GPT-OSS 120B [Groq]",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        **kwargs,
    )


async def test_summarize_sends_a_compact_json_mode_request():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers["authorization"]
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=completion(GOOD_JSON))

    summarizer = summarizer_for(handler)
    insight = await summarizer.summarize(REQUEST)
    await summarizer.aclose()

    assert insight == CandidateInsight(
        summary="Strong fit.",
        strengths=("Kotlin",),
        gaps=("No Hilt",),
        interview_questions=("Why Compose?",),
        model="openai/gpt-oss-120b",
    )
    assert captured["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert captured["auth"] == "Bearer test-key"
    body = captured["body"]
    assert body["model"] == "openai/gpt-oss-120b"
    assert body["reasoning_effort"] == "low"
    assert body["response_format"] == {"type": "json_object"}
    assert body["max_completion_tokens"] == 1200
    user = body["messages"][1]["content"]
    assert "R1 (95): Kotlin" in user and "R2 (2, nice to have): Hilt" in user
    assert "81.2/100" in user
    assert summarizer.name == "GPT-OSS 120B [Groq]" and summarizer.host == "api.groq.com"


async def test_reasoning_effort_is_optional():
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=completion(GOOD_JSON))

    await summarizer_for(handler, reasoning_effort=None).summarize(REQUEST)
    assert "reasoning_effort" not in bodies[0]


@pytest.mark.parametrize(
    ("status", "message"),
    [
        (401, "api.groq.com rejected the API key (HTTP 401)."),
        (429, "api.groq.com rate limit reached; try again in a minute."),
        (500, "api.groq.com returned HTTP 500: boom"),
    ],
)
async def test_http_errors_become_readable_insight_errors(status, message):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": {"message": "boom"}})

    with pytest.raises(InsightError) as caught:
        await summarizer_for(handler).summarize(REQUEST)
    assert str(caught.value) == message


async def test_unreachable_host():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route", request=request)

    with pytest.raises(InsightError, match=r"Could not reach api\.groq\.com"):
        await summarizer_for(handler).summarize(REQUEST)


async def test_truncated_response_asks_for_more_tokens():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=completion("", finish_reason="length"))

    with pytest.raises(InsightError, match="LLM_MAX_TOKENS"):
        await summarizer_for(handler).summarize(REQUEST)


def test_messages_carry_rules_and_the_redacted_resume():
    system, user = build_messages(REQUEST)
    assert system["role"] == "system" and "Never invent" in system["content"]
    assert '"""\nThree years of Kotlin. Contact: [email]\n"""' in user["content"]


@pytest.mark.parametrize(
    "content",
    [
        GOOD_JSON,
        f"```json\n{GOOD_JSON}\n```",
        f"Here is the assessment:\n{GOOD_JSON}\nHope this helps.",
    ],
)
def test_parse_insight_tolerates_fences_and_prose(content):
    assert parse_insight(content, "m").summary == "Strong fit."


def test_parse_insight_normalises_lists():
    data = {
        "summary": "  Good\n fit ",
        "strengths": ["a", "", "b", "c", "d", "e", 7, {"x": 1}],
        "gaps": "not a list",
        "interview_questions": ["q" * 400],
    }
    insight = parse_insight(json.dumps(data), "m")
    assert insight.summary == "Good fit"
    assert insight.strengths == ("a", "b", "c", "d")
    assert insight.gaps == ()
    assert len(insight.interview_questions[0]) == 300


@pytest.mark.parametrize("content", ["not json at all", "[1, 2]", '{"summary": "  "}'])
def test_parse_insight_rejects_unusable_output(content):
    with pytest.raises(InsightError):
        parse_insight(content, "m")
