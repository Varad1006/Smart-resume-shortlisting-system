import pytest

from resume_shortlister.application.dto import (
    RunStatus,
    ShortlistOptions,
    Stage,
    UploadedDocument,
    UploadLimits,
)
from resume_shortlister.domain.errors import InvalidInputError, NotFoundError, PayloadTooLargeError
from resume_shortlister.domain.scoring import ScoringPolicy
from tests.conftest import ACCOUNTANT_RESUME, ANDROID_JD, ANDROID_RESUME, WEB_RESUME
from tests.fakes import FakeSummarizer

pytestmark = pytest.mark.anyio


def doc(name: str, text: str | bytes) -> UploadedDocument:
    return UploadedDocument(name, text if isinstance(text, bytes) else text.encode())


RESUMES = [
    doc("accountant.txt", ACCOUNTANT_RESUME),
    doc("android.txt", ANDROID_RESUME),
    doc("web.txt", WEB_RESUME),
]


async def test_relevant_resume_ranks_first_with_evidence(make_service):
    result = await make_service().rank(ANDROID_JD, RESUMES)

    ranked = [c.filename for c in result.shortlisted]
    assert ranked[0] == "android.txt"
    assert ranked[-1] == "accountant.txt"
    best = result.shortlisted[0]
    assert best.final_rank == 1 and best.reranked and best.shortlisted
    assert best.coverage_score > 80
    assert len(best.requirement_matches) == len(result.requirements) == 6
    kotlin = best.requirement_matches[1]
    assert kotlin.score > 0.5 and "Kotlin" in kotlin.evidence
    assert best.profiles.github == "priya-dev"
    assert best.language == "en"


async def test_scores_are_consistent(make_service):
    result = await make_service().rank(ANDROID_JD, RESUMES)
    for c in result.candidates:
        assert 0 <= c.semantic_score <= 100
        expected = min(100.0, c.relevance_score + c.social_bonus)
        assert c.final_score == pytest.approx(expected, abs=0.11)
        assert c.final_score <= 100
    finals = [c.final_score for c in result.shortlisted]
    assert finals == sorted(finals, reverse=True)


async def test_only_top_k_resumes_are_reranked(make_service, reranker):
    service = make_service(policy=ScoringPolicy(retrieval_top_k=2, semantic_ceiling=0.8))
    result = await service.rank(ANDROID_JD, RESUMES)

    reranked = [c for c in result.candidates if c.reranked]
    assert len(reranked) == 2
    left_out = [c for c in result.candidates if not c.reranked]
    assert [c.filename for c in left_out] == ["accountant.txt"]
    assert left_out[0].final_score is None and not left_out[0].shortlisted
    assert result.by_retrieval[-1].filename == "accountant.txt"
    assert reranker.pairs_scored > 0


async def test_shortlist_size_limits_the_shortlist(make_service):
    result = await make_service().rank(ANDROID_JD, RESUMES, ShortlistOptions(shortlist_size=1))
    assert [c.filename for c in result.shortlisted] == ["android.txt"]
    assert result.shortlist_size == 1
    assert sum(c.reranked for c in result.candidates) == 3


async def test_unreadable_duplicate_and_empty_files_are_skipped_with_reasons(make_service):
    documents = [
        doc("android.txt", ANDROID_RESUME),
        doc("copy.txt", ANDROID_RESUME),
        doc("broken.pdf", b"FAIL: corrupt"),
        doc("empty.pdf", b""),
        doc("tiny.txt", "hi"),
    ]
    result = await make_service().rank(ANDROID_JD, documents)

    assert [c.filename for c in result.candidates] == ["android.txt"]
    assert {s.filename: s.reason for s in result.skipped} == {
        "copy.txt": "Duplicate of android.txt.",
        "broken.pdf": "Could not open this PDF (corrupt file).",
        "empty.pdf": "File is empty.",
        "tiny.txt": "No readable text found in this file.",
    }
    assert result.total_documents == 5


async def test_nothing_readable_returns_an_empty_result(make_service):
    result = await make_service().rank(ANDROID_JD, [doc("broken.pdf", b"FAIL")])
    assert result.candidates == ()
    assert len(result.skipped) == 1


async def test_profile_bonus_never_penalises_candidates_without_profiles(make_service, social):
    result = await make_service().rank(ANDROID_JD, RESUMES, ShortlistOptions(include_social=True))
    by_name = {c.filename: c for c in result.candidates}

    android = by_name["android.txt"]
    assert android.social is not None and android.social.github is not None
    assert android.social_bonus > 0
    assert android.final_score == pytest.approx(android.relevance_score + android.social_bonus)

    web = by_name["web.txt"]  # no profiles in this resume
    assert web.social_bonus == 0.0
    assert web.final_score == web.relevance_score
    assert [h.github for h in social.queried] == ["priya-dev"]  # only candidates with handles


async def test_profiles_are_not_fetched_unless_requested(make_service, social):
    result = await make_service().rank(ANDROID_JD, RESUMES)
    assert social.queried == []
    assert all(c.social is None and c.social_bonus == 0 for c in result.candidates)
    assert result.candidates[0].profiles.github == "priya-dev"  # handles are still shown


async def test_progress_is_reported_for_every_stage(make_service, repository):
    stages = []

    async def progress(stage, done, total):
        stages.append((stage, done, total))

    await make_service().rank(ANDROID_JD, RESUMES, ShortlistOptions(include_social=True), progress)
    seen = [s for s, _, _ in stages]
    for stage in (Stage.EXTRACTING, Stage.RETRIEVING, Stage.MATCHING, Stage.PROFILES):
        assert stage in seen
    assert (Stage.EXTRACTING, 3, 3) in stages


async def test_submit_processes_in_background_and_persists(make_service, repository):
    service = make_service()
    run = await service.submit(ANDROID_JD, RESUMES)
    assert run.status is RunStatus.QUEUED

    await service.wait_for(run.id)
    stored = await service.get_run(run.id)
    assert stored.status is RunStatus.COMPLETED
    assert stored.result.shortlisted[0].filename == "android.txt"
    assert stored.title == "Android Developer (Kotlin)"
    items, total = await service.list_runs()
    assert total == 1 and items[0].id == run.id


async def test_pipeline_errors_mark_the_run_failed(make_service):
    class BrokenEmbedder:
        def embed_queries(self, texts):
            raise RuntimeError("model crashed")

        embed_documents = embed_queries

    service = make_service(embedder=BrokenEmbedder())
    run = await service.submit(ANDROID_JD, RESUMES)
    await service.wait_for(run.id)
    stored = await service.get_run(run.id)
    assert stored.status is RunStatus.FAILED
    assert stored.error == "RuntimeError: model crashed"


async def test_delete_run(make_service):
    service = make_service()
    run = await service.submit(ANDROID_JD, RESUMES)
    await service.wait_for(run.id)
    await service.delete_run(run.id)
    with pytest.raises(NotFoundError):
        await service.get_run(run.id)
    with pytest.raises(NotFoundError):
        await service.delete_run(run.id)


@pytest.mark.parametrize(
    ("job", "documents", "options", "error"),
    [
        ("   ", RESUMES, ShortlistOptions(), InvalidInputError),
        (ANDROID_JD, [], ShortlistOptions(), InvalidInputError),
        (ANDROID_JD, RESUMES * 2, ShortlistOptions(), InvalidInputError),
        (ANDROID_JD, [doc("big.pdf", b"x" * 101)], ShortlistOptions(), PayloadTooLargeError),
        (ANDROID_JD, RESUMES, ShortlistOptions(shortlist_size=0), InvalidInputError),
    ],
)
async def test_submit_validates_input(make_service, job, documents, options, error):
    limits = UploadLimits(max_files=5, max_file_bytes=100, max_total_bytes=10_000)
    with pytest.raises(error):
        await make_service(limits=limits).submit(job, documents, options)


async def test_extract_single_document(make_service):
    extracted, language = await make_service().extract(doc("cv.txt", ANDROID_RESUME))
    assert "Kotlin" in extracted.text
    assert language == "en"


# ----------------------------------------------------------------------- AI insights
async def test_insights_are_written_for_the_top_shortlisted_candidates_only(make_service):
    summarizer = FakeSummarizer()
    service = make_service(summarizer=summarizer, max_insights=2)
    result = await service.rank(ANDROID_JD, RESUMES, ShortlistOptions(include_insights=True))

    with_insight = [c.filename for c in result.shortlisted if c.insight]
    assert with_insight == ["android.txt", "web.txt"]
    assert result.shortlisted[2].insight is None
    assert result.insights_enabled
    best = result.shortlisted[0]
    assert (
        best.insight.summary.startswith("Scored ") and "Android Developer" in best.insight.summary
    )
    assert "insights_s" in result.timings
    assert [r.final_score for r in summarizer.requests] == [
        c.final_score for c in result.shortlisted[:2]
    ]


async def test_insight_requests_never_contain_contact_details(make_service):
    summarizer = FakeSummarizer()
    resume = ANDROID_RESUME + "\nContact: priya@example.com, +91 98765 43210"
    await make_service(summarizer=summarizer).rank(
        ANDROID_JD, [doc("cv.txt", resume)], ShortlistOptions(include_insights=True)
    )
    (request,) = summarizer.requests
    assert "priya@example.com" not in request.resume_text
    assert "98765" not in request.resume_text
    assert "[email]" in request.resume_text and "[phone]" in request.resume_text
    assert len(request.matches) == len(request.requirements) == 6


async def test_insight_failures_are_reported_without_affecting_scores(make_service):
    service = make_service(summarizer=FakeSummarizer())
    failing = doc("flaky.txt", ANDROID_RESUME + " FAIL-LLM")
    with_llm = await service.rank(ANDROID_JD, [failing], ShortlistOptions(include_insights=True))
    without_llm = await service.rank(ANDROID_JD, [failing])

    (candidate,) = with_llm.candidates
    assert candidate.insight is None
    assert candidate.insight_error == "api.example.com rate limit reached; try again in a minute."
    assert candidate.final_score == without_llm.candidates[0].final_score


async def test_insights_are_skipped_unless_requested(make_service):
    summarizer = FakeSummarizer()
    result = await make_service(summarizer=summarizer).rank(ANDROID_JD, RESUMES)
    assert summarizer.requests == []
    assert not result.insights_enabled


async def test_insights_require_a_configured_summarizer(make_service):
    service = make_service()
    assert service.summarizer_name is None
    with pytest.raises(InvalidInputError, match="not available"):
        await service.submit(ANDROID_JD, RESUMES, ShortlistOptions(include_insights=True))


async def test_insights_stage_reports_progress(make_service):
    stages = []

    async def progress(stage, done, total):
        stages.append((stage, done, total))

    service = make_service(summarizer=FakeSummarizer(), max_insights=5)
    await service.rank(ANDROID_JD, RESUMES, ShortlistOptions(include_insights=True), progress)
    assert (Stage.INSIGHTS, 0, 3) in stages
    assert (Stage.INSIGHTS, 3, 3) in stages
