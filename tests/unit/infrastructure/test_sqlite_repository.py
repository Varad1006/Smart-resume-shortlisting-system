import pytest

from resume_shortlister.application.dto import RunStatus, ShortlistOptions, ShortlistRun, Stage
from resume_shortlister.domain.models import (
    CandidateResult,
    ExtractionMethod,
    ProfileHandles,
    Requirement,
    RequirementMatch,
    ShortlistResult,
    SkippedDocument,
)
from resume_shortlister.infrastructure.persistence.sqlite_repository import (
    SqliteShortlistRepository,
)
from tests.fakes import utc

pytestmark = pytest.mark.anyio


def make_run(run_id: str, day: int = 1) -> ShortlistRun:
    return ShortlistRun(
        id=run_id,
        status=RunStatus.QUEUED,
        stage=Stage.QUEUED,
        created_at=utc(day=day),
        updated_at=utc(day=day),
        job_description="Android Developer\n- Kotlin",
        options=ShortlistOptions(shortlist_size=5, include_social=True),
        document_count=2,
        progress_total=2,
    )


def make_result() -> ShortlistResult:
    candidate = CandidateResult(
        candidate_id="abc",
        filename="jane.pdf",
        language="hi",
        extraction_methods=(ExtractionMethod.TEXT_LAYER, ExtractionMethod.CHANDRA),
        pages=2,
        semantic_similarity=0.9,
        semantic_score=80.0,
        retrieval_rank=1,
        reranked=True,
        coverage_score=70.0,
        requirement_matches=(RequirementMatch(1, 0.93, "Kotlin for 3 years"),),
        relevance_score=73.0,
        profiles=ProfileHandles(github="jane"),
        final_score=73.0,
        final_rank=1,
        shortlisted=True,
    )
    return ShortlistResult(
        requirements=(Requirement(1, "Kotlin"),),
        candidates=(candidate,),
        skipped=(SkippedDocument("bad.pdf", "corrupt"),),
        shortlist_size=5,
        social_enabled=True,
        total_documents=2,
        timings={"total_s": 1.5},
    )


@pytest.fixture
async def repo(tmp_path):
    repository = SqliteShortlistRepository(tmp_path / "nested" / "runs.db")
    await repository.initialize()
    return repository


async def test_full_lifecycle_round_trip(repo):
    await repo.add(make_run("r1"))
    await repo.update_progress("r1", Stage.MATCHING, 1, 2)
    running = await repo.get("r1")
    assert (running.status, running.stage, running.progress_done) == (
        RunStatus.RUNNING,
        Stage.MATCHING,
        1,
    )

    result = make_result()
    await repo.complete("r1", result)
    stored = await repo.get("r1")
    assert stored.status is RunStatus.COMPLETED and stored.stage is Stage.DONE
    assert stored.result == result
    assert stored.options == ShortlistOptions(shortlist_size=5, include_social=True)
    assert stored.progress_done == 2

    await repo.fail("r1", "too late")  # finished runs are never overwritten
    assert (await repo.get("r1")).status is RunStatus.COMPLETED


async def test_list_orders_newest_first_with_summary(repo):
    await repo.add(make_run("old", day=1))
    await repo.add(make_run("new", day=2))
    await repo.complete("old", make_result())
    items, total = await repo.list(limit=10, offset=0)
    assert total == 2
    assert [i.id for i in items] == ["new", "old"]
    assert (items[1].top_candidate, items[1].top_score, items[1].shortlisted_count) == (
        "jane.pdf",
        73.0,
        1,
    )
    assert items[0].title == "Android Developer"
    page, _ = await repo.list(limit=1, offset=1)
    assert [i.id for i in page] == ["old"]


async def test_fail_unfinished_and_delete(repo):
    await repo.add(make_run("a"))
    await repo.add(make_run("b"))
    await repo.complete("b", make_result())
    assert await repo.fail_unfinished("restart") == 1
    failed = await repo.get("a")
    assert (failed.status, failed.error) == (RunStatus.FAILED, "restart")
    assert await repo.delete("a")
    assert not await repo.delete("a")
    assert await repo.get("a") is None
