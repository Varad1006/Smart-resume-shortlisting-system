"""The shortlisting use case: job description + resumes -> explainable, ranked shortlist.

Pipeline
  1. Extract   text from every file (PDF text layer, DOCX, or OCR via Chandra/Tesseract).
  2. Retrieve  split resumes into passages, embed them with a multilingual bi-encoder and
               rank every resume by its best passage (fast, cross-lingual).
  3. Match     for the top-K resumes, score each job requirement against the most
               relevant passages with a cross-encoder -> requirement coverage + evidence.
  4. Profiles  optionally look up public GitHub/LeetCode/CodeChef activity as a bonus.
  5. Rank      relevance (semantic + coverage) + optional bonus -> shortlist.
  6. Insights  optionally ask an LLM for a recruiter summary of the top shortlisted
               candidates. Runs after ranking, so it can never change a score.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np

from resume_shortlister.application.dto import (
    InsightRequest,
    RunStatus,
    RunSummary,
    ShortlistOptions,
    ShortlistRun,
    Stage,
    UploadedDocument,
    UploadLimits,
    job_title,
)
from resume_shortlister.application.ports import (
    CandidateSummarizer,
    DocumentExtractor,
    Embedder,
    LanguageDetector,
    Reranker,
    ShortlistRepository,
    SocialStatsProvider,
)
from resume_shortlister.domain import scoring
from resume_shortlister.domain.chunking import chunk_text, evidence_snippet
from resume_shortlister.domain.errors import (
    ExtractionError,
    InsightError,
    InvalidInputError,
    NotFoundError,
    PayloadTooLargeError,
    ShortlisterError,
    UnsupportedDocumentError,
)
from resume_shortlister.domain.models import (
    CandidateResult,
    ExtractedDocument,
    ProfileHandles,
    Requirement,
    RequirementMatch,
    ShortlistResult,
    SkippedDocument,
    SocialSnapshot,
)
from resume_shortlister.domain.privacy import redact_contact_details
from resume_shortlister.domain.profiles import extract_profile_handles
from resume_shortlister.domain.requirements import extract_requirements
from resume_shortlister.domain.scoring import ScoringPolicy

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[Stage, int, int], Awaitable[None]]

MIN_TEXT_CHARS = 20
EXCERPT_WORDS = 60


async def _no_progress(stage: Stage, done: int, total: int) -> None:
    return None


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _describe(exc: BaseException) -> str:
    message = str(exc) if isinstance(exc, ShortlisterError) else f"{type(exc).__name__}: {exc}"
    return message[:500]


@dataclass(slots=True)
class _Candidate:
    """Working state for one readable resume while the pipeline runs."""

    candidate_id: str
    document: ExtractedDocument
    language: str | None = None
    chunks: list[str] = field(default_factory=list)
    chunk_vectors: np.ndarray | None = None
    semantic_similarity: float = 0.0
    best_chunk: int = 0


class ShortlistService:
    def __init__(
        self,
        *,
        extractor: DocumentExtractor,
        embedder: Embedder,
        reranker: Reranker,
        language_detector: LanguageDetector,
        social: SocialStatsProvider,
        repository: ShortlistRepository,
        policy: ScoringPolicy | None = None,
        limits: UploadLimits | None = None,
        max_concurrent_runs: int = 1,
        extraction_concurrency: int = 4,
        summarizer: CandidateSummarizer | None = None,
        max_insights: int = 5,
        max_insight_chars: int = 4000,
    ) -> None:
        self._extractor = extractor
        self._embedder = embedder
        self._reranker = reranker
        self._language = language_detector
        self._social = social
        self._repository = repository
        self.policy = policy or ScoringPolicy()
        self.limits = limits or UploadLimits()
        self._run_slots = asyncio.Semaphore(max_concurrent_runs)
        self._extraction_concurrency = max(1, extraction_concurrency)
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._summarizer = summarizer
        self.max_insights = max_insights
        self._max_insight_chars = max_insight_chars

    @property
    def summarizer_name(self) -> str | None:
        """Display name of the configured LLM, or None when AI summaries are disabled."""
        return self._summarizer.name if self._summarizer else None

    # ------------------------------------------------------------------ queries
    async def get_run(self, run_id: str) -> ShortlistRun:
        run = await self._repository.get(run_id)
        if run is None:
            raise NotFoundError(f"Shortlist run {run_id!r} not found.")
        return run

    async def list_runs(self, limit: int = 20, offset: int = 0) -> tuple[list[RunSummary], int]:
        return await self._repository.list(limit=limit, offset=offset)

    # ----------------------------------------------------------------- commands
    async def submit(
        self,
        job_description: str,
        documents: Sequence[UploadedDocument],
        options: ShortlistOptions | None = None,
    ) -> ShortlistRun:
        """Validate and persist a new run, then process it in the background."""
        options = options or ShortlistOptions()
        job_description = self._validate(job_description, documents, options)
        now = _utcnow()
        run = ShortlistRun(
            id=uuid.uuid4().hex,
            status=RunStatus.QUEUED,
            stage=Stage.QUEUED,
            created_at=now,
            updated_at=now,
            job_description=job_description,
            options=options,
            document_count=len(documents),
            progress_total=len(documents),
        )
        await self._repository.add(run)
        task = asyncio.create_task(
            self._execute(run.id, job_description, list(documents), options),
            name=f"shortlist-{run.id}",
        )
        self._tasks[run.id] = task
        task.add_done_callback(lambda _task, run_id=run.id: self._tasks.pop(run_id, None))
        return run

    async def delete_run(self, run_id: str) -> None:
        task = self._tasks.get(run_id)
        if task is not None:
            task.cancel()
        if not await self._repository.delete(run_id):
            raise NotFoundError(f"Shortlist run {run_id!r} not found.")

    async def wait_for(self, run_id: str) -> None:
        """Wait until a background run finishes (used by tests and scripts)."""
        task = self._tasks.get(run_id)
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)

    async def shutdown(self) -> None:
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def extract(self, document: UploadedDocument) -> tuple[ExtractedDocument, str | None]:
        """Extract a single document (diagnostics: check OCR quality)."""
        self._validate_documents([document])
        extracted = await asyncio.to_thread(
            self._extractor.extract, document.filename, document.content
        )
        language = await asyncio.to_thread(self._language.detect, extracted.text)
        return extracted, language

    # ----------------------------------------------------------------- pipeline
    async def rank(
        self,
        job_description: str,
        documents: Sequence[UploadedDocument],
        options: ShortlistOptions | None = None,
        progress: ProgressCallback = _no_progress,
    ) -> ShortlistResult:
        """Run the full pipeline synchronously (no persistence)."""
        options = options or ShortlistOptions()
        policy = self.policy.with_shortlist_size(options.shortlist_size)
        timings: dict[str, float] = {}
        started = time.perf_counter()
        requirements = extract_requirements(job_description, policy.max_requirements)

        mark = time.perf_counter()
        candidates, skipped = await self._extract_all(documents, progress)
        timings["extraction_s"] = round(time.perf_counter() - mark, 2)

        if not candidates:
            timings["total_s"] = round(time.perf_counter() - started, 2)
            return ShortlistResult(
                requirements=tuple(requirements),
                candidates=(),
                skipped=tuple(skipped),
                shortlist_size=policy.shortlist_size,
                social_enabled=options.include_social,
                total_documents=len(documents),
                timings=timings,
                insights_enabled=options.include_insights,
            )

        await progress(Stage.RETRIEVING, 0, len(candidates))
        mark = time.perf_counter()
        requirement_vectors = await asyncio.to_thread(
            self._embed_and_retrieve, job_description, requirements, candidates, policy
        )
        ordered = sorted(candidates, key=lambda c: -c.semantic_similarity)
        top = ordered[: policy.retrieval_top_k]
        timings["retrieval_s"] = round(time.perf_counter() - mark, 2)
        await progress(Stage.RETRIEVING, len(candidates), len(candidates))

        mark = time.perf_counter()
        matches = await self._match_requirements(
            top, requirements, requirement_vectors, policy, progress
        )
        timings["matching_s"] = round(time.perf_counter() - mark, 2)

        handles = {
            c.candidate_id: extract_profile_handles(c.document.text, c.document.links)
            for c in candidates
        }
        snapshots: dict[str, SocialSnapshot] = {}
        if options.include_social:
            mark = time.perf_counter()
            snapshots = await self._fetch_social(top, handles, progress)
            timings["profiles_s"] = round(time.perf_counter() - mark, 2)

        results = self._assemble(
            ordered, top, requirements, matches, handles, snapshots, policy, options
        )
        if options.include_insights:
            mark = time.perf_counter()
            texts = {c.candidate_id: c.document.text for c in candidates}
            await self._generate_insights(job_description, requirements, results, texts, progress)
            timings["insights_s"] = round(time.perf_counter() - mark, 2)
        timings["total_s"] = round(time.perf_counter() - started, 2)
        return ShortlistResult(
            requirements=tuple(requirements),
            candidates=tuple(results),
            skipped=tuple(skipped),
            shortlist_size=policy.shortlist_size,
            social_enabled=options.include_social,
            total_documents=len(documents),
            timings=timings,
            insights_enabled=options.include_insights,
        )

    async def _execute(
        self,
        run_id: str,
        job_description: str,
        documents: list[UploadedDocument],
        options: ShortlistOptions,
    ) -> None:
        async def progress(stage: Stage, done: int, total: int) -> None:
            await self._repository.update_progress(run_id, stage, done, total)

        async with self._run_slots:
            try:
                await progress(Stage.EXTRACTING, 0, len(documents))
                result = await self.rank(job_description, documents, options, progress)
                await self._repository.complete(run_id, result)
                logger.info(
                    "Run %s completed: %d files, %d ranked, %d skipped in %.1fs",
                    run_id,
                    result.total_documents,
                    len(result.candidates),
                    len(result.skipped),
                    result.timings.get("total_s", 0.0),
                )
            except asyncio.CancelledError:
                await self._repository.fail(run_id, "Cancelled before completion.")
                raise
            except Exception as exc:
                logger.exception("Run %s failed", run_id)
                await self._repository.fail(run_id, _describe(exc))

    async def _extract_all(
        self, documents: Sequence[UploadedDocument], progress: ProgressCallback
    ) -> tuple[list[_Candidate], list[SkippedDocument]]:
        digests = [hashlib.sha256(d.content).hexdigest() for d in documents]
        first_seen: dict[str, int] = {}
        to_extract: list[int] = []
        for index, (document, digest) in enumerate(zip(documents, digests, strict=True)):
            if document.content and digest not in first_seen:
                first_seen[digest] = index
                to_extract.append(index)

        total = len(documents)
        done = total - len(to_extract)
        await progress(Stage.EXTRACTING, done, total)
        outcomes: dict[int, ExtractedDocument | str] = {}
        slots = asyncio.Semaphore(self._extraction_concurrency)

        async def extract_one(index: int) -> tuple[int, ExtractedDocument | str]:
            document = documents[index]
            async with slots:
                try:
                    extracted = await asyncio.to_thread(
                        self._extractor.extract, document.filename, document.content
                    )
                except (ExtractionError, UnsupportedDocumentError) as exc:
                    return index, str(exc)
                except Exception:
                    logger.exception("Unexpected failure while extracting document %d", index)
                    return index, "Unexpected error while reading this file."
            if len(extracted.text.strip()) < MIN_TEXT_CHARS:
                return index, "No readable text found in this file."
            return index, extracted

        for next_done in asyncio.as_completed([extract_one(i) for i in to_extract]):
            index, outcome = await next_done
            outcomes[index] = outcome
            done += 1
            await progress(Stage.EXTRACTING, done, total)

        candidates: list[_Candidate] = []
        skipped: list[SkippedDocument] = []
        for index, (document, digest) in enumerate(zip(documents, digests, strict=True)):
            if not document.content:
                skipped.append(SkippedDocument(document.filename, "File is empty."))
            elif first_seen[digest] != index:
                original = documents[first_seen[digest]].filename
                skipped.append(SkippedDocument(document.filename, f"Duplicate of {original}."))
            elif isinstance(outcome := outcomes[index], str):
                skipped.append(SkippedDocument(document.filename, outcome))
            else:
                candidates.append(_Candidate(candidate_id=digest[:12], document=outcome))
        return candidates, skipped

    def _embed_and_retrieve(
        self,
        job_description: str,
        requirements: list[Requirement],
        candidates: list[_Candidate],
        policy: ScoringPolicy,
    ) -> np.ndarray:
        """Stage 2 (blocking): language detection, chunking, embeddings, semantic scores."""
        for candidate in candidates:
            candidate.language = self._language.detect(candidate.document.text)
            candidate.chunks = chunk_text(
                candidate.document.text, policy.chunk_words, policy.chunk_overlap_words
            )

        queries = [job_description, *(r.text for r in requirements)]
        query_vectors = np.asarray(self._embedder.embed_queries(queries))
        passages = [chunk for c in candidates for chunk in c.chunks]
        passage_vectors = np.asarray(self._embedder.embed_documents(passages))

        job_vector = query_vectors[0]
        offset = 0
        for candidate in candidates:
            count = len(candidate.chunks)
            candidate.chunk_vectors = passage_vectors[offset : offset + count]
            offset += count
            similarities = candidate.chunk_vectors @ job_vector
            candidate.best_chunk = int(np.argmax(similarities))
            candidate.semantic_similarity = float(similarities[candidate.best_chunk])
        return query_vectors[1:]

    async def _match_requirements(
        self,
        top: list[_Candidate],
        requirements: list[Requirement],
        requirement_vectors: np.ndarray,
        policy: ScoringPolicy,
        progress: ProgressCallback,
    ) -> dict[str, list[RequirementMatch]]:
        """Stage 3: cross-encode each requirement against the candidate's best passages."""
        results: dict[str, list[RequirementMatch]] = {}
        await progress(Stage.MATCHING, 0, len(top))
        for position, candidate in enumerate(top, start=1):
            assert candidate.chunk_vectors is not None
            similarities = candidate.chunk_vectors @ requirement_vectors.T  # (chunks, reqs)
            per_requirement = min(policy.evidence_chunks_per_requirement, len(candidate.chunks))
            pairs: list[tuple[str, str]] = []
            owners: list[tuple[int, int]] = []
            for j, requirement in enumerate(requirements):
                for chunk_index in np.argsort(-similarities[:, j])[:per_requirement]:
                    pairs.append((requirement.text, candidate.chunks[int(chunk_index)]))
                    owners.append((j, int(chunk_index)))

            probabilities = await asyncio.to_thread(self._reranker.score, pairs)

            best_score = [0.0] * len(requirements)
            best_chunk: list[int | None] = [None] * len(requirements)
            for (j, chunk_index), probability in zip(owners, probabilities, strict=True):
                if best_chunk[j] is None or probability > best_score[j]:
                    best_score[j] = float(probability)
                    best_chunk[j] = chunk_index

            results[candidate.candidate_id] = [
                RequirementMatch(
                    requirement_id=requirement.id,
                    score=round(best_score[j], 4),
                    evidence=None
                    if best_chunk[j] is None
                    else evidence_snippet(candidate.chunks[best_chunk[j]], requirement.text),
                )
                for j, requirement in enumerate(requirements)
            ]
            await progress(Stage.MATCHING, position, len(top))
        return results

    async def _fetch_social(
        self,
        top: list[_Candidate],
        handles: dict[str, ProfileHandles],
        progress: ProgressCallback,
    ) -> dict[str, SocialSnapshot]:
        """Stage 4: public profile lookups, concurrently (network bound)."""
        targets = [c for c in top if handles[c.candidate_id].any()]
        await progress(Stage.PROFILES, 0, len(targets))

        async def fetch(candidate: _Candidate) -> tuple[str, SocialSnapshot]:
            try:
                snapshot = await self._social.fetch(handles[candidate.candidate_id])
            except Exception:
                logger.exception("Profile lookup failed for candidate %s", candidate.candidate_id)
                snapshot = SocialSnapshot(errors=("Profile lookup failed.",))
            return candidate.candidate_id, snapshot

        snapshots: dict[str, SocialSnapshot] = {}
        for done, next_done in enumerate(asyncio.as_completed([fetch(c) for c in targets]), 1):
            candidate_id, snapshot = await next_done
            snapshots[candidate_id] = snapshot
            await progress(Stage.PROFILES, done, len(targets))
        return snapshots

    async def _generate_insights(
        self,
        job_description: str,
        requirements: list[Requirement],
        results: list[CandidateResult],
        texts: dict[str, str],
        progress: ProgressCallback,
    ) -> None:
        """Stage 6: LLM summaries for the top shortlisted candidates.

        Sequential on purpose: hosted LLMs limit tokens per minute (Groq's free tier allows
        8,000), and the client already retries rate-limited calls after ``retry-after``.
        Contact details are removed before any resume text leaves the server.
        """
        assert self._summarizer is not None
        targets = [r for r in results if r.shortlisted][: self.max_insights]
        await progress(Stage.INSIGHTS, 0, len(targets))
        title = job_title(job_description)
        for done, result in enumerate(targets, start=1):
            request = InsightRequest(
                job_title=title,
                requirements=tuple(requirements),
                matches=result.requirement_matches,
                resume_text=redact_contact_details(texts[result.candidate_id])[
                    : self._max_insight_chars
                ],
                language=result.language,
                final_score=result.final_score or 0.0,
            )
            try:
                result.insight = await self._summarizer.summarize(request)
            except InsightError as exc:
                result.insight_error = str(exc)
            except Exception:
                logger.exception("AI summary failed for candidate %s", result.candidate_id)
                result.insight_error = "The AI summary failed unexpectedly."
            await progress(Stage.INSIGHTS, done, len(targets))

    def _assemble(
        self,
        ordered: list[_Candidate],
        top: list[_Candidate],
        requirements: list[Requirement],
        matches: dict[str, list[RequirementMatch]],
        handles: dict[str, ProfileHandles],
        snapshots: dict[str, SocialSnapshot],
        policy: ScoringPolicy,
        options: ShortlistOptions,
    ) -> list[CandidateResult]:
        """Stage 5: combine scores, rank, and mark the shortlist."""
        top_ids = {c.candidate_id for c in top}
        results: list[CandidateResult] = []
        for retrieval_rank, candidate in enumerate(ordered, start=1):
            semantic = scoring.calibrate_similarity(
                candidate.semantic_similarity, policy.semantic_floor, policy.semantic_ceiling
            )
            result = CandidateResult(
                candidate_id=candidate.candidate_id,
                filename=candidate.document.filename,
                language=candidate.language,
                extraction_methods=candidate.document.methods,
                pages=candidate.document.pages,
                semantic_similarity=round(candidate.semantic_similarity, 4),
                semantic_score=round(100 * semantic, 1),
                retrieval_rank=retrieval_rank,
                profiles=handles[candidate.candidate_id],
                excerpt=evidence_snippet(
                    candidate.chunks[candidate.best_chunk], "", max_words=EXCERPT_WORDS
                ),
                warnings=candidate.document.warnings,
            )
            if candidate.candidate_id in top_ids:
                requirement_matches = matches[candidate.candidate_id]
                coverage = scoring.requirement_coverage(
                    (r, m.score) for r, m in zip(requirements, requirement_matches, strict=True)
                )
                relevance = scoring.relevance_score(semantic, coverage, policy)
                snapshot = snapshots.get(candidate.candidate_id) if options.include_social else None
                bonus = scoring.social_bonus(
                    snapshot.score if snapshot else None, policy.social_max_bonus
                )
                result.reranked = True
                result.coverage_score = round(100 * coverage, 1)
                result.requirement_matches = tuple(requirement_matches)
                result.relevance_score = round(relevance, 1)
                result.social = snapshot
                result.social_bonus = round(bonus, 1)
                result.final_score = round(scoring.final_score(relevance, bonus), 1)
            results.append(result)

        reranked = sorted(
            (r for r in results if r.reranked),
            key=lambda r: (
                -(r.final_score or 0.0),
                -(r.relevance_score or 0.0),  # breaks ties at the 100 cap
                -r.semantic_similarity,
            ),
        )
        for position, result in enumerate(reranked, start=1):
            result.final_rank = position
            result.shortlisted = position <= policy.shortlist_size
        return reranked + [r for r in results if not r.reranked]

    # --------------------------------------------------------------- validation
    def _validate(
        self,
        job_description: str,
        documents: Sequence[UploadedDocument],
        options: ShortlistOptions,
    ) -> str:
        text = job_description.strip()
        if not text:
            raise InvalidInputError("The job description must not be empty.")
        if len(text) > self.limits.max_job_description_chars:
            raise InvalidInputError(
                f"The job description is too long (max {self.limits.max_job_description_chars}"
                " characters)."
            )
        if options.include_insights and self._summarizer is None:
            raise InvalidInputError(
                "AI summaries are not configured on this server (set LLM_API_KEY)."
            )
        size = options.shortlist_size
        if size is not None and not 1 <= size <= self.limits.max_shortlist_size:
            raise InvalidInputError(
                f"Shortlist size must be between 1 and {self.limits.max_shortlist_size}."
            )
        self._validate_documents(documents)
        return text

    def _validate_documents(self, documents: Sequence[UploadedDocument]) -> None:
        if not documents:
            raise InvalidInputError("Upload at least one resume.")
        if len(documents) > self.limits.max_files:
            raise InvalidInputError(
                f"Too many files ({len(documents)}); the limit is {self.limits.max_files}."
            )
        total = 0
        for document in documents:
            if len(document.content) > self.limits.max_file_bytes:
                raise PayloadTooLargeError(
                    f"{document.filename} is larger than "
                    f"{self.limits.max_file_bytes // (1024 * 1024)} MB."
                )
            total += len(document.content)
        if total > self.limits.max_total_bytes:
            raise PayloadTooLargeError(
                f"Uploads exceed the total limit of {self.limits.max_total_bytes // (1024 * 1024)}"
                " MB."
            )
