"""Core domain entities and value objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class ExtractionMethod(StrEnum):
    """How the text of a document was obtained."""

    TEXT_LAYER = "text_layer"  # embedded PDF text
    DOCX = "docx"
    PLAIN_TEXT = "plain_text"
    CHANDRA = "chandra"  # Chandra OCR (vision-language model)
    TESSERACT = "tesseract"  # Tesseract OCR (fallback)

    @property
    def is_ocr(self) -> bool:
        return self in (ExtractionMethod.CHANDRA, ExtractionMethod.TESSERACT)


@dataclass(frozen=True, slots=True)
class ExtractedDocument:
    """Plain text recovered from an uploaded resume."""

    filename: str
    text: str
    methods: tuple[ExtractionMethod, ...]
    pages: int = 1
    links: tuple[str, ...] = ()  # hyperlink targets embedded in the file (PDF/DOCX)
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SkippedDocument:
    """An uploaded file that could not be ranked, and why."""

    filename: str
    reason: str


@dataclass(frozen=True, slots=True)
class Requirement:
    """A single requirement parsed from the job description."""

    id: int  # 1-based, shown as R1, R2, ...
    text: str
    optional: bool = False  # from a "nice to have" / "preferred" section

    @property
    def weight(self) -> float:
        return 0.5 if self.optional else 1.0


@dataclass(frozen=True, slots=True)
class RequirementMatch:
    """How well a resume satisfies one requirement, with the supporting passage."""

    requirement_id: int
    score: float  # 0..1 match probability from the reranker
    evidence: str | None = None


@dataclass(frozen=True, slots=True)
class ProfileHandles:
    """Public coding-profile usernames found in a resume."""

    github: str | None = None
    leetcode: str | None = None
    codechef: str | None = None

    def any(self) -> bool:
        return bool(self.github or self.leetcode or self.codechef)


@dataclass(frozen=True, slots=True)
class GitHubStats:
    username: str
    public_repos: int
    followers: int
    total_stars: int  # stars on the user's own (non-fork) repositories


@dataclass(frozen=True, slots=True)
class LeetCodeStats:
    username: str
    solved_total: int
    easy: int
    medium: int
    hard: int


@dataclass(frozen=True, slots=True)
class CodeChefStats:
    username: str
    rating: int | None
    stars: int | None
    problems_solved: int | None


@dataclass(frozen=True, slots=True)
class SocialSnapshot:
    """Public coding activity for one candidate. ``score`` is None when no data exists."""

    github: GitHubStats | None = None
    leetcode: LeetCodeStats | None = None
    codechef: CodeChefStats | None = None
    score: float | None = None  # 0..100 over the platforms that returned data
    errors: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CandidateInsight:
    """LLM-written, recruiter-facing assessment of one shortlisted candidate."""

    summary: str
    strengths: tuple[str, ...] = ()
    gaps: tuple[str, ...] = ()
    interview_questions: tuple[str, ...] = ()


@dataclass(slots=True)
class CandidateResult:
    """Everything the system knows about one ranked resume."""

    candidate_id: str
    filename: str
    language: str | None
    extraction_methods: tuple[ExtractionMethod, ...]
    pages: int
    semantic_similarity: float  # raw cosine similarity (best passage vs. job description)
    semantic_score: float  # 0..100, calibrated for display and blending
    retrieval_rank: int  # 1-based rank after stage 1 (semantic retrieval)
    reranked: bool = False  # made it into stage 2 (requirement matching)
    coverage_score: float | None = None  # 0..100 weighted requirement coverage
    requirement_matches: tuple[RequirementMatch, ...] = ()
    relevance_score: float | None = None  # 0..100 blend of semantic + coverage
    profiles: ProfileHandles = field(default_factory=ProfileHandles)
    social: SocialSnapshot | None = None
    social_bonus: float = 0.0  # never negative: missing profiles are not penalised
    final_score: float | None = None  # relevance + social bonus
    final_rank: int | None = None  # 1-based rank among reranked candidates
    shortlisted: bool = False
    excerpt: str = ""  # most relevant passage
    warnings: tuple[str, ...] = ()
    insight: CandidateInsight | None = None  # optional AI summary (never affects the score)
    insight_error: str | None = None


@dataclass(slots=True)
class ShortlistResult:
    """Output of one shortlisting run."""

    requirements: tuple[Requirement, ...]
    candidates: tuple[CandidateResult, ...]  # shortlisted first (by final rank), then the rest
    skipped: tuple[SkippedDocument, ...]
    shortlist_size: int
    social_enabled: bool
    total_documents: int
    timings: dict[str, float] = field(default_factory=dict)
    insights_enabled: bool = False

    @property
    def shortlisted(self) -> list[CandidateResult]:
        return [c for c in self.candidates if c.shortlisted]

    @property
    def by_retrieval(self) -> list[CandidateResult]:
        return sorted(self.candidates, key=lambda c: c.retrieval_rank)
