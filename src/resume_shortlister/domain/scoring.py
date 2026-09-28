"""Scoring rules: how model outputs and profile statistics become explainable scores.

final score = min(100, relevance (0-100) + profile bonus (0..social_max_bonus, opt-in))
relevance   = weighted blend of calibrated semantic similarity and requirement coverage
coverage    = weighted mean over requirements of the best reranker match probability
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from resume_shortlister.domain.models import (
    CodeChefStats,
    GitHubStats,
    LeetCodeStats,
    Requirement,
)

STRONG_MATCH_THRESHOLD = 60.0
PARTIAL_MATCH_THRESHOLD = 30.0


@dataclass(frozen=True, slots=True)
class ScoringPolicy:
    """Tunable knobs of the ranking pipeline (all overridable via environment variables)."""

    retrieval_top_k: int = 20  # candidates promoted from stage 1 to requirement matching
    shortlist_size: int = 10
    semantic_weight: float = 0.3
    coverage_weight: float = 0.7
    # Cosine bounds used to calibrate the bi-encoder similarity onto 0..1. Defaults are
    # measured for intfloat/multilingual-e5-small (unrelated resume ~0.80, strong ~0.91).
    semantic_floor: float = 0.78
    semantic_ceiling: float = 0.93
    social_max_bonus: float = 10.0
    max_requirements: int = 15
    # Short passages keep each reranker input focused: measured coverage for strong
    # candidates rose from ~75 to ~92 when going from 120- to 50-word passages.
    chunk_words: int = 50
    chunk_overlap_words: int = 15
    evidence_chunks_per_requirement: int = 2  # passages per requirement sent to the reranker

    def __post_init__(self) -> None:
        if self.retrieval_top_k < 1 or self.shortlist_size < 1:
            raise ValueError("retrieval_top_k and shortlist_size must be >= 1")
        if self.semantic_weight < 0 or self.coverage_weight < 0:
            raise ValueError("score weights must be non-negative")
        if self.semantic_weight + self.coverage_weight == 0:
            raise ValueError("at least one score weight must be positive")
        if not -1.0 <= self.semantic_floor < self.semantic_ceiling <= 1.0:
            raise ValueError("require -1 <= semantic_floor < semantic_ceiling <= 1")
        if self.social_max_bonus < 0:
            raise ValueError("social_max_bonus must be non-negative")
        if self.max_requirements < 1 or self.evidence_chunks_per_requirement < 1:
            raise ValueError("max_requirements and evidence_chunks_per_requirement must be >= 1")

    def with_shortlist_size(self, size: int | None) -> ScoringPolicy:
        return dataclasses.replace(self, shortlist_size=size) if size else self


class MatchBand(StrEnum):
    STRONG = "strong"
    PARTIAL = "partial"
    WEAK = "weak"


def match_band(score: float | None) -> MatchBand | None:
    """Human-readable interpretation of a 0-100 score."""
    if score is None:
        return None
    if score >= STRONG_MATCH_THRESHOLD:
        return MatchBand.STRONG
    if score >= PARTIAL_MATCH_THRESHOLD:
        return MatchBand.PARTIAL
    return MatchBand.WEAK


def calibrate_similarity(similarity: float, floor: float, ceiling: float) -> float:
    """Map a raw cosine similarity onto 0..1 using model-specific bounds (clipped)."""
    return min(1.0, max(0.0, (similarity - floor) / (ceiling - floor)))


def requirement_coverage(matches: Iterable[tuple[Requirement, float]]) -> float:
    """Weighted mean (0..1) of per-requirement match probabilities; optional ones count half."""
    total_weight = 0.0
    total = 0.0
    for requirement, probability in matches:
        total_weight += requirement.weight
        total += requirement.weight * min(1.0, max(0.0, probability))
    return total / total_weight if total_weight else 0.0


def relevance_score(semantic: float, coverage: float, policy: ScoringPolicy) -> float:
    """Blend calibrated semantic similarity and coverage (both 0..1) into a 0..100 score."""
    weight_sum = policy.semantic_weight + policy.coverage_weight
    blended = policy.semantic_weight * semantic + policy.coverage_weight * coverage
    return 100.0 * blended / weight_sum


def saturate(value: float, half_point: float) -> float:
    """Diminishing returns: 0 -> 0, ``half_point`` -> 0.5, infinity -> 1."""
    value = max(0.0, value)
    return value / (value + half_point)


def github_score(stats: GitHubStats) -> float:
    return 100.0 * (
        0.40 * saturate(stats.total_stars, 50)
        + 0.35 * saturate(stats.public_repos, 15)
        + 0.25 * saturate(stats.followers, 25)
    )


def leetcode_score(stats: LeetCodeStats) -> float:
    weighted = stats.easy + 2 * stats.medium + 3 * stats.hard
    if weighted == 0:
        weighted = stats.solved_total  # per-difficulty breakdown unavailable
    return 100.0 * saturate(weighted, 250)


def codechef_score(stats: CodeChefStats) -> float | None:
    parts: list[tuple[float, float]] = []
    if stats.rating is not None:
        parts.append((0.7, saturate(stats.rating - 1000, 600)))
    if stats.problems_solved is not None:
        parts.append((0.3, saturate(stats.problems_solved, 150)))
    if not parts:
        return None
    weight_sum = sum(weight for weight, _ in parts)
    return 100.0 * sum(weight * value for weight, value in parts) / weight_sum


def social_score(
    github: GitHubStats | None, leetcode: LeetCodeStats | None, codechef: CodeChefStats | None
) -> float | None:
    """Score of the candidate's strongest public profile (0..100), or None without data.

    Using the strongest profile keeps the bonus monotonic: listing an extra, weaker
    profile never lowers it, and candidates active on one platform are not penalised.
    """
    scores = [
        score
        for score in (
            github_score(github) if github else None,
            leetcode_score(leetcode) if leetcode else None,
            codechef_score(codechef) if codechef else None,
        )
        if score is not None
    ]
    return max(scores) if scores else None


def final_score(relevance: float, bonus: float) -> float:
    """Relevance plus the profile bonus, capped at 100 so scores stay on one 0-100 scale.

    Ties at the cap are broken by relevance when ranking.
    """
    return min(100.0, relevance + bonus)


def social_bonus(score: float | None, max_bonus: float) -> float:
    """Additive bonus in [0, max_bonus]. Candidates without profile data get 0, never less."""
    if score is None:
        return 0.0
    return max_bonus * min(100.0, max(0.0, score)) / 100.0
