import pytest

from resume_shortlister.domain.models import CodeChefStats, GitHubStats, LeetCodeStats, Requirement
from resume_shortlister.domain.scoring import (
    MatchBand,
    ScoringPolicy,
    calibrate_similarity,
    codechef_score,
    final_score,
    github_score,
    leetcode_score,
    match_band,
    relevance_score,
    requirement_coverage,
    saturate,
    social_bonus,
    social_score,
)


def test_calibrate_similarity_clips_to_unit_interval():
    assert calibrate_similarity(0.70, 0.78, 0.93) == 0.0
    assert calibrate_similarity(0.99, 0.78, 0.93) == 1.0
    assert calibrate_similarity(0.855, 0.78, 0.93) == pytest.approx(0.5)


def test_coverage_is_a_weighted_mean_with_optional_items_at_half_weight():
    required, optional = Requirement(1, "Kotlin"), Requirement(2, "Flutter", optional=True)
    assert requirement_coverage([(required, 1.0), (optional, 0.0)]) == pytest.approx(1 / 1.5)
    assert requirement_coverage([]) == 0.0


def test_relevance_blends_semantic_and_coverage():
    policy = ScoringPolicy(semantic_weight=0.3, coverage_weight=0.7)
    assert relevance_score(1.0, 0.0, policy) == pytest.approx(30.0)
    assert relevance_score(0.5, 1.0, policy) == pytest.approx(85.0)


def test_saturate_has_diminishing_returns():
    assert saturate(0, 10) == 0
    assert saturate(10, 10) == 0.5
    assert saturate(-5, 10) == 0
    assert saturate(1_000_000, 10) < 1


def test_platform_scores_grow_with_activity():
    weak = GitHubStats("a", public_repos=1, followers=0, total_stars=0)
    strong = GitHubStats("b", public_repos=40, followers=200, total_stars=900)
    assert 0 < github_score(weak) < github_score(strong) < 100
    assert leetcode_score(LeetCodeStats("x", 300, 100, 150, 50)) > leetcode_score(
        LeetCodeStats("y", 300, 300, 0, 0)
    )
    assert codechef_score(CodeChefStats("c", rating=None, stars=None, problems_solved=None)) is None
    assert codechef_score(CodeChefStats("c", rating=1800, stars=4, problems_solved=None)) > 0


def test_social_score_uses_the_strongest_profile_so_extra_profiles_never_hurt():
    github = GitHubStats("a", public_repos=40, followers=200, total_stars=900)
    weak_leetcode = LeetCodeStats("a", 3, 3, 0, 0)
    alone = social_score(github, None, None)
    assert social_score(github, weak_leetcode, None) == alone
    assert social_score(None, None, None) is None


def test_final_score_stays_on_the_0_100_scale():
    assert final_score(72.5, 5.0) == 77.5
    assert final_score(92.5, 7.7) == 100.0  # regression: used to show 100.2


def test_social_bonus_is_zero_without_data_and_capped():
    assert social_bonus(None, 10) == 0.0
    assert social_bonus(50, 10) == 5.0
    assert social_bonus(250, 10) == 10.0


@pytest.mark.parametrize(
    ("score", "band"),
    [(None, None), (80, MatchBand.STRONG), (45, MatchBand.PARTIAL), (10, MatchBand.WEAK)],
)
def test_match_bands(score, band):
    assert match_band(score) == band


@pytest.mark.parametrize(
    "kwargs",
    [
        {"retrieval_top_k": 0},
        {"semantic_weight": 0, "coverage_weight": 0},
        {"semantic_floor": 0.9, "semantic_ceiling": 0.8},
        {"social_max_bonus": -1},
    ],
)
def test_invalid_policies_are_rejected(kwargs):
    with pytest.raises(ValueError, match=r"must|require|at least"):
        ScoringPolicy(**kwargs)
