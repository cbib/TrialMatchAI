"""Shortlist depth policies (matching/shortlist_depth.py).

The shortlist is where the pipeline loses most of its relevant trials, so the depth
decision must be explicit, bounded, and never silently change under the default config.
"""

import pytest

from trialmatchai.matching.shortlist_depth import (
    choose_shortlist_depth,
    depth_report,
    shortlist_config,
)


def _scores(values):
    return {f"NCT{i:04d}": v for i, v in enumerate(values)}


def test_default_policy_returns_the_fixed_depth_unchanged():
    """Enabling adaptive depth must be an explicit A/B, never a silent behaviour change."""
    depth = choose_shortlist_depth(
        first_level_scores=_scores([1.0] * 500),
        fixed_depth=196,
        upper_bound=300,
        search_config={},
    )
    assert depth == 196


def test_relative_to_max_keeps_trials_above_alpha_times_the_top_score():
    # Top score 1.0, alpha 0.25 -> keep while score >= 0.25, so the first four.
    depth = choose_shortlist_depth(
        first_level_scores=_scores([1.0, 0.8, 0.4, 0.25, 0.2, 0.1]),
        fixed_depth=2,
        upper_bound=100,
        search_config={"shortlist": {"policy": "relative_to_max", "min_depth": 1}},
    )
    assert depth == 4


def test_peaked_curve_gets_less_depth_than_flat_curve():
    """The whole point: a confident retrieval needs fewer trials than an ambiguous one."""
    cfg = {"shortlist": {"policy": "relative_to_max", "min_depth": 1}}
    peaked = choose_shortlist_depth(
        first_level_scores=_scores([1.0] + [0.01] * 99),
        fixed_depth=50,
        upper_bound=100,
        search_config=cfg,
    )
    flat = choose_shortlist_depth(
        first_level_scores=_scores([1.0] * 100),
        fixed_depth=50,
        upper_bound=100,
        search_config=cfg,
    )
    assert peaked == 1
    assert flat == 100
    assert peaked < flat


def test_depth_is_clamped_by_min_depth_and_upper_bound():
    cfg = {"shortlist": {"policy": "relative_to_max", "min_depth": 20, "max_depth": 40}}
    # A single dominant trial would give depth 1; the floor lifts it to min_depth.
    assert (
        choose_shortlist_depth(
            first_level_scores=_scores([1.0] + [0.001] * 99),
            fixed_depth=10,
            upper_bound=100,
            search_config=cfg,
        )
        == 20
    )
    # A flat curve would give 100; max_depth caps it at 40.
    assert (
        choose_shortlist_depth(
            first_level_scores=_scores([1.0] * 100),
            fixed_depth=10,
            upper_bound=100,
            search_config=cfg,
        )
        == 40
    )


def test_upper_bound_always_wins_over_configured_max_depth():
    """upper_bound is what the reasoner can actually consume; exceeding it drops trials
    silently from the final ranking."""
    depth = choose_shortlist_depth(
        first_level_scores=_scores([1.0] * 500),
        fixed_depth=10,
        upper_bound=30,
        search_config={
            "shortlist": {"policy": "relative_to_max", "min_depth": 1, "max_depth": 400}
        },
    )
    assert depth == 30


def test_missing_scores_degrade_to_fixed_depth():
    """A resumed run can lack first_level_scores.json; guessing a depth would be worse."""
    for scores in (None, {}):
        depth = choose_shortlist_depth(
            first_level_scores=scores,
            fixed_depth=77,
            upper_bound=300,
            search_config={"shortlist": {"policy": "relative_to_max"}},
        )
        assert depth == 77


def test_nonpositive_top_score_keeps_the_whole_pool():
    depth = choose_shortlist_depth(
        first_level_scores=_scores([0.0, 0.0, 0.0]),
        fixed_depth=1,
        upper_bound=100,
        search_config={"shortlist": {"policy": "relative_to_max", "min_depth": 1}},
    )
    assert depth == 3


def test_unknown_policy_falls_back_to_fixed(caplog):
    assert shortlist_config({"shortlist": {"policy": "wishful"}})["policy"] == "fixed"
    depth = choose_shortlist_depth(
        first_level_scores=_scores([1.0] * 100),
        fixed_depth=12,
        upper_bound=100,
        search_config={"shortlist": {"policy": "wishful"}},
    )
    assert depth == 12


@pytest.mark.parametrize("search_config", [None, {}, {"shortlist": None}])
def test_absent_config_is_tolerated(search_config):
    assert shortlist_config(search_config)["policy"] == "fixed"


def test_depth_report_records_the_decision():
    report = depth_report(
        chosen=40,
        fixed_depth=196,
        first_level_scores=_scores([1.0, 0.5, 0.2]),
        search_config={"shortlist": {"policy": "relative_to_max"}},
    )
    assert report["policy"] == "relative_to_max"
    assert report["chosen_depth"] == 40
    assert report["fixed_depth"] == 196  # what the old sizing would have used
    assert report["candidate_pool"] == 3
    assert report["top_score"] == 1.0
