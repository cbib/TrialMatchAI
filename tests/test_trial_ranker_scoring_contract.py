"""Contract tests for trial_ranker.score_trial (audit finding C1, PR1).

A Violated exclusion hard-disqualifies a trial: it must rank strictly below every
trial that violates no exclusion. Eligible trials are scored in [0, 1] by the
fraction of decided inclusion criteria (Met or Not Met) that are Met.
See REFACTOR_PLAN.md (PR1).
"""

from trialmatchai.matching.trial_ranker import (
    DISQUALIFIED_SCORE,
    rank_trials,
    score_trial,
)

# Ineligible: every inclusion is Met, but an exclusion criterion is Violated.
TRIAL_VIOLATED_EXCLUSION = {
    "TrialID": "VIOLATED",
    "Inclusion_Criteria_Evaluation": [
        {"Classification": "Met"},
        {"Classification": "Met"},
    ],
    "Exclusion_Criteria_Evaluation": [
        {"Classification": "Not Violated"},
        {"Classification": "Violated"},
    ],
}

# Eligible partial match: half the decided inclusions are Met, no violations.
TRIAL_PARTIAL_ELIGIBLE = {
    "TrialID": "PARTIAL",
    "Inclusion_Criteria_Evaluation": [
        {"Classification": "Met"},
        {"Classification": "Not Met"},
    ],
    "Exclusion_Criteria_Evaluation": [],
}

# Eligible but a poor match: all inclusions Not Met, still violates nothing.
TRIAL_ALL_NOT_MET = {
    "TrialID": "NOT_MET",
    "Inclusion_Criteria_Evaluation": [
        {"Classification": "Not Met"},
        {"Classification": "Not Met"},
    ],
    "Exclusion_Criteria_Evaluation": [{"Classification": "Not Violated"}],
}


def test_violated_exclusion_is_disqualified():
    assert score_trial(TRIAL_VIOLATED_EXCLUSION) == DISQUALIFIED_SCORE


def test_eligible_scored_by_met_fraction():
    assert score_trial(TRIAL_PARTIAL_ELIGIBLE) == 0.5
    assert score_trial(TRIAL_ALL_NOT_MET) == 0.0
    assert score_trial(
        {"Inclusion_Criteria_Evaluation": [{"Classification": "Met"}]}
    ) == 1.0


def test_unclear_and_irrelevant_inclusions_are_ignored():
    # Only Met/Not Met count toward the fraction; Unclear/Irrelevant are excluded.
    trial = {
        "Inclusion_Criteria_Evaluation": [
            {"Classification": "Met"},
            {"Classification": "Unclear"},
            {"Classification": "Irrelevant"},
        ],
        "Exclusion_Criteria_Evaluation": [],
    }
    assert score_trial(trial) == 1.0


def test_violated_exclusion_ranks_below_eligible():
    ranked = rank_trials([TRIAL_VIOLATED_EXCLUSION, TRIAL_PARTIAL_ELIGIBLE])
    assert ranked[0]["TrialID"] == "PARTIAL"
    assert ranked[-1]["TrialID"] == "VIOLATED"


def test_disqualified_ranks_below_even_a_zero_score_eligible_trial():
    # An all-Not-Met eligible trial (0.0) still outranks a disqualified one (-1.0).
    ranked = rank_trials([TRIAL_VIOLATED_EXCLUSION, TRIAL_ALL_NOT_MET])
    assert ranked[0]["TrialID"] == "NOT_MET"
    assert ranked[-1]["TrialID"] == "VIOLATED"
