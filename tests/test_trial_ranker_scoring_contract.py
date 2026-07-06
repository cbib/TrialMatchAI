"""Contract tests for trial_ranker.score_trial (audit finding C1, PR1).

A Violated exclusion hard-disqualifies a trial: it must rank strictly below every
trial that violates no exclusion. Eligible trials are scored in [0, 1] by the Met
fraction of the counted inclusion criteria — Met=1, Not Met=0, Unclear at partial
credit; Irrelevant is excluded.
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


def test_unclear_counts_as_partial_credit_irrelevant_excluded():
    # Unclear is counted at 0.5 (it dominates the label mix, so dropping it collapsed the
    # band); Irrelevant (criterion does not apply) is excluded. Met=1, Unclear=0.5 over the
    # two counted criteria -> (1 + 0.5) / 2 = 0.75.
    trial = {
        "Inclusion_Criteria_Evaluation": [
            {"Classification": "Met"},
            {"Classification": "Unclear"},
            {"Classification": "Irrelevant"},
        ],
        "Exclusion_Criteria_Evaluation": [],
    }
    assert score_trial(trial) == 0.75
    # A trial with only Irrelevant inclusions has nothing to score -> 0.0.
    assert score_trial({"Inclusion_Criteria_Evaluation": [{"Classification": "Irrelevant"}]}) == 0.0


def test_violated_exclusion_ranks_below_eligible():
    ranked = rank_trials([TRIAL_VIOLATED_EXCLUSION, TRIAL_PARTIAL_ELIGIBLE])
    assert ranked[0]["TrialID"] == "PARTIAL"
    assert ranked[-1]["TrialID"] == "VIOLATED"


def test_disqualified_ranks_below_even_a_zero_score_eligible_trial():
    # An all-Not-Met eligible trial (0.0) still outranks a disqualified one (-1.0).
    ranked = rank_trials([TRIAL_VIOLATED_EXCLUSION, TRIAL_ALL_NOT_MET])
    assert ranked[0]["TrialID"] == "NOT_MET"
    assert ranked[-1]["TrialID"] == "VIOLATED"


def test_score_trial_normalizes_classification_variants():
    """Model output varies in case, markdown, and trailing punctuation; a
    disqualifying Violated exclusion must still be detected (audit P0)."""
    for variant in ("violated", "**Violated**", "Violated ", "Violated.", "VIOLATED"):
        trial = {
            "Inclusion_Criteria_Evaluation": [{"Classification": "Met"}],
            "Exclusion_Criteria_Evaluation": [{"Classification": variant}],
        }
        assert score_trial(trial) == DISQUALIFIED_SCORE, variant

    # "Not Violated" must NOT disqualify (substring trap), and inclusion variants
    # in mixed case / whitespace must still count toward the Met fraction.
    assert score_trial(
        {
            "Inclusion_Criteria_Evaluation": [{"Classification": "met"}],
            "Exclusion_Criteria_Evaluation": [{"Classification": "Not Violated"}],
        }
    ) == 1.0
    assert score_trial(
        {
            "Inclusion_Criteria_Evaluation": [{"Classification": "MET"}, {"Classification": "not met "}],
            "Exclusion_Criteria_Evaluation": [],
        }
    ) == 0.5
