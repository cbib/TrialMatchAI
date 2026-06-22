"""Characterization + contract tests for trial_ranker.score_trial.

PR0 (safety net): these lock the CURRENT behavior of the scorer and pin the
DESIRED behavior of the eligibility-scoring contract as an xfail. PR1 fixes
score_trial so that a Violated exclusion becomes a hard disqualifier; at that
point the xfail below flips to pass and the characterization test is updated.

See REFACTOR_PLAN.md (PR1) and audit finding C1.
"""

import pytest

from trialmatchai.matching.trial_ranker import rank_trials, score_trial

# A trial the patient is clearly ineligible for: every inclusion is Met, but an
# exclusion criterion is Violated.
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

# A trial the patient partially matches and violates no exclusions.
TRIAL_PARTIAL_ELIGIBLE = {
    "TrialID": "PARTIAL",
    "Inclusion_Criteria_Evaluation": [
        {"Classification": "Met"},
        {"Classification": "Not Met"},
    ],
    "Exclusion_Criteria_Evaluation": [],
}


def test_characterization_violated_exclusion_only_partially_penalized():
    """CURRENT behavior: a Violated exclusion is averaged away rather than
    disqualifying. This documents the C1 bug and will be updated in PR1."""
    # inclusion_ratio = (2 - 0) / 2 = 1.0
    # exclusion_ratio = (1 - 1) / 2 = 0.0  (Not Violated positive, Violated negative)
    # score = (1.0 + 0.0) / 2 = 0.5
    assert score_trial(TRIAL_VIOLATED_EXCLUSION) == 0.5


def test_characterization_violated_exclusion_outranks_eligible_trial():
    """CURRENT (buggy) ranking: the trial with a Violated exclusion (0.5) ranks
    ABOVE the violation-free partial match (0.0)."""
    ranked = rank_trials([TRIAL_PARTIAL_ELIGIBLE, TRIAL_VIOLATED_EXCLUSION])
    assert ranked[0]["TrialID"] == "VIOLATED"


@pytest.mark.xfail(reason="PR1: a Violated exclusion must hard-disqualify", strict=True)
def test_contract_violated_exclusion_ranks_below_eligible():
    """DESIRED contract (PR1): any trial with a Violated exclusion ranks strictly
    below any trial that violates no exclusions."""
    ranked = rank_trials([TRIAL_VIOLATED_EXCLUSION, TRIAL_PARTIAL_ELIGIBLE])
    assert ranked[-1]["TrialID"] == "VIOLATED"
