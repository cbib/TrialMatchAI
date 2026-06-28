"""Deterministic tie-break in rank_trials + shortlist scoping in load_trial_data."""

import json

from trialmatchai.matching.trial_ranker import load_trial_data, rank_trials


def _eligible():
    # Identical eligibility -> identical Score, so the tie-break decides order.
    return {
        "Inclusion_Criteria_Evaluation": [{"Classification": "Met"}],
        "Exclusion_Criteria_Evaluation": [],
    }


def test_tiebreak_reranker_then_firstlevel_then_nctid():
    trials = [{"TrialID": f"NCT-{c}", **_eligible()} for c in "ABCDE"]
    reranker = {"NCT-A": 0.9, "NCT-B": 0.9, "NCT-C": 0.5, "NCT-D": 0.5, "NCT-E": 0.5}
    first = {"NCT-A": 0.1, "NCT-B": 0.5, "NCT-C": 0.9, "NCT-D": 0.5, "NCT-E": 0.5}
    ranked = rank_trials(
        trials, first_level_scores=first, second_level_scores=reranker
    )
    order = [r["TrialID"] for r in ranked]
    # equal Score: reranker 0.9 first (B>A by first-level), then 0.5 (C by first-level),
    # then the full D/E tie resolves by ascending NCT id.
    assert order == ["NCT-B", "NCT-A", "NCT-C", "NCT-D", "NCT-E"]


def test_tiebreak_is_deterministic_regardless_of_input_order():
    a = [{"TrialID": f"NCT-{c}", **_eligible()} for c in "ABC"]
    reranker = {"NCT-A": 0.5, "NCT-B": 0.9, "NCT-C": 0.5}
    out1 = [r["TrialID"] for r in rank_trials(a, second_level_scores=reranker)]
    out2 = [r["TrialID"] for r in rank_trials(list(reversed(a)), second_level_scores=reranker)]
    assert out1 == out2 == ["NCT-B", "NCT-A", "NCT-C"]


def test_load_trial_data_scopes_to_allowed_ids(tmp_path):
    for nct in ["NCT1", "NCT2", "NCT3"]:
        (tmp_path / f"{nct}.json").write_text(
            json.dumps({"Inclusion_Criteria_Evaluation": []})
        )
    (tmp_path / "keywords.json").write_text("{}")  # sidecar must be ignored
    loaded = {
        t["TrialID"]
        for t in load_trial_data(str(tmp_path), allowed_ids={"NCT1", "NCT3"})
    }
    assert loaded == {"NCT1", "NCT3"}  # NCT2 (stale, off-shortlist) excluded
