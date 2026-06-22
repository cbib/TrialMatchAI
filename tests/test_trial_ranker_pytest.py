import json

from trialmatchai.matching.trial_ranker import (
    load_trial_data,
    rank_trials,
    save_ranked_trials,
    score_trial,
)


def test_score_trial_basic():
    # A violated exclusion hard-disqualifies the trial regardless of inclusions.
    trial = {
        "Inclusion_Criteria_Evaluation": [
            {"Classification": "Met"},
            {"Classification": "Met"},
        ],
        "Exclusion_Criteria_Evaluation": [
            {"Classification": "Not Violated"},
            {"Classification": "Violated"},
        ],
    }
    assert score_trial(trial) == -1.0


def test_rank_trials_orders_by_score():
    trial_data = [
        {"TrialID": "T1", "Inclusion_Criteria_Evaluation": [], "Exclusion_Criteria_Evaluation": []},
        {
            "TrialID": "T2",
            "Inclusion_Criteria_Evaluation": [{"Classification": "Met"}],
            "Exclusion_Criteria_Evaluation": [],
        },
    ]
    ranked = rank_trials(trial_data)
    assert ranked[0]["TrialID"] == "T2"


def test_load_and_save_ranked_trials(tmp_path):
    trial_folder = tmp_path / "trials"
    trial_folder.mkdir()
    (trial_folder / "NCT0001.json").write_text(
        json.dumps(
            {"Inclusion_Criteria_Evaluation": [], "Exclusion_Criteria_Evaluation": []}
        )
    )
    (trial_folder / "NCT0002.json").write_text(
        json.dumps(
            {
                "Inclusion_Criteria_Evaluation": [{"Classification": "Met"}],
                "Exclusion_Criteria_Evaluation": [],
            }
        )
    )
    # Run sidecars in the same folder must NOT be loaded as trials.
    (trial_folder / "keywords.json").write_text(json.dumps({"keywords": ["x"]}))
    (trial_folder / "patient_profile.json").write_text(json.dumps({"id": "p1"}))

    trials = load_trial_data(str(trial_folder))
    assert {t["TrialID"] for t in trials} == {"NCT0001", "NCT0002"}

    ranked = rank_trials(trials)
    out_file = tmp_path / "ranked.json"
    save_ranked_trials(ranked, str(out_file))

    saved = json.loads(out_file.read_text())
    assert "RankedTrials" in saved
    assert saved["RankedTrials"][0]["TrialID"] == "NCT0002"
