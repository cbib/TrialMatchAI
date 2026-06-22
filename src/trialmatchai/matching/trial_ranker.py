import os
from typing import Dict, List

from trialmatchai.utils.file_utils import read_json_file, write_json_file
from trialmatchai.utils.logging_config import setup_logging

logger = setup_logging(__name__)


def load_trial_data(json_folder: str) -> List[Dict]:
    trial_data = []
    for file_name in os.listdir(json_folder):
        if file_name.endswith(".json"):
            file_path = os.path.join(json_folder, file_name)
            trial_id = os.path.splitext(file_name)[0]
            try:
                trial = read_json_file(file_path)
                trial["TrialID"] = trial_id
                trial_data.append(trial)
            except Exception as e:
                logger.error(f"Failed to load {file_name}: {e}")
    return trial_data


# Eligibility scoring contract (see REFACTOR_PLAN.md PR1, audit finding C1).
#
# The eligibility model classifies each inclusion criterion as one of
# {Met, Not Met, Unclear, Irrelevant} and each exclusion criterion as one of
# {Violated, Not Violated, Unclear, Irrelevant}. A single Violated exclusion
# makes the patient ineligible, so it HARD-DISQUALIFIES the trial rather than
# being averaged against the inclusion score (the previous behavior, which let a
# violated trial outrank an eligible one). Eligible trials are ranked in [0, 1]
# by the fraction of decided inclusion criteria (Met or Not Met) that are Met.
DISQUALIFIED_SCORE = -1.0

_DECIDED_INCLUSION = {"Met", "Not Met"}


def score_trial(trial: Dict) -> float:
    inclusion = [
        c.get("Classification") for c in trial.get("Inclusion_Criteria_Evaluation", [])
    ]
    exclusion = [
        c.get("Classification") for c in trial.get("Exclusion_Criteria_Evaluation", [])
    ]

    # Any violated exclusion is a hard disqualifier: rank below all eligible trials.
    if "Violated" in exclusion:
        return DISQUALIFIED_SCORE

    # Eligible: score by the fraction of decided inclusion criteria that are Met.
    decided = [c for c in inclusion if c in _DECIDED_INCLUSION]
    if not decided:
        return 0.0
    met = sum(1 for c in decided if c == "Met")
    return met / len(decided)


def rank_trials(trial_data: List[Dict]) -> List[Dict]:
    ranked_trials = []
    for trial in trial_data:
        trial_id = trial.get("TrialID", "Unknown")
        score = score_trial(trial)
        ranked_trials.append({"TrialID": trial_id, "Score": score})
    ranked_trials.sort(key=lambda x: x["Score"], reverse=True)
    return ranked_trials


def save_ranked_trials(ranked_trials: List[Dict], output_file: str):
    try:
        write_json_file({"RankedTrials": ranked_trials}, output_file)
        logger.info(f"Ranked trials saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save ranked trials: {e}")
