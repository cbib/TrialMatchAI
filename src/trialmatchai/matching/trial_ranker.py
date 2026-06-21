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


def score_trial(trial: Dict) -> float:
    def calculate_ratio(
        criteria_list, positive_classifications, negative_classifications
    ):
        criteria_to_exclude = ["Irrelevant", "Unclear"]
        criteria_list = [
            c
            for c in criteria_list
            if c.get("Classification") not in criteria_to_exclude
        ]
        total_criteria = len(criteria_list)
        if total_criteria == 0:
            return 0.0
        positive_count = sum(
            1
            for c in criteria_list
            if c.get("Classification") in positive_classifications
        )
        negative_count = sum(
            1
            for c in criteria_list
            if c.get("Classification") in negative_classifications
        )
        penalty_factor_negative = 1.0
        reward_factor_positive = 1.0
        score = (
            reward_factor_positive * positive_count
            - penalty_factor_negative * negative_count
        ) / total_criteria
        return score

    inclusion_criteria = trial.get("Inclusion_Criteria_Evaluation", [])
    exclusion_criteria = trial.get("Exclusion_Criteria_Evaluation", [])
    inclusion_ratio = calculate_ratio(
        inclusion_criteria, ["Met", "Not Violated"], ["Violated", "Not Met"]
    )
    exclusion_ratio = calculate_ratio(
        exclusion_criteria, ["Not Violated", "Met"], ["Violated"]
    )
    return (inclusion_ratio + exclusion_ratio) / 2


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
