import os
import json
import logging
from typing import Dict, List

########################################
# Logging Configuration
########################################
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

########################################
# Ranking Script
########################################


def load_trial_data(json_folder: str) -> List[Dict]:
    """
    Load all JSON files from the specified folder.
    """
    trial_data = []
    for file_name in os.listdir(json_folder):
        if file_name.endswith(".json"):
            file_path = os.path.join(json_folder, file_name)
            trial_id = os.path.splitext(file_name)[
                0
            ]  # Extract TrialID from the file name
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    trial = json.load(f)
                    trial["TrialID"] = trial_id  # Add TrialID to the trial data
                    trial_data.append(trial)
            except Exception as e:
                logger.error(f"Failed to load {file_name}: {e}")
    return trial_data


def score_trial(trial: Dict) -> float:
    """
    Calculate a composite score for a trial based on updated inclusion and exclusion criteria.
    The score prioritizes trials with more satisfied inclusion criteria and fewer violated exclusion criteria,
    while handling "Insufficient Information" as neutral.
    """

    def calculate_ratio(
        criteria_list, positive_classifications, negative_classifications
    ):
        criteria_to_exclude = ["Irrelevant", "Unclear"]
        criteria_list = [
            c
            for c in criteria_list
            if c.get("Classification") not in criteria_to_exclude
        ]
        total_criteria = len(
            criteria_list
        )  # Adjusted total excludes 'Insufficient Information'
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

        # Weights and penalties
        penalty_factor_negative = 1.0  # Adjust as needed for stronger penalties
        reward_factor_positive = 1.0

        # Calculate score using positive and negative contributions
        score = (
            reward_factor_positive * positive_count
            - penalty_factor_negative * negative_count
        ) / total_criteria

        return score

    # Calculate inclusion and exclusion ratios
    inclusion_criteria = trial.get("Inclusion_Criteria_Evaluation", [])
    exclusion_criteria = trial.get("Exclusion_Criteria_Evaluation", [])

    inclusion_ratio = calculate_ratio(
        inclusion_criteria,
        positive_classifications=["Met", "Not Violated"],
        negative_classifications=["Violated", "Not Met"],
    )

    exclusion_ratio = calculate_ratio(
        exclusion_criteria,
        positive_classifications=["Not Violated", "Met"],
        negative_classifications=["Violated"],
    )

    # Composite score as an average of inclusion and exclusion ratios
    composite_score = (inclusion_ratio + exclusion_ratio) / 2

    return composite_score


def rank_trials(trial_data: List[Dict]) -> List[Dict]:
    """
    Rank trials based on their composite scores.
    """
    ranked_trials = []
    for trial in trial_data:
        trial_id = trial.get("TrialID", "Unknown")
        score = score_trial(trial)
        ranked_trials.append({"TrialID": trial_id, "Score": score})

    # Sort trials by composite score in descending order
    ranked_trials.sort(key=lambda x: x["Score"], reverse=True)
    return ranked_trials


def save_ranked_trials(ranked_trials: List[Dict], output_file: str):
    """
    Save the ranked trials to a JSON file.
    """
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(ranked_trials, f, indent=4, ensure_ascii=False)
        logger.info(f"Ranked trials saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save ranked trials: {e}")
