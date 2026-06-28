import os
from typing import Dict, List

from trialmatchai.utils.file_utils import read_json_file, write_json_file
from trialmatchai.utils.logging_config import setup_logging

logger = setup_logging(__name__)


def load_trial_data(
    json_folder: str, allowed_ids: set[str] | None = None
) -> List[Dict]:
    """Load per-trial eligibility outputs from a patient's result folder.

    ``allowed_ids`` scopes loading to the current shortlist, so per-trial files
    left over from a prior run with a different shortlist (e.g. after the index
    changed) are ignored instead of being scored into the final ranking.
    """
    trial_data = []
    for file_name in os.listdir(json_folder):
        # Only NCT-named files are trials; skip run sidecars written to the same
        # folder (keywords.json, patient_profile.json, first_level_scores.json,
        # rag_output.json), which would otherwise be scored as bogus trials.
        if file_name.endswith(".json") and file_name.upper().startswith("NCT"):
            file_path = os.path.join(json_folder, file_name)
            trial_id = os.path.splitext(file_name)[0]
            if allowed_ids is not None and trial_id not in allowed_ids:
                continue
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


def rank_trials(
    trial_data: List[Dict],
    *,
    first_level_scores: Dict[str, float] | None = None,
    second_level_scores: Dict[str, float] | None = None,
) -> List[Dict]:
    """Rank trials by eligibility, breaking ties deterministically.

    The eligibility score is coarse (small rationals), so many trials tie. Rather
    than let ties resolve by arbitrary filesystem order, break them by the
    continuous second-level reranker probability, then the first-level retrieval
    score, then the NCT id — a meaningful order within each eligibility bucket
    that is fully reproducible. (Tie-aware nDCG further ensures genuine ties are
    scored fairly regardless of this order.)
    """
    first_level_scores = first_level_scores or {}
    second_level_scores = second_level_scores or {}
    ranked_trials = []
    for trial in trial_data:
        trial_id = trial.get("TrialID", "Unknown")
        ranked_trials.append(
            {
                "TrialID": trial_id,
                "Score": score_trial(trial),
                "RerankerScore": float(second_level_scores.get(trial_id, 0.0)),
                "FirstLevelScore": float(first_level_scores.get(trial_id, 0.0)),
            }
        )
    # Descending eligibility -> reranker -> first-level; ascending NCT id last
    # (negate numeric keys so a single ascending sort gives the right order and
    # the NCT-id tie-break stays ascending).
    ranked_trials.sort(
        key=lambda x: (
            -x["Score"],
            -x["RerankerScore"],
            -x["FirstLevelScore"],
            x["TrialID"],
        )
    )
    return ranked_trials


def save_ranked_trials(ranked_trials: List[Dict], output_file: str):
    # Do NOT swallow write failures: the caller treats a returned-without-raising
    # call as a completed patient. A failed final write must surface so the
    # patient is counted as failed (and retried), not marked done with no marker.
    write_json_file({"RankedTrials": ranked_trials}, output_file)
    logger.info(f"Ranked trials saved to {output_file}")
