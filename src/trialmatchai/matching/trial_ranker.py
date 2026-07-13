import os
from typing import Dict, List

from trialmatchai.utils.file_utils import read_json_file, write_json_file
from trialmatchai.utils.logging_config import setup_logging

logger = setup_logging(__name__)


def load_trial_data(
    json_folder: str, allowed_ids: set[str] | None = None
) -> List[Dict]:
    """Load per-trial eligibility outputs from a patient's result folder.

    ``allowed_ids`` scopes to the current shortlist so stale per-trial files from a
    prior run (different shortlist) are not scored into the final ranking.
    """
    trial_data = []
    for file_name in os.listdir(json_folder):
        # Only NCT-named files are trials; skip run sidecars (keywords.json, etc.)
        # that would otherwise be scored as bogus trials.
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


# Eligibility scoring contract (REFACTOR_PLAN.md PR1, audit C1): a single Violated exclusion
# HARD-DISQUALIFIES (not averaged in, which let a violated trial outrank an eligible one);
# eligible trials score [0, 1] by the fraction of decided inclusion criteria (Met/Not Met) Met.
DISQUALIFIED_SCORE = -1.0

# "Unclear" (info insufficient to decide) is the dominant classification. Partial credit — vs
# dropping it — keeps a mostly-Unclear trial off the all-Met band (which let merely-relevant
# trials outrank eligible ones). "Irrelevant" (criterion does not apply) is still excluded.
_UNCLEAR_CREDIT = 0.5

# Classifications vary in case/markdown/punctuation ("**Violated**", "Met."); normalize
# so a disqualifying exclusion is never missed on a formatting variant.
_CLASSIFICATION_STRIP = " \t\r\n*_`.,;:!\"'"


def _normalize_classification(value: object) -> str:
    return value.strip(_CLASSIFICATION_STRIP).casefold() if isinstance(value, str) else ""


def score_trial(trial: Dict) -> float:
    inclusion = trial.get("Inclusion_Criteria_Evaluation")
    exclusion = trial.get("Exclusion_Criteria_Evaluation")
    # An error-output trial (no eval lists) was never assessed; disqualify rather than
    # score 0.0, which would rank it above genuinely evaluated trials.
    if not inclusion and not exclusion:
        return DISQUALIFIED_SCORE if "error" in trial else 0.0

    # Guard off-shape LLM output (bare strings/nulls): score only dict criteria, else
    # c.get raises AttributeError and discards every ranked trial for the patient.
    inclusion = [
        _normalize_classification(c.get("Classification"))
        for c in (inclusion or [])
        if isinstance(c, dict)
    ]
    exclusion = [
        _normalize_classification(c.get("Classification"))
        for c in (exclusion or [])
        if isinstance(c, dict)
    ]

    # Any violated exclusion is a hard disqualifier: rank below all eligible trials.
    if "violated" in exclusion:
        return DISQUALIFIED_SCORE

    # Eligible: Met fraction of the counted inclusion criteria. "Met"=1, "Not Met"=0, and
    # "Unclear"=_UNCLEAR_CREDIT are counted; "Irrelevant" (criterion does not apply) is excluded.
    numerator = 0.0
    denominator = 0
    for classification in inclusion:
        if classification == "met":
            numerator += 1.0
            denominator += 1
        elif classification == "not met":
            denominator += 1
        elif classification == "unclear":
            numerator += _UNCLEAR_CREDIT
            denominator += 1
    if denominator == 0:
        return 0.0
    return numerator / denominator


# Fraction of the gap to the next eligibility band the reranker score may occupy. Strictly
# < 1.0 keeps a high-reranker trial from crossing bands: CoT stays primary, reranker only
# breaks within-band ties.
_BLEND_HEADROOM = 0.9


def rank_trials(
    trial_data: List[Dict],
    *,
    first_level_scores: Dict[str, float] | None = None,
    second_level_scores: Dict[str, float] | None = None,
) -> List[Dict]:
    """Rank by eligibility, folding the reranker score in to break the coarse ties.

    ``score_trial`` yields a coarse band (many trials collapse onto one value, e.g. all-Met =
    1.0) that a tie-aware nDCG can't credit, so the continuous reranker probability is folded
    into ``Score`` as a within-band refinement (normalized to [0, 1], scaled by
    ``_BLEND_HEADROOM`` to stay inside the gap to the next band). Bands never cross; ``Score``
    stays continuous while ``EligibilityScore`` keeps the raw band.
    """
    first_level_scores = first_level_scores or {}
    second_level_scores = second_level_scores or {}

    entries = [
        (
            t.get("TrialID", "Unknown"),
            score_trial(t),
            float(second_level_scores.get(t.get("TrialID", "Unknown"), 0.0)),
            float(first_level_scores.get(t.get("TrialID", "Unknown"), 0.0)),
        )
        for t in trial_data
    ]
    reranker_vals = [rer for _, _, rer, _ in entries]
    lo, hi = (min(reranker_vals), max(reranker_vals)) if reranker_vals else (0.0, 0.0)
    span = (hi - lo) or 1.0
    bands = sorted({band for _, band, _, _ in entries})
    min_gap = min((b - a for a, b in zip(bands, bands[1:])), default=1.0)
    delta = min_gap * _BLEND_HEADROOM

    ranked_trials = []
    for trial_id, band, reranker, first_level in entries:
        norm_reranker = (reranker - lo) / span  # [0, 1] continuous second-level signal
        ranked_trials.append(
            {
                "TrialID": trial_id,
                "Score": round(band + norm_reranker * delta, 10),
                "EligibilityScore": band,
                "RerankerScore": reranker,
                "FirstLevelScore": first_level,
            }
        )
    # Blended Score already encodes eligibility-then-reranker; the rest only settle a
    # rare exact-Score tie (ascending NCT id last).
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
    # Do NOT swallow write failures: the caller treats a clean return as a completed
    # patient, so a failed final write must surface to retry it (not mark it done).
    write_json_file({"RankedTrials": ranked_trials}, output_file)
    logger.info(f"Ranked trials saved to {output_file}")


def rerank_patient_dir(patient_dir: str) -> int:
    """Re-rank one patient's cached chain-of-thought outputs with the current scoring,
    overwriting ``ranked_trials.json`` in place. No model inference — reuses the stored per-trial
    ``NCT*.json`` outputs plus the reranker/first-level scores in ``ranked_trials.json``, so a
    ranking-logic change re-applies to a finished run without re-matching. Returns the number of
    trials ranked (0 = skipped).
    """
    ranked_path = os.path.join(patient_dir, "ranked_trials.json")
    if not os.path.exists(ranked_path):
        return 0
    existing = (read_json_file(ranked_path) or {}).get("RankedTrials", [])
    shortlist_ids = {r["TrialID"] for r in existing if isinstance(r, dict) and r.get("TrialID")}
    if not shortlist_ids:
        return 0
    first_level = {
        r["TrialID"]: float(r.get("FirstLevelScore", 0.0))
        for r in existing
        if isinstance(r, dict) and r.get("TrialID")
    }
    second_level = {
        r["TrialID"]: float(r.get("RerankerScore", 0.0))
        for r in existing
        if isinstance(r, dict) and r.get("TrialID")
    }
    trial_data = load_trial_data(patient_dir, allowed_ids=shortlist_ids)
    if not trial_data:
        # No per-trial CoT outputs to re-score (e.g. files absent); leave the run untouched.
        return 0
    ranked = rank_trials(
        trial_data, first_level_scores=first_level, second_level_scores=second_level
    )
    save_ranked_trials(ranked, ranked_path)
    return len(ranked)
