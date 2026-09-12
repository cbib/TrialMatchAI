"""Shared assessment controls and result provenance for matching and reporting."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path


def assessment_enabled(config: Mapping) -> bool:
    """Eligibility assessment is default-on, independently of its prompt style."""
    return bool(config.get("rag", {}).get("enabled", True))


def assessment_settings(config: Mapping) -> dict:
    return {
        "enabled": assessment_enabled(config),
        "use_cot_reasoning": bool(config.get("use_cot_reasoning", True)),
        "backend": config.get("rag", {}).get("backend", "vllm"),
        "no_think": bool(config.get("rag", {}).get("no_think", False)),
        "max_trials_rag": config.get("rag", {}).get("max_trials_rag", 20),
    }


def has_assessment_output(value: object) -> bool:
    """Recognize an assessment payload, without claiming clinical completeness."""
    if not isinstance(value, Mapping) or "error" in value:
        return False
    decision = value.get("Final Decision")
    if isinstance(decision, str) and decision.strip():
        return True
    return any(
        isinstance(items := value.get(key), list)
        and any(isinstance(item, Mapping) and item.get("Classification") for item in items)
        for key in ("Inclusion_Criteria_Evaluation", "Exclusion_Criteria_Evaluation")
    )


def _matching_run(path: str | Path, config: Mapping) -> dict:
    try:
        result = json.loads(Path(path).read_text(encoding="utf-8"))
        run = result.get("Run", {})
        if (
            isinstance(result.get("RankedTrials"), list)
            and isinstance(run, dict)
            and run.get("schema_version") == 1
            and run.get("assessment") == assessment_settings(config)
        ):
            return run
    except (OSError, ValueError, AttributeError):
        pass
    return {}


def match_controls_current(path: str | Path, config: Mapping) -> bool:
    """Compare assessment controls, independently of completion or retryability.

    This binds only assessment controls, not all patient/corpus/model inputs.
    """
    return bool(_matching_run(path, config))


def reusable_assessment_ids(path: str | Path, config: Mapping) -> set[str]:
    """Only outputs explicitly associated with a compatible run may be reused."""
    run = _matching_run(path, config)
    ids = run.get("assessed_trial_ids")
    if run.get("mode") != "eligibility_assessment" or not isinstance(ids, list):
        return set()
    available = set()
    for trial_id in ids:
        # Per-trial outputs are NCT files, never paths supplied by result metadata.
        if not isinstance(trial_id, str) or not trial_id.isalnum() or not trial_id.upper().startswith("NCT"):
            continue
        try:
            data = json.loads((Path(path).parent / f"{trial_id}.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if has_assessment_output(data):
            available.add(trial_id)
    return available


def match_is_complete(path: str | Path, config: Mapping) -> bool:
    """Incomplete/failed assessments stay pending; intentional bypasses are final."""
    run = _matching_run(path, config)
    if not run:
        return False
    status = run.get("assessment_status")
    if not assessment_enabled(config):
        return status == "disabled"
    if status == "no_candidates":
        return run.get("candidate_count") == 0 and run.get("assessed_trial_ids") == []
    if status != "outputs_available":
        return False
    available = reusable_assessment_ids(path, config)
    return bool(available) and len(available) == run.get("candidate_count")


def assessment_run_info(config: Mapping, trial_data: list[dict], candidate_ids: set[str]) -> dict:
    settings = assessment_settings(config)
    assessed_ids = sorted({
        trial["TrialID"] for trial in trial_data
        if settings["enabled"] and trial.get("TrialID") in candidate_ids and has_assessment_output(trial)
    })
    if not settings["enabled"]:
        status = "disabled"
    elif not candidate_ids:
        status = "no_candidates"
    elif not assessed_ids:
        status = "unavailable"
    elif len(assessed_ids) < len(candidate_ids):
        status = "partial"
    else:
        status = "outputs_available"
    return {
        "schema_version": 1,
        "assessment": settings,
        "mode": "eligibility_assessment" if assessed_ids else "retrieval_only",
        "assessment_status": status,
        "assessed_trial_ids": assessed_ids,
        "candidate_count": len(candidate_ids),
    }
