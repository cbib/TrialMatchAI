from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

from trialmatchai.constraints.models import CriterionConstraintEvaluation
from trialmatchai.utils.file_utils import write_json_file, write_text_file


def write_constraint_reports(
    *,
    output_folder: str | Path,
    evaluations: Iterable[CriterionConstraintEvaluation],
    top_trials: list[dict],
) -> None:
    folder = Path(output_folder)
    payload = [evaluation.model_dump(mode="json") for evaluation in evaluations]
    explained_trials = _build_explained_trials(payload, top_trials)
    write_json_file(
        {"criteria": payload, "top_trials": top_trials},
        str(folder / "constraint_evaluations.json"),
    )
    write_text_file(
        _render_markdown(payload, top_trials),
        str(folder / "constraint_summary.md"),
    )
    write_json_file(
        {"top_trials": explained_trials},
        str(folder / "top_trials_explained.json"),
    )


def _render_markdown(criteria: list[dict], top_trials: list[dict]) -> list[str]:
    lines = ["# Constraint-Aware Retrieval Summary", ""]
    if not criteria:
        lines.extend(
            [
                "No constraint evaluations were produced for this patient.",
                "",
            ]
        )
        return lines

    scores = {trial.get("nct_id"): trial.get("score") for trial in top_trials}
    by_trial: dict[str, list[dict]] = defaultdict(list)
    for item in criteria:
        by_trial[str(item.get("nct_id", "unknown"))].append(item)

    ordered_ids = [str(trial.get("nct_id")) for trial in top_trials if trial.get("nct_id")]
    ordered_ids.extend(sorted(set(by_trial) - set(ordered_ids)))
    for nct_id in ordered_ids:
        trial_items = by_trial.get(nct_id, [])
        lines.append(f"## {nct_id}")
        score = scores.get(nct_id)
        if isinstance(score, (int, float)):
            lines.append(f"- Retrieval score: {score:.4f}")
        shift = _ranking_shift(trial_items)
        if shift:
            lines.append(f"- Constraint ranking effect: {shift}")
        for status, title in (
            ("matched", "Matched"),
            ("violated", "Violated"),
            ("unknown", "Unknown"),
        ):
            entries = _entries_for_status(trial_items, status)
            if entries:
                lines.append(f"- {title}:")
                lines.extend(f"  - {entry}" for entry in entries[:8])
        lines.append("")
    return lines


def _build_explained_trials(criteria: list[dict], top_trials: list[dict]) -> list[dict]:
    by_trial: dict[str, list[dict]] = defaultdict(list)
    for item in criteria:
        by_trial[str(item.get("nct_id", "unknown"))].append(item)

    explained: list[dict] = []
    for trial in top_trials:
        nct_id = str(trial.get("nct_id", ""))
        trial_items = by_trial.get(nct_id, [])
        explained.append(
            {
                **trial,
                "constraint_effect": _ranking_shift(trial_items) or "unchanged",
                "matched_constraints": _entries_for_status(trial_items, "matched"),
                "violated_constraints": _entries_for_status(trial_items, "violated"),
                "unknown_constraints": _entries_for_status(trial_items, "unknown"),
            }
        )
    return explained


def _ranking_shift(trial_items: list[dict]) -> str | None:
    if not trial_items:
        return None
    signal = sum(float(item.get("constraint_signal") or 0.0) for item in trial_items)
    if signal > 0.05:
        return "boosted"
    if signal < -0.05:
        return "penalized"
    return "unchanged"


def _entries_for_status(trial_items: list[dict], status: str) -> list[str]:
    entries: list[str] = []
    for item in trial_items:
        criterion = item.get("criterion", "")
        for evaluation in item.get("evaluations", []):
            if evaluation.get("status") != status:
                continue
            constraint = evaluation.get("constraint", {})
            label = constraint.get("label") or constraint.get("kind")
            reason = evaluation.get("reason") or ""
            patient = evaluation.get("patient_evidence") or "no patient evidence"
            trial = constraint.get("evidence_text") or criterion
            entries.append(f"{label}: {reason} Patient: {patient}. Trial: {trial}.")
    return entries
