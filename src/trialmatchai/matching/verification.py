"""Deterministic verification of the reasoner's per-criterion eligibility claims.

The eligibility stage reads criterion text and emits Met / Not Met / Unclear / Irrelevant
(inclusion) or Violated / Not Violated / ... (exclusion). It does that in free text, which
is exactly where language models are weakest: comparing a patient's age or a lab value
against a numeric threshold. On MedCalc-Bench, GPT-4 scores ~51% on medical calculation
with chain-of-thought, and giving it the computation instead of asking it to reason raises
that to 81-85%. AgentMD reports 87.7% against 40.9% for CoT on the same shape of task.

TrialMatchAI already has a deterministic constraint engine (``constraints/``), but it is
only consulted during second-level retrieval scoring, blended in at weight 0.25. The
reasoner never sees it. This module closes that gap: it re-derives the constraints for each
criterion, evaluates them against the patient, and compares the verdict with what the
reasoner claimed.

**The split matters.** Deterministic verdicts are authoritative ONLY for constraint kinds
that are genuinely decidable -- age, sex, labs, performance status. Semantic kinds
(condition, phenotype, medication, procedure, biomarker identity) stay with the reasoner,
which is better at them than a regex. This is the neuro-symbolic division that alphaNeSy-CTM
measured on TREC CT 2021-2023: adding a symbolic verifier moved specificity -- the ability
to correctly REJECT a trial -- from 24.7% to 75.7%, while the agentic loop around it added
only 1.4 accuracy points. The verifier is where the value is.

Disabled by default (``verification.enabled``). Nothing here calls a model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from trialmatchai.constraints import (
    CriterionConstraintEvaluation,
    PatientConstraintContext,
    evaluate_constraint_set,
    extract_constraint_set,
)
from trialmatchai.matching.trial_ranker import _normalize_classification
from trialmatchai.utils.logging_config import setup_logging

logger = setup_logging(__name__)

# Constraint kinds whose deterministic verdict outranks the reasoner. These are decidable
# comparisons against a structured patient value: "age >= 18", "ANC > 1500", "ECOG 0-2".
# Everything else -- condition, phenotype, medication, procedure, biomarker, temporal -- is
# semantic or needs clinical judgement, and the reasoner keeps the final word there.
AUTHORITATIVE_KINDS = frozenset({"age", "sex", "lab", "performance_status"})

# Below this the extractor is guessing; a regex that half-matched must not overturn the model.
MIN_CONFIDENCE = 0.75

_INCLUSION_MET = "met"
_INCLUSION_NOT_MET = "not met"
_EXCLUSION_VIOLATED = "violated"
_EXCLUSION_NOT_VIOLATED = "not violated"


def verification_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    raw = (config or {}).get("verification") or {}
    if not isinstance(raw, Mapping):
        raw = {}
    return {
        "enabled": bool(raw.get("enabled", False)),
        "min_confidence": float(raw.get("min_confidence", MIN_CONFIDENCE)),
        "authoritative_kinds": frozenset(
            raw.get("authoritative_kinds") or AUTHORITATIVE_KINDS
        ),
        # When false, disagreements are recorded but the classification is left alone --
        # useful for measuring how often the verifier fires before letting it act.
        "apply_corrections": bool(raw.get("apply_corrections", True)),
    }


def _decisive_evaluations(
    evaluation: CriterionConstraintEvaluation,
    *,
    authoritative_kinds: frozenset[str],
    min_confidence: float,
) -> list[Any]:
    """Constraint evaluations that are allowed to overrule the reasoner."""
    decisive = []
    for item in evaluation.evaluations:
        constraint = item.constraint
        if constraint.kind not in authoritative_kinds:
            continue
        if constraint.confidence < min_confidence:
            continue
        if item.status not in ("matched", "violated"):
            continue  # unknown / not_applicable decide nothing
        decisive.append(item)
    return decisive


def _verdict_for(polarity: str, decisive: Sequence[Any]) -> str | None:
    """The label the deterministic engine implies, or None if it implies nothing.

    A violated constraint dominates: for an inclusion criterion the patient fails to meet
    it, and for an exclusion criterion the patient is excluded by it. Requiring every
    decisive constraint to match before claiming the positive label keeps the verifier
    conservative -- it is far more willing to say "this does not hold" than "this holds".
    """
    if not decisive:
        return None
    any_violated = any(item.status == "violated" for item in decisive)
    all_matched = all(item.status == "matched" for item in decisive)
    if polarity == "exclusion":
        if any_violated:
            return _EXCLUSION_VIOLATED
        return _EXCLUSION_NOT_VIOLATED if all_matched else None
    if any_violated:
        return _INCLUSION_NOT_MET
    return _INCLUSION_MET if all_matched else None


def verify_trial_output(
    *,
    trial_output: Mapping[str, Any],
    criteria: Sequence[Mapping[str, Any]],
    patient_context: PatientConstraintContext,
    nct_id: str,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Check one trial's reasoner output against the deterministic constraint engine.

    ``criteria`` supplies the criterion text and inclusion/exclusion type, keyed so each
    reasoner claim can be paired with the criterion it judged. Returns a report and, when
    corrections are enabled, a corrected copy of the trial output.
    """
    cfg = verification_config(config)
    by_text = {
        str(c.get("criterion") or c.get("text") or "").strip(): c
        for c in criteria
        if (c.get("criterion") or c.get("text"))
    }

    disagreements: list[dict[str, Any]] = []
    corrected = {k: (list(v) if isinstance(v, list) else v) for k, v in trial_output.items()}

    for section, polarity in (
        ("Inclusion_Criteria_Evaluation", "inclusion"),
        ("Exclusion_Criteria_Evaluation", "exclusion"),
    ):
        rows = corrected.get(section)
        if not isinstance(rows, list):
            continue
        new_rows = []
        for row in rows:
            if not isinstance(row, dict):
                new_rows.append(row)
                continue
            row = dict(row)
            text = str(row.get("Criterion") or "").strip()
            claimed = _normalize_classification(row.get("Classification"))
            source = by_text.get(text)
            if not text or not source:
                new_rows.append(row)
                continue
            try:
                constraint_set = extract_constraint_set(
                    nct_id=nct_id,
                    criteria_id=str(source.get("criteria_id") or text[:64]),
                    criterion=text,
                    eligibility_type=str(source.get("eligibility_type") or polarity),
                    entities=source.get("entities"),
                )
                evaluation = evaluate_constraint_set(constraint_set, patient_context)
            except Exception as exc:  # never let verification break a finished run
                logger.warning("Verification skipped for %s criterion %r: %s", nct_id, text[:60], exc)
                new_rows.append(row)
                continue

            decisive = _decisive_evaluations(
                evaluation,
                authoritative_kinds=cfg["authoritative_kinds"],
                min_confidence=cfg["min_confidence"],
            )
            verdict = _verdict_for(polarity, decisive)
            if verdict is None or verdict == claimed:
                new_rows.append(row)
                continue

            disagreements.append(
                {
                    "nct_id": nct_id,
                    "polarity": polarity,
                    "criterion": text,
                    "reasoner_said": claimed,
                    "verifier_said": verdict,
                    "kinds": sorted({d.constraint.kind for d in decisive}),
                    "reasons": [d.reason for d in decisive if d.reason][:3],
                }
            )
            if cfg["apply_corrections"]:
                # Record what the reasoner said BEFORE overwriting it, so a corrected run
                # stays auditable: score_trial only reads Classification.
                row["ReasonerClassification"] = row.get("Classification")
                row["VerifiedBy"] = "deterministic-constraints"
                row["Classification"] = verdict.title()
            new_rows.append(row)
        corrected[section] = new_rows

    return {
        "nct_id": nct_id,
        "disagreements": disagreements,
        "n_disagreements": len(disagreements),
        "corrected_output": corrected if cfg["apply_corrections"] else trial_output,
        "applied": bool(cfg["apply_corrections"] and disagreements),
    }
