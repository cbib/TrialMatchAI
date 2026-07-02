from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Any

from trialmatchai.constraints.models import (
    Constraint,
    ConstraintEvaluation,
    ConstraintSet,
    CriterionConstraintEvaluation,
    EvaluationStatus,
    PatientConstraintContext,
    PatientConstraintFact,
)
from trialmatchai.interop.models import ClinicalFact, PatientProfile
from trialmatchai.utils.text import flatten_text


MATCH_SIGNAL = 1.0
INCLUSION_VIOLATION_SIGNAL = -0.6
EXCLUSION_VIOLATION_SIGNAL = -1.0
EXCLUSION_NOT_VIOLATED_SIGNAL = 0.25
# Applied to an inclusion criterion we cannot confirm when unknown_is_neutral=False.
UNKNOWN_INCLUSION_PENALTY = -0.1


def build_patient_constraint_context(profile: PatientProfile) -> PatientConstraintContext:
    facts: list[PatientConstraintFact] = []
    facts.extend(_facts_from_clinical(profile.conditions, "condition"))
    facts.extend(_facts_from_clinical(profile.phenotypes, "phenotype"))
    facts.extend(_facts_from_clinical(profile.observations, "lab"))
    facts.extend(_facts_from_clinical(profile.medications, "medication"))
    facts.extend(_facts_from_clinical(profile.procedures, "procedure"))
    facts.extend(_facts_from_clinical(profile.genomic_findings, "biomarker"))
    facts.extend(_facts_from_clinical(profile.cancer_profile, "condition"))
    # Family history is a relative's disease, not the patient's own, and no constraint kind consumes it; omit.
    facts.extend(_performance_facts(profile))
    return PatientConstraintContext(
        patient_id=profile.patient_id,
        age_years=profile.demographics.age_years,
        sex=profile.demographics.sex,
        gender=profile.demographics.gender,
        facts=facts,
    )


def evaluate_constraint_set(
    constraint_set: ConstraintSet,
    patient_context: PatientConstraintContext,
    *,
    unknown_is_neutral: bool = True,
) -> CriterionConstraintEvaluation:
    evaluations = [
        evaluate_constraint(constraint, constraint_set, patient_context)
        for constraint in constraint_set.constraints
    ]
    signals = [evaluation.score_signal for evaluation in evaluations if evaluation.score_signal != 0]
    if signals:
        if constraint_set.polarity == "exclusion":
            # One exclusion hit disqualifies, so take the min: averaging could dilute or sign-flip a real violation.
            signal = _clamp(min(signals))
        else:
            # Inclusion sub-constraints must all hold, so take the min: averaging could dilute or sign-flip a real miss.
            signal = _clamp(min(signals))
    elif (
        not unknown_is_neutral
        and constraint_set.polarity == "inclusion"
        and evaluations
    ):
        # We could not confirm any inclusion constraint; apply a small penalty.
        signal = UNKNOWN_INCLUSION_PENALTY
    else:
        signal = 0.0
    return CriterionConstraintEvaluation(
        nct_id=constraint_set.nct_id,
        criteria_id=constraint_set.criteria_id,
        criterion=constraint_set.source_text,
        polarity=constraint_set.polarity,
        evaluations=evaluations,
        constraint_signal=signal,
        matched_count=sum(1 for evaluation in evaluations if evaluation.status == "matched"),
        violated_count=sum(1 for evaluation in evaluations if evaluation.status == "violated"),
        unknown_count=sum(1 for evaluation in evaluations if evaluation.status == "unknown"),
        not_applicable_count=sum(
            1 for evaluation in evaluations if evaluation.status == "not_applicable"
        ),
    )


def evaluate_constraint(
    constraint: Constraint,
    constraint_set: ConstraintSet,
    patient_context: PatientConstraintContext,
) -> ConstraintEvaluation:
    relation, evidence, reason = _evaluate_relation(constraint, patient_context)
    status, signal = _status_and_signal(constraint_set.polarity, relation)
    return ConstraintEvaluation(
        nct_id=constraint_set.nct_id,
        criteria_id=constraint_set.criteria_id,
        polarity=constraint_set.polarity,
        constraint=constraint,
        status=status,
        score_signal=signal,
        patient_evidence=evidence,
        reason=reason,
    )


def apply_constraint_score(
    base_score: float,
    signal: float,
    *,
    score_weight: float,
) -> float:
    return _clamp(base_score + score_weight * signal, lower=0.0, upper=1.0)


def _facts_from_clinical(
    facts: Sequence[ClinicalFact],
    kind: str,
) -> list[PatientConstraintFact]:
    output: list[PatientConstraintFact] = []
    for fact in facts:
        numeric_value, unit = _parse_numeric_value(fact.description)
        output.append(
            PatientConstraintFact(
                kind=kind,  # type: ignore[arg-type]
                label=fact.label,
                normalized_codes=[
                    code.model_dump(mode="json", exclude_none=True)
                    for code in fact.normalized_codes
                ],
                value=numeric_value,
                unit=unit,
                negated=fact.negated,
                temporality=fact.temporality,
                evidence_text=fact.evidence_text or _fact_text(fact),
                source=fact.provenance.model_dump(mode="json", exclude_none=True),
            )
        )
    return output


def _performance_facts(profile: PatientProfile) -> list[PatientConstraintFact]:
    output: list[PatientConstraintFact] = []
    buckets = [
        *profile.observations,
        *profile.cancer_profile,
        *profile.diagnostic_reports,
    ]
    for fact in buckets:
        text = _fact_text(fact)
        for label, pattern in (
            ("ECOG", r"\becog\b[^0-9]{0,24}([0-4])\b"),
            ("Karnofsky", r"\bkarnofsky\b[^0-9]*(\d{2,3})"),
        ):
            match = re.search(pattern, text, re.IGNORECASE)
            if not match:
                continue
            output.append(
                PatientConstraintFact(
                    kind="performance_status",
                    label=label,
                    value=float(match.group(1)),
                    negated=fact.negated,
                    temporality=fact.temporality,
                    evidence_text=fact.evidence_text or text,
                    source=fact.provenance.model_dump(mode="json", exclude_none=True),
                )
            )
    return output


def _evaluate_relation(
    constraint: Constraint,
    patient_context: PatientConstraintContext,
) -> tuple[str, str | None, str]:
    if constraint.kind == "age":
        return _evaluate_age(constraint, patient_context)
    if constraint.kind == "sex":
        return _evaluate_sex(constraint, patient_context)
    if constraint.kind == "lab":
        return _evaluate_numeric_fact(constraint, patient_context, ("lab",))
    if constraint.kind == "performance_status":
        return _evaluate_numeric_fact(
            constraint,
            patient_context,
            ("performance_status", "lab"),
        )
    if constraint.kind == "biomarker":
        return _evaluate_biomarker(constraint, patient_context)
    if constraint.kind == "temporal":
        # No dated patient timeline to check against; return unknown rather than a default "absent" that soft-matches.
        return "unknown", None, "Temporal criteria are not evaluated (no dated patient timeline)."
    return _evaluate_concept_fact(constraint, patient_context, (constraint.kind,))


def _evaluate_age(
    constraint: Constraint,
    patient_context: PatientConstraintContext,
) -> tuple[str, str | None, str]:
    age = patient_context.age_years
    if age is None:
        return "unknown", None, "Patient age is unavailable."
    satisfied = _compare_numeric(age, constraint)
    evidence = f"Patient age {age:g} years"
    if satisfied:
        return "satisfied", evidence, "Patient age satisfies the age constraint."
    return "unsatisfied", evidence, "Patient age does not satisfy the age constraint."


def _evaluate_sex(
    constraint: Constraint,
    patient_context: PatientConstraintContext,
) -> tuple[str, str | None, str]:
    requested = _normalize_text(str(constraint.value or constraint.label))
    current = _normalize_text(patient_context.sex or patient_context.gender or "")
    if not current:
        return "unknown", None, "Patient sex/gender is unavailable."
    if requested in {"all", "any", "both"} or current in {"all", "any", "both"}:
        return "satisfied", current, "Sex/gender constraint is broadly compatible."
    if requested == current:
        return "satisfied", current, "Patient sex/gender satisfies the constraint."
    return "unsatisfied", current, "Patient sex/gender does not satisfy the constraint."


def _units_incompatible(fact_unit: str | None, constraint_unit: str | None) -> bool:
    """True only when both sides declare a unit and they normalize differently (missing unit = compatible)."""
    if not fact_unit or not constraint_unit:
        return False
    normalize = lambda u: re.sub(r"[\s.]+", "", u).casefold()  # noqa: E731
    return normalize(fact_unit) != normalize(constraint_unit)


def _evaluate_numeric_fact(
    constraint: Constraint,
    patient_context: PatientConstraintContext,
    kinds: tuple[str, ...],
) -> tuple[str, str | None, str]:
    candidates = _matching_facts(patient_context.facts, constraint, kinds)
    if not candidates:
        return "unknown", None, f"No patient fact found for {constraint.label}."
    for fact in candidates:
        if fact.negated:
            return "unsatisfied", fact.evidence_text, f"Patient fact is explicitly absent: {fact.label}."
        if fact.value is None:
            continue
        # Abstain on incompatible units (ANC 1.5 x10^9/L vs a >=1500 /mm3 threshold would give a wrong verdict).
        if _units_incompatible(fact.unit, constraint.unit):
            return (
                "unknown",
                fact.evidence_text,
                f"Units differ ({fact.unit} vs {constraint.unit}); not compared for {constraint.label}.",
            )
        if _compare_numeric(fact.value, constraint):
            return "satisfied", fact.evidence_text, f"Patient value satisfies {constraint.label}."
        return "unsatisfied", fact.evidence_text, f"Patient value does not satisfy {constraint.label}."
    return "unknown", candidates[0].evidence_text, f"No numeric value available for {constraint.label}."


_NEGATIVE_BIOMARKER_TERMS = (
    "negative",
    "wild type",
    "wildtype",
    "wild-type",
    "not detected",
    "no mutation",
    "absent",
)


def _evaluate_biomarker(
    constraint: Constraint,
    patient_context: PatientConstraintContext,
) -> tuple[str, str | None, str]:
    candidates = _matching_facts(
        patient_context.facts,
        constraint,
        ("biomarker", "lab", "condition"),
    )
    if not candidates:
        return "absent", None, f"No patient biomarker fact found for {constraint.label}."

    fact = candidates[0]
    fact_text = _normalize_text(flatten_text([fact.label, fact.evidence_text]))
    patient_negative = fact.negated or any(
        term in fact_text for term in _NEGATIVE_BIOMARKER_TERMS
    )
    expected_negative = str(constraint.comparator) in {"negative", "wildtype"}

    if expected_negative:
        if patient_negative:
            return "satisfied", fact.evidence_text, "Patient biomarker is negative/wild-type as required."
        return "unsatisfied", fact.evidence_text, "Patient biomarker is not negative/wild-type."
    # expected present / positive / mutated
    if patient_negative:
        return (
            "unsatisfied",
            fact.evidence_text,
            "Patient biomarker is negative/wild-type; conflicts with the required presence.",
        )
    return "satisfied", fact.evidence_text, "Patient biomarker evidence satisfies the constraint."


def _evaluate_concept_fact(
    constraint: Constraint,
    patient_context: PatientConstraintContext,
    kinds: tuple[str, ...],
) -> tuple[str, str | None, str]:
    candidates = _matching_facts(patient_context.facts, constraint, kinds)
    requires_absence = constraint.comparator == "absent"

    # The patient "has" the item when a matching, non-negated fact exists.
    present_fact = next((f for f in candidates if not f.negated), None)
    evidence = candidates[0].evidence_text if candidates else None

    if requires_absence:
        if present_fact is not None:
            return "unsatisfied", present_fact.evidence_text, f"Patient has {constraint.label}, but absence is required."
        return "satisfied", evidence, f"Patient has no record of {constraint.label} (absence required)."

    if present_fact is not None:
        return "satisfied", present_fact.evidence_text, f"Patient fact matches {constraint.label}."
    # No matching fact, or only negated ones -> the patient lacks the item.
    return "absent", evidence, f"Patient has no record of {constraint.label}."


def _status_and_signal(polarity: str, relation: str) -> tuple[EvaluationStatus, float]:
    """Map a (polarity, relation) pair to a scoring status and signal.

    Relations: satisfied (meets/has), unsatisfied (contradicts a bound),
    absent (no record/negated), unknown (indeterminate).
    """
    if relation == "not_applicable":
        return "not_applicable", 0.0

    if polarity == "exclusion":
        if relation == "satisfied":
            # Patient has the excluded item -> excluded.
            return "violated", EXCLUSION_VIOLATION_SIGNAL
        if relation in {"unsatisfied", "absent"}:
            # Patient does not meet / lacks the excluded item -> not excluded.
            return "matched", EXCLUSION_NOT_VIOLATED_SIGNAL
        return "unknown", 0.0

    # inclusion / unknown polarity
    if relation == "satisfied":
        return "matched", MATCH_SIGNAL
    if relation == "unsatisfied":
        return "violated", INCLUSION_VIOLATION_SIGNAL
    # Absent inclusion: can't confirm required presence from an incomplete profile, so stay neutral.
    return "unknown", 0.0


def _matching_facts(
    facts: Iterable[PatientConstraintFact],
    constraint: Constraint,
    kinds: tuple[str, ...],
) -> list[PatientConstraintFact]:
    constraint_text = _normalize_text(constraint.label)
    constraint_codes = {
        f"{code.get('vocabulary')}:{code.get('code')}".casefold()
        for code in constraint.normalized_codes
        if code.get("vocabulary") and code.get("code")
    }
    matches: list[tuple[float, PatientConstraintFact]] = []
    for fact in facts:
        if fact.kind not in kinds:
            continue
        fact_codes = {
            f"{code.get('vocabulary')}:{code.get('code')}".casefold()
            for code in fact.normalized_codes
            if code.get("vocabulary") and code.get("code")
        }
        if constraint_codes and constraint_codes & fact_codes:
            matches.append((1.0, fact))
            continue
        fact_text = _normalize_text(flatten_text([fact.label, fact.evidence_text]))
        score = _text_match_score(constraint_text, fact_text)
        if score > 0:
            matches.append((score, fact))
    matches.sort(key=lambda item: item[0], reverse=True)
    return [fact for _, fact in matches]


def _compare_numeric(value: float, constraint: Constraint) -> bool:
    comparator = constraint.comparator
    target = _float_or_none(constraint.value)
    if comparator == "between":
        if constraint.min_value is not None and value < constraint.min_value:
            return False
        if constraint.max_value is not None and value > constraint.max_value:
            return False
        return True
    if target is None:
        return False
    if comparator == "gt":
        return value > target
    if comparator == "ge":
        return value >= target
    if comparator == "lt":
        return value < target
    if comparator == "le":
        return value <= target
    if comparator == "eq":
        return value == target
    if comparator == "ne":
        return value != target
    return False


def _parse_numeric_value(value: str | None) -> tuple[float | None, str | None]:
    if not value:
        return None, None
    # Accept thousands separators ("1,500 /mm3") so the value isn't truncated at the comma.
    match = re.search(r"(-?\d[\d,]*(?:\.\d+)?)\s*([A-Za-z/%0-9^µμ.-]+)?", value)
    if not match:
        return None, None
    unit = (match.group(2) or "").strip() or None
    return float(match.group(1).replace(",", "")), unit


def _fact_text(fact: ClinicalFact) -> str:
    return flatten_text([fact.label, fact.description, fact.evidence_text])


def _text_match_score(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    # Whole-word containment only: raw substring lets short markers collide ("alk" in "alkaline phosphatase").
    shorter, longer = (left, right) if len(left) <= len(right) else (right, left)
    if len(shorter) >= 3 and re.search(rf"\b{re.escape(shorter)}\b", longer):
        return 0.95
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    if overlap == 0:
        return 0.0
    coverage = overlap / len(left_tokens)
    jaccard = overlap / len(left_tokens | right_tokens)
    score = 0.75 * coverage + 0.25 * jaccard
    return score if score >= 0.5 else 0.0


def _normalize_text(value: str) -> str:
    value = value.casefold()
    value = value.replace("non small", "non-small")
    return " ".join(re.findall(r"[a-z0-9-]+", value))


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp(value: float, *, lower: float = -1.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))
