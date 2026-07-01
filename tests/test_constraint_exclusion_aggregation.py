"""Exclusion-polarity constraint aggregation must be worst-case, not mean-averaged.

A compound exclusion criterion where the patient trips one excluded item but lacks
several others must stay a disqualifier: mean-averaging the -1.0 violation with the
+0.25 not-violated credits could dilute it, or with enough not-violated items even
sign-flip it into a positive (trial-boosting) signal.
"""

from trialmatchai.constraints.evaluation import evaluate_constraint_set
from trialmatchai.constraints.models import (
    Constraint,
    ConstraintSet,
    PatientConstraintContext,
    PatientConstraintFact,
)

_DRUGS = ["drugA", "drugB", "drugC", "drugD", "drugE", "drugF", "drugG"]


def _exclusion_set() -> ConstraintSet:
    return ConstraintSet(
        nct_id="N1",
        criteria_id="C1",
        polarity="exclusion",
        source_text="Excluded if taking any of these drugs.",
        constraints=[Constraint(kind="medication", label=d) for d in _DRUGS],
    )


def test_single_exclusion_violation_dominates_not_averaged():
    # Patient is on ONE excluded drug (a real disqualifier) but lacks the other six.
    # signals = [-1.0, +0.25 x6]; old mean = (-1.0 + 1.5)/7 = +0.07 (a boosting sign-flip).
    context = PatientConstraintContext(
        patient_id="P1",
        facts=[PatientConstraintFact(kind="medication", label="drugA", evidence_text="on drugA")],
    )
    evaluation = evaluate_constraint_set(_exclusion_set(), context)
    assert evaluation.constraint_signal == -1.0  # worst-case: the violation dominates


def test_exclusion_with_no_violation_keeps_not_violated_credit():
    # Patient on none of the excluded drugs -> all "not violated"; the small +0.25 credit
    # (which the legacy ranker also awarded) is preserved, not zeroed by the fix.
    context = PatientConstraintContext(patient_id="P2", facts=[])
    evaluation = evaluate_constraint_set(_exclusion_set(), context)
    assert evaluation.constraint_signal == 0.25
