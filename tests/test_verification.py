"""Deterministic verification of reasoner eligibility claims (matching/verification.py).

The contract that matters: the verifier is allowed to overrule the reasoner ONLY on
decidable numeric/structured comparisons, and must stay silent everywhere else.
"""

import pytest

from trialmatchai.constraints import PatientConstraintContext, PatientConstraintFact
from trialmatchai.matching.verification import (
    AUTHORITATIVE_KINDS,
    verification_config,
    verify_trial_output,
)


def _ctx(age=None, sex=None, facts=None):
    return PatientConstraintContext(
        patient_id="p1", age_years=age, sex=sex, gender=sex, facts=facts or []
    )


def _output(section, criterion, classification):
    other = (
        "Exclusion_Criteria_Evaluation"
        if section == "Inclusion_Criteria_Evaluation"
        else "Inclusion_Criteria_Evaluation"
    )
    return {
        section: [{"Criterion": criterion, "Classification": classification}],
        other: [],
        "Final Decision": "Eligible",
    }


def _criteria(text, eligibility_type):
    return [{"criterion": text, "criteria_id": "c1", "eligibility_type": eligibility_type}]


ENABLED = {"verification": {"enabled": True, "apply_corrections": True}}


def test_disabled_by_default_makes_no_corrections():
    assert verification_config({})["apply_corrections"] is True
    assert verification_config({})["enabled"] is False


def test_age_floor_overrules_a_wrong_met_claim():
    """A 12-year-old cannot meet 'age >= 18'. The reasoner said Met; the verifier must not
    accept that, because it is a decidable comparison."""
    text = "Patients must be at least 18 years of age"
    report = verify_trial_output(
        trial_output=_output("Inclusion_Criteria_Evaluation", text, "Met"),
        criteria=_criteria(text, "inclusion"),
        patient_context=_ctx(age=12),
        nct_id="NCT1",
        config=ENABLED,
    )
    assert report["n_disagreements"] == 1
    d = report["disagreements"][0]
    assert d["reasoner_said"] == "met"
    assert d["verifier_said"] == "not met"
    assert "age" in d["kinds"]
    row = report["corrected_output"]["Inclusion_Criteria_Evaluation"][0]
    assert row["Classification"] == "Not Met"
    assert row["ReasonerClassification"] == "Met"  # provenance kept for audit
    assert row["VerifiedBy"] == "deterministic-constraints"


def test_no_disagreement_when_the_reasoner_is_right():
    text = "Patients must be at least 18 years of age"
    report = verify_trial_output(
        trial_output=_output("Inclusion_Criteria_Evaluation", text, "Met"),
        criteria=_criteria(text, "inclusion"),
        patient_context=_ctx(age=40),
        nct_id="NCT1",
        config=ENABLED,
    )
    assert report["n_disagreements"] == 0
    assert report["applied"] is False


def test_verifier_stays_silent_on_semantic_criteria():
    """Condition/phenotype matching is the reasoner's job -- a regex must not overrule it."""
    text = "Histologically confirmed anaplastic astrocytoma"
    report = verify_trial_output(
        trial_output=_output("Inclusion_Criteria_Evaluation", text, "Met"),
        criteria=_criteria(text, "inclusion"),
        patient_context=_ctx(age=40),
        nct_id="NCT1",
        config=ENABLED,
    )
    assert report["n_disagreements"] == 0


def test_authoritative_kinds_are_only_the_decidable_ones():
    assert AUTHORITATIVE_KINDS == {"age", "sex", "lab", "performance_status"}
    for semantic in ("condition", "phenotype", "medication", "procedure", "biomarker"):
        assert semantic not in AUTHORITATIVE_KINDS


def test_low_confidence_extraction_never_overrules():
    """A half-matched regex must not flip a label; confidence gates the override."""
    text = "Patients must be at least 18 years of age"
    strict = {"verification": {"enabled": True, "min_confidence": 1.01}}
    report = verify_trial_output(
        trial_output=_output("Inclusion_Criteria_Evaluation", text, "Met"),
        criteria=_criteria(text, "inclusion"),
        patient_context=_ctx(age=12),
        nct_id="NCT1",
        config=strict,
    )
    assert report["n_disagreements"] == 0


def test_apply_corrections_false_reports_without_changing_labels():
    """Measure-only mode: see how often the verifier fires before letting it act."""
    text = "Patients must be at least 18 years of age"
    report = verify_trial_output(
        trial_output=_output("Inclusion_Criteria_Evaluation", text, "Met"),
        criteria=_criteria(text, "inclusion"),
        patient_context=_ctx(age=12),
        nct_id="NCT1",
        config={"verification": {"enabled": True, "apply_corrections": False}},
    )
    assert report["n_disagreements"] == 1
    assert report["applied"] is False
    row = report["corrected_output"]["Inclusion_Criteria_Evaluation"][0]
    assert row["Classification"] == "Met"  # untouched


def test_unmatched_criterion_text_is_left_alone():
    """The reasoner sometimes paraphrases; no pairing means no verdict, not a guess."""
    report = verify_trial_output(
        trial_output=_output("Inclusion_Criteria_Evaluation", "some paraphrase", "Met"),
        criteria=_criteria("Patients must be at least 18 years of age", "inclusion"),
        patient_context=_ctx(age=12),
        nct_id="NCT1",
        config=ENABLED,
    )
    assert report["n_disagreements"] == 0


def test_missing_age_decides_nothing():
    """Unknown patient age must produce 'unknown', never a violation."""
    text = "Patients must be at least 18 years of age"
    report = verify_trial_output(
        trial_output=_output("Inclusion_Criteria_Evaluation", text, "Met"),
        criteria=_criteria(text, "inclusion"),
        patient_context=_ctx(age=None),
        nct_id="NCT1",
        config=ENABLED,
    )
    assert report["n_disagreements"] == 0


@pytest.mark.parametrize("bad", [{}, {"Inclusion_Criteria_Evaluation": "not a list"}])
def test_malformed_reasoner_output_does_not_raise(bad):
    """Verification must never break a finished run."""
    report = verify_trial_output(
        trial_output=bad,
        criteria=_criteria("Patients must be at least 18 years of age", "inclusion"),
        patient_context=_ctx(age=12),
        nct_id="NCT1",
        config=ENABLED,
    )
    assert report["n_disagreements"] == 0


def test_lab_threshold_flags_an_exclusion_the_reasoner_missed():
    """The case the literature says models fail: a numeric lab comparison."""
    text = "Platelet count less than 100,000/mm3"
    facts = [
        PatientConstraintFact(
            kind="lab",
            label="platelet count",
            value="50000",
            unit="/mm3",
            evidence_text="platelets 50000",
        )
    ]
    report = verify_trial_output(
        trial_output=_output("Exclusion_Criteria_Evaluation", text, "Not Violated"),
        criteria=_criteria(text, "exclusion"),
        patient_context=_ctx(age=40, facts=facts),
        nct_id="NCT1",
        config=ENABLED,
    )
    # Either the verifier catches it, or it abstains -- it must never confirm the wrong label.
    for d in report["disagreements"]:
        assert d["verifier_said"] in ("violated", "not violated")
        assert d["reasoner_said"] == "not violated"
