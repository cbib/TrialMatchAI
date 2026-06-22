import json

from trialmatchai.constraints import (
    build_patient_constraint_context,
    evaluate_constraint_set,
    extract_constraint_set,
    write_constraint_reports,
)
from trialmatchai.constraints.models import Constraint, ConstraintSet
from trialmatchai.interop.models import ClinicalFact, Demographics, PatientProfile, Provenance


def test_extracts_common_deterministic_constraints():
    parsed = extract_constraint_set(
        nct_id="N1",
        criteria_id="C1",
        criterion=(
            "Adults aged 18-75 years with ANC >= 1500/mm3, ECOG 0-1, "
            "EGFR mutated disease, and treatment within 6 months."
        ),
        eligibility_type="Inclusion Criteria",
    )

    kinds = {constraint.kind for constraint in parsed.constraints}
    assert {"age", "lab", "performance_status", "biomarker", "temporal"} <= kinds
    assert any(
        constraint.kind == "age"
        and constraint.comparator == "between"
        and constraint.min_value == 18
        and constraint.max_value == 75
        for constraint in parsed.constraints
    )
    assert any(
        constraint.kind == "lab"
        and constraint.label == "absolute neutrophil count"
        and constraint.comparator == "ge"
        and constraint.value == 1500
        for constraint in parsed.constraints
    )


def test_extracts_age_lower_bound_and_sex():
    parsed = extract_constraint_set(
        nct_id="N1",
        criteria_id="C2",
        criterion="Female participants 18 years or older.",
        eligibility_type="Inclusion Criteria",
    )

    assert any(
        constraint.kind == "age"
        and constraint.comparator == "ge"
        and constraint.value == 18
        for constraint in parsed.constraints
    )
    assert any(
        constraint.kind == "sex" and constraint.value == "female"
        for constraint in parsed.constraints
    )


def test_patient_context_evaluates_matches_and_exclusion_violations():
    profile = PatientProfile(
        patient_id="P1",
        demographics=Demographics(age_years=64, sex="female"),
        conditions=[
            _fact(
                "condition-1",
                "condition",
                "non-small cell lung cancer",
                evidence_text="Patient has metastatic non-small cell lung cancer.",
            )
        ],
        observations=[
            _fact(
                "obs-1",
                "observation",
                "absolute neutrophil count",
                description="1800 /mm3",
                evidence_text="ANC 1800/mm3.",
            )
        ],
        genomic_findings=[
            _fact(
                "gene-1",
                "genomic_finding",
                "EGFR mutated",
                evidence_text="EGFR exon 19 deletion detected.",
            )
        ],
        medications=[
            _fact(
                "med-1",
                "medication",
                "osimertinib",
                evidence_text="Prior osimertinib documented.",
                temporality="prior",
            )
        ],
    )
    context = build_patient_constraint_context(profile)
    inclusion = ConstraintSet(
        nct_id="N1",
        criteria_id="C1",
        polarity="inclusion",
        source_text="Adults with NSCLC, EGFR mutation, and ANC >= 1500/mm3.",
        constraints=[
            Constraint(
                kind="age",
                label="age",
                comparator="between",
                min_value=18,
                max_value=75,
            ),
            Constraint(kind="condition", label="non-small cell lung cancer"),
            Constraint(kind="biomarker", label="EGFR", comparator="mutated"),
            Constraint(
                kind="lab",
                label="absolute neutrophil count",
                comparator="ge",
                value=1500,
                unit="/mm3",
            ),
        ],
    )
    exclusion = ConstraintSet(
        nct_id="N1",
        criteria_id="C2",
        polarity="exclusion",
        source_text="Prior osimertinib is excluded.",
        constraints=[
            Constraint(kind="medication", label="osimertinib", comparator="prior")
        ],
    )

    inclusion_eval = evaluate_constraint_set(inclusion, context)
    exclusion_eval = evaluate_constraint_set(exclusion, context)

    assert inclusion_eval.violated_count == 0
    assert inclusion_eval.matched_count == 4
    assert inclusion_eval.constraint_signal > 0
    assert exclusion_eval.violated_count == 1
    assert exclusion_eval.constraint_signal < 0


def test_unknown_constraints_are_neutral():
    context = build_patient_constraint_context(
        PatientProfile(patient_id="P1", demographics=Demographics(age_years=42))
    )
    parsed = ConstraintSet(
        nct_id="N1",
        criteria_id="C1",
        polarity="inclusion",
        source_text="Documented diabetes mellitus.",
        constraints=[Constraint(kind="condition", label="diabetes mellitus")],
    )

    evaluation = evaluate_constraint_set(parsed, context)

    assert evaluation.unknown_count == 1
    assert evaluation.constraint_signal == 0


def test_constraint_reports_are_written(tmp_path):
    context = build_patient_constraint_context(
        PatientProfile(
            patient_id="P1",
            conditions=[
                _fact(
                    "condition-1",
                    "condition",
                    "lung cancer",
                    evidence_text="Patient has lung cancer.",
                )
            ],
        )
    )
    evaluation = evaluate_constraint_set(
        ConstraintSet(
            nct_id="N1",
            criteria_id="C1",
            polarity="inclusion",
            source_text="Patients with lung cancer.",
            constraints=[Constraint(kind="condition", label="lung cancer")],
        ),
        context,
    )

    write_constraint_reports(
        output_folder=tmp_path,
        evaluations=[evaluation],
        top_trials=[{"nct_id": "N1", "score": 0.91}],
    )

    payload = json.loads((tmp_path / "constraint_evaluations.json").read_text())
    explained = json.loads((tmp_path / "top_trials_explained.json").read_text())
    summary = (tmp_path / "constraint_summary.md").read_text()
    assert payload["criteria"][0]["nct_id"] == "N1"
    assert explained["top_trials"][0]["constraint_effect"] == "boosted"
    assert "Matched" in summary


def test_exclusion_clean_pass_is_rewarded():
    # Patient does NOT have the excluded item -> not excluded -> should reward.
    context = build_patient_constraint_context(
        PatientProfile(patient_id="P1", demographics=Demographics(age_years=60))
    )
    exclusion = ConstraintSet(
        nct_id="N1",
        criteria_id="C1",
        polarity="exclusion",
        source_text="Prior chemotherapy is excluded.",
        constraints=[
            Constraint(kind="medication", label="chemotherapy", comparator="prior")
        ],
    )
    evaluation = evaluate_constraint_set(exclusion, context)
    assert evaluation.matched_count == 1
    assert evaluation.violated_count == 0
    assert evaluation.constraint_signal > 0


def test_dosing_and_unit_numbers_are_not_ages():
    for criterion in (
        "Administer 100 to 200 mg daily.",
        "Give at least 100 mg of the study drug.",
        "Up to 3 cycles of therapy.",
    ):
        parsed = extract_constraint_set(
            nct_id="N1",
            criteria_id="C1",
            criterion=criterion,
            eligibility_type="Inclusion Criteria",
        )
        assert not any(c.kind == "age" for c in parsed.constraints), criterion


def test_comparator_less_lab_is_skipped():
    parsed = extract_constraint_set(
        nct_id="N1",
        criteria_id="C1",
        criterion="Creatinine 1.5 at screening.",
        eligibility_type="Exclusion Criteria",
    )
    assert not any(c.kind == "lab" for c in parsed.constraints)
    # An explicit comparator is still extracted.
    parsed2 = extract_constraint_set(
        nct_id="N1",
        criteria_id="C2",
        criterion="Creatinine <= 1.5 mg/dL.",
        eligibility_type="Exclusion Criteria",
    )
    assert any(c.kind == "lab" and c.comparator == "le" for c in parsed2.constraints)


def test_biomarker_wildtype_patient_is_not_a_false_positive():
    context = build_patient_constraint_context(
        PatientProfile(
            patient_id="P1",
            genomic_findings=[
                _fact(
                    "g1",
                    "genomic_finding",
                    "EGFR wild-type",
                    evidence_text="EGFR wild-type, no mutation detected.",
                )
            ],
        )
    )
    # Exclusion "EGFR mutation" against a wild-type patient -> not excluded -> reward.
    exclusion = ConstraintSet(
        nct_id="N1",
        criteria_id="C1",
        polarity="exclusion",
        source_text="EGFR mutation excluded.",
        constraints=[Constraint(kind="biomarker", label="EGFR", comparator="mutated")],
    )
    evaluation = evaluate_constraint_set(exclusion, context)
    assert evaluation.violated_count == 0
    assert evaluation.constraint_signal > 0


def test_unknown_is_neutral_false_penalizes_unconfirmable_inclusion():
    context = build_patient_constraint_context(
        PatientProfile(patient_id="P1", demographics=Demographics(age_years=42))
    )
    inclusion = ConstraintSet(
        nct_id="N1",
        criteria_id="C1",
        polarity="inclusion",
        source_text="Documented diabetes mellitus.",
        constraints=[Constraint(kind="condition", label="diabetes mellitus")],
    )
    neutral = evaluate_constraint_set(inclusion, context, unknown_is_neutral=True)
    penalized = evaluate_constraint_set(inclusion, context, unknown_is_neutral=False)
    assert neutral.constraint_signal == 0
    assert penalized.constraint_signal < 0


def _fact(
    fact_id: str,
    category: str,
    label: str,
    *,
    description: str | None = None,
    evidence_text: str | None = None,
    temporality: str | None = None,
) -> ClinicalFact:
    return ClinicalFact(
        fact_id=fact_id,
        category=category,
        label=label,
        description=description,
        evidence_text=evidence_text,
        temporality=temporality,
        provenance=Provenance(source_format="test"),
    )
