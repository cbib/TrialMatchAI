"""Regression tests for the hardened FHIR importer (real-EHR edge cases)."""

from __future__ import annotations

import json

from trialmatchai.interop.importers.fhir import (
    _medication_codes_label,
    _resource_disposition,
    _value_x,
    import_fhir,
)
from trialmatchai.interop.utils import (
    code_from_fhir_codeable,
    codes_from_fhir_codeable,
    parse_date,
)


# --------------------------------------------------------------------- unit level


def test_status_disposition():
    cases = [
        ("Condition", {"clinicalStatus": {"coding": [{"code": "resolved"}]}}, ("keep", True)),
        ("Condition", {"clinicalStatus": {"coding": [{"code": "inactive"}]}}, ("keep", True)),
        ("Condition", {"verificationStatus": {"coding": [{"code": "refuted"}]}}, ("keep", True)),
        ("Condition", {"verificationStatus": {"coding": [{"code": "entered-in-error"}]}}, ("drop", False)),
        ("Observation", {"status": "entered-in-error"}, ("drop", False)),
        ("Procedure", {"status": "not-done"}, ("drop", False)),
        # Completed/stopped medications are real prior exposure -> kept, not negated.
        ("MedicationStatement", {"status": "completed"}, ("keep", False)),
        ("MedicationStatement", {"status": "stopped"}, ("keep", False)),
        ("Condition", {"clinicalStatus": {"coding": [{"code": "active"}]}}, ("keep", False)),
    ]
    for rtype, resource, expected in cases:
        assert _resource_disposition(rtype, resource) == expected, (rtype, resource)


def test_partial_fhir_dates():
    assert parse_date("1980") is not None and parse_date("1980").year == 1980
    assert parse_date("1980-03").month == 3
    assert parse_date("1980-03-15").day == 15
    assert parse_date("1980-03-15T09:00:00Z").year == 1980
    assert parse_date("garbage") is None


def test_multi_coding_prefers_known_vocabulary_and_keeps_all():
    cc = {
        "coding": [
            {"system": "urn:oid:1.2.840.114350", "code": "EPIC1"},
            {"system": "http://snomed.info/sct", "code": "254637007", "display": "NSCLC"},
        ]
    }
    best = code_from_fhir_codeable(cc)
    assert best.vocabulary == "SNOMED" and best.code == "254637007"
    all_codes = codes_from_fhir_codeable(cc)
    assert {c.vocabulary for c in all_codes} == {"SNOMED", "urn:oid:1.2.840.114350"}


def test_value_x_variants_and_comparator():
    assert _value_x({"valueQuantity": {"value": 9.5, "unit": "g/dL"}}) == "9.5 g/dL"
    assert _value_x({"valueQuantity": {"comparator": "<", "value": 0.01, "unit": "ng/mL"}}) == "<0.01 ng/mL"
    assert _value_x({"valueString": "positive"}) == "positive"
    assert _value_x({"valueCodeableConcept": {"text": "detected"}}) == "detected"
    assert _value_x({"valueRange": {"low": {"value": 1}, "high": {"value": 5}}}) == "1-5"


def test_medication_reference_and_contained():
    # medicationReference with display only
    codes, label = _medication_codes_label(
        {"medicationReference": {"reference": "Medication/123", "display": "osimertinib"}}
    )
    assert label == "osimertinib"

    # contained Medication resolved by #ref
    codes, label = _medication_codes_label(
        {
            "medicationReference": {"reference": "#med1"},
            "contained": [
                {
                    "resourceType": "Medication",
                    "id": "med1",
                    "code": {"coding": [{"system": "http://www.nlm.nih.gov/research/umls/rxnorm", "code": "1", "display": "pembrolizumab"}]},
                }
            ],
        }
    )
    assert label == "pembrolizumab"
    assert codes and codes[0].vocabulary == "RxNorm"


# -------------------------------------------------------------------- integration


def _import_bundle(tmp_path, entries):
    path = tmp_path / "bundle.json"
    path.write_text(
        json.dumps(
            {
                "resourceType": "Bundle",
                "entry": [{"resource": r} for r in entries],
            }
        )
    )
    return import_fhir(path, input_format="fhir")


def test_resolved_condition_is_negated_and_error_is_dropped(tmp_path):
    profiles = _import_bundle(
        tmp_path,
        [
            {"resourceType": "Patient", "id": "p1", "birthDate": "1975"},
            {
                "resourceType": "Condition",
                "id": "c-active",
                "subject": {"reference": "Patient/p1"},
                "clinicalStatus": {"coding": [{"code": "active"}]},
                "code": {"text": "lung cancer"},
            },
            {
                "resourceType": "Condition",
                "id": "c-resolved",
                "subject": {"reference": "Patient/p1"},
                "clinicalStatus": {"coding": [{"code": "resolved"}]},
                "verificationStatus": {"coding": [{"code": "confirmed"}]},
                "code": {"text": "pneumonia"},
            },
            {
                "resourceType": "Condition",
                "id": "c-error",
                "subject": {"reference": "Patient/p1"},
                "verificationStatus": {"coding": [{"code": "entered-in-error"}]},
                "code": {"text": "typo diagnosis"},
            },
        ],
    )
    profile = profiles[0]
    # Partial birthDate populated age.
    assert profile.demographics.age_years is not None
    by_label = {c.label: c for c in profile.conditions}
    assert by_label["lung cancer"].negated is False
    assert by_label["pneumonia"].negated is True  # resolved -> not currently present
    assert "typo diagnosis" not in by_label  # entered-in-error dropped
    assert any(u.get("id") == "c-error" for u in profile.unsupported)


def test_observation_component_and_interpretation_captured(tmp_path):
    profiles = _import_bundle(
        tmp_path,
        [
            {"resourceType": "Patient", "id": "p1"},
            {
                "resourceType": "Observation",
                "id": "bp",
                "subject": {"reference": "Patient/p1"},
                "status": "final",
                "code": {"text": "Blood pressure"},
                "interpretation": [{"coding": [{"code": "H", "display": "High"}]}],
                "component": [
                    {"code": {"text": "Systolic"}, "valueQuantity": {"value": 160, "unit": "mmHg"}},
                    {"code": {"text": "Diastolic"}, "valueQuantity": {"value": 95, "unit": "mmHg"}},
                ],
            },
        ],
    )
    obs = profiles[0].observations[0]
    assert "Systolic: 160 mmHg" in obs.description
    assert "Diastolic: 95 mmHg" in obs.description
    assert "High" in obs.description


def test_ndjson_skips_malformed_lines(tmp_path):
    path = tmp_path / "export.ndjson"
    path.write_text(
        "\n".join(
            [
                json.dumps({"resourceType": "Patient", "id": "p1"}),
                "{ this is not valid json",
                json.dumps(
                    {
                        "resourceType": "Condition",
                        "subject": {"reference": "Patient/p1"},
                        "code": {"text": "diabetes"},
                    }
                ),
            ]
        )
    )
    profiles = import_fhir(path, input_format="fhir-ndjson")
    assert len(profiles) == 1
    assert any(c.label == "diabetes" for c in profiles[0].conditions)


def test_reference_resolution_urn_uuid_and_absolute_url(tmp_path):
    path = tmp_path / "bundle.json"
    path.write_text(
        json.dumps(
            {
                "resourceType": "Bundle",
                "entry": [
                    {"fullUrl": "urn:uuid:abc-123", "resource": {"resourceType": "Patient", "id": "p1"}},
                    {"resource": {"resourceType": "Condition", "subject": {"reference": "urn:uuid:abc-123"}, "code": {"text": "asthma"}}},
                    {"resource": {"resourceType": "Condition", "subject": {"reference": "http://ehr.example.org/fhir/Patient/p1"}, "code": {"text": "eczema"}}},
                ],
            }
        )
    )
    profile = import_fhir(path, input_format="fhir")[0]
    labels = {c.label for c in profile.conditions}
    assert {"asthma", "eczema"} <= labels  # both reference styles resolved


def test_fhir_ndjson_strict_raises_on_malformed_line(tmp_path):
    """A malformed NDJSON line is skipped in non-strict mode but must raise in
    strict mode (audit P2: importer strict raise-paths)."""
    import pytest

    from trialmatchai.interop.importers.fhir import import_fhir

    path = tmp_path / "p.ndjson"
    path.write_text(
        '{"resourceType": "Patient", "id": "P1"}\n{ this is not valid json\n',
        encoding="utf-8",
    )
    profiles = import_fhir(path, input_format="fhir-ndjson")  # non-strict: bad line skipped
    assert len(profiles) >= 1
    with pytest.raises(Exception):
        import_fhir(path, input_format="fhir-ndjson", strict=True)
