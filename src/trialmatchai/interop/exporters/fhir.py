from __future__ import annotations

from trialmatchai.interop.models import ClinicalFact, PatientProfile


def profile_to_fhir_bundle(profile: PatientProfile) -> dict:
    patient_ref = f"Patient/{profile.patient_id}"
    entries = [
        {
            "resource": {
                "resourceType": "Patient",
                "id": profile.patient_id,
                "gender": _fhir_gender(profile.demographics.sex),
                **(
                    {"birthDate": profile.demographics.birth_date.isoformat()}
                    if profile.demographics.birth_date
                    else {}
                ),
            }
        }
    ]
    entries.extend(
        _fact_entry("Condition", fact, patient_ref) for fact in profile.conditions
    )
    entries.extend(
        _fact_entry("Observation", fact, patient_ref)
        for fact in [*profile.phenotypes, *profile.observations]
    )
    entries.extend(
        _fact_entry("MedicationStatement", fact, patient_ref)
        for fact in profile.medications
    )
    entries.extend(
        _fact_entry("Procedure", fact, patient_ref) for fact in profile.procedures
    )
    entries.extend(
        _fact_entry("DiagnosticReport", fact, patient_ref)
        for fact in profile.diagnostic_reports
    )
    return {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": entries,
    }


def _fact_entry(resource_type: str, fact: ClinicalFact, patient_ref: str) -> dict:
    return {
        "resource": {
            "resourceType": resource_type,
            "id": fact.fact_id,
            "subject": {"reference": patient_ref},
            "code": _codeable(fact),
            "note": [{"text": fact.description or fact.evidence_text}]
            if fact.description or fact.evidence_text
            else [],
        }
    }


def _codeable(fact: ClinicalFact) -> dict:
    code = fact.normalized_codes[0] if fact.normalized_codes else fact.original_code
    if code is None:
        return {"text": fact.label}
    return {
        "text": code.label or fact.label,
        "coding": [
            {
                "system": code.system or code.vocabulary,
                "code": code.code,
                "display": code.label or fact.label,
            }
        ],
    }


def _fhir_gender(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.casefold()
    if normalized in {"male", "female", "other", "unknown"}:
        return normalized
    return "unknown"
