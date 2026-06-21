from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from trialmatchai.interop.models import Demographics, PatientProfile, Provenance, SourceDocument
from trialmatchai.interop.utils import (
    age_years_from_birth_date,
    clean_text,
    code_from_fhir_codeable,
    label_from_fhir_codeable,
    make_fact,
    normalize_gender,
    parse_date,
    safe_patient_id,
    source_path_string,
)


def import_fhir(
    path: str | Path,
    *,
    input_format: str = "fhir",
    strict: bool = False,
) -> list[PatientProfile]:
    source_path = Path(path)
    resources = (
        _load_ndjson(source_path)
        if input_format == "fhir-ndjson"
        else _load_json_resources(source_path)
    )
    patients = [res for res in resources if res.get("resourceType") == "Patient"]
    if not patients:
        patients = [{}]

    profiles = [
        _profile_from_patient(patient, source_path=source_path, input_format=input_format)
        for patient in patients
    ]
    profiles_by_reference = _profiles_by_reference(patients, profiles)

    for resource in resources:
        if resource.get("resourceType") == "Patient":
            continue
        profile = _profile_for_resource(resource, profiles, profiles_by_reference)
        if profile is None:
            message = "FHIR resource has no resolvable patient reference"
            if strict:
                raise ValueError(
                    f"{message}: {resource.get('resourceType')}/{resource.get('id')}"
                )
            for candidate in profiles:
                candidate.unsupported.append(
                    {
                        "resourceType": resource.get("resourceType"),
                        "id": resource.get("id"),
                        "reason": message,
                    }
                )
            continue
        try:
            base_provenance = profile.provenance[0]
            _add_resource(profile, resource, base_provenance)
        except Exception:
            if strict:
                raise
            profile.unsupported.append(
                {
                    "resourceType": resource.get("resourceType"),
                    "id": resource.get("id"),
                    "reason": "resource mapping failed",
                }
            )
    return profiles


def _profile_from_patient(
    patient: Mapping[str, Any],
    *,
    source_path: Path,
    input_format: str,
) -> PatientProfile:
    patient_id = safe_patient_id(patient.get("id"), source_path.stem)
    provenance = Provenance(
        source_format=input_format,
        source_id=patient_id,
        source_path=source_path_string(source_path),
    )
    birth_date = parse_date(patient.get("birthDate"))
    return PatientProfile(
        patient_id=patient_id,
        demographics=Demographics(
            sex=normalize_gender(patient.get("gender")),
            gender=normalize_gender(patient.get("gender")),
            birth_date=birth_date,
            age_years=age_years_from_birth_date(birth_date),
        ),
        provenance=[provenance],
    )


def _load_json_resources(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if not isinstance(data, dict):
        return []
    if data.get("resourceType") == "Bundle":
        resources = []
        for entry in data.get("entry") or []:
            if not isinstance(entry, dict) or not isinstance(entry.get("resource"), dict):
                continue
            resource = dict(entry["resource"])
            if entry.get("fullUrl"):
                resource["_bundle_full_url"] = entry["fullUrl"]
            resources.append(resource)
        return resources
    return [data]


def _load_ndjson(path: Path) -> list[dict[str, Any]]:
    resources = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                resources.append(json.loads(line))
    return resources


def _profiles_by_reference(
    patients: list[Mapping[str, Any]],
    profiles: list[PatientProfile],
) -> dict[str, PatientProfile]:
    mapping: dict[str, PatientProfile] = {}
    for patient, profile in zip(patients, profiles):
        patient_id = str(patient.get("id") or "").strip()
        if patient_id:
            mapping[patient_id] = profile
            mapping[f"Patient/{patient_id}"] = profile
        full_url = str(patient.get("_bundle_full_url") or "").strip()
        if full_url:
            mapping[full_url] = profile
    return mapping


def _profile_for_resource(
    resource: Mapping[str, Any],
    profiles: list[PatientProfile],
    profiles_by_reference: Mapping[str, PatientProfile],
) -> PatientProfile | None:
    reference = _patient_reference(resource)
    if reference:
        return profiles_by_reference.get(reference)
    if len(profiles) == 1:
        return profiles[0]
    return None


def _patient_reference(resource: Mapping[str, Any]) -> str | None:
    for key in ("subject", "patient", "beneficiary"):
        reference = _reference_value(resource.get(key))
        if reference:
            return reference
    return None


def _reference_value(value: Any) -> str | None:
    if isinstance(value, Mapping):
        reference = value.get("reference")
        if isinstance(reference, str) and reference.strip():
            return reference.strip()
    return None


def _add_resource(
    profile: PatientProfile,
    resource: Mapping[str, Any],
    base_provenance: Provenance,
) -> None:
    resource_type = resource.get("resourceType")
    if resource_type in {None, "Patient"}:
        return
    provenance = base_provenance.model_copy(
        update={
            "source_resource": f"{resource_type}/{resource.get('id', 'unknown')}",
        }
    )
    if resource_type == "Condition":
        _add_condition(profile, resource, provenance)
    elif resource_type == "Observation":
        _add_observation(profile, resource, provenance)
    elif resource_type in {
        "MedicationRequest",
        "MedicationStatement",
        "MedicationAdministration",
    }:
        _add_medication(profile, resource, provenance)
    elif resource_type == "Procedure":
        _add_procedure(profile, resource, provenance)
    elif resource_type == "DiagnosticReport":
        _add_diagnostic_report(profile, resource, provenance)
    elif resource_type == "DocumentReference":
        _add_document_reference(profile, resource, provenance)
    elif resource_type == "AllergyIntolerance":
        _add_allergy(profile, resource, provenance)
    elif resource_type == "FamilyMemberHistory":
        _add_family_history(profile, resource, provenance)
    elif resource_type in {"MolecularSequence", "GenomicStudy"}:
        _add_genomic(profile, resource, provenance)
    elif resource_type == "Specimen":
        _add_specimen(profile, resource, provenance)
    else:
        profile.unsupported.append(
            {
                "resourceType": resource_type,
                "id": resource.get("id"),
                "reason": "FHIR resource type not mapped in v1 importer",
            }
        )


def _add_condition(
    profile: PatientProfile,
    resource: Mapping[str, Any],
    provenance: Provenance,
) -> None:
    code = code_from_fhir_codeable(resource.get("code") or {})
    label = label_from_fhir_codeable(resource.get("code") or {})
    if not label:
        return
    profile.conditions.append(
        make_fact(
            category="condition",
            label=label,
            original_code=code,
            provenance=provenance,
            description=clean_text(resource.get("note")) or None,
            temporality=clean_text(resource.get("onsetDateTime")) or None,
            negated=_is_negated(resource),
        )
    )


def _add_observation(
    profile: PatientProfile,
    resource: Mapping[str, Any],
    provenance: Provenance,
) -> None:
    code = code_from_fhir_codeable(resource.get("code") or {})
    label = label_from_fhir_codeable(resource.get("code") or {})
    if not label:
        return
    value = _observation_value(resource)
    category = "genomic_finding" if _is_genomic_observation(resource) else "observation"
    profile.add_fact(
        make_fact(
            category=category,
            label=label,
            original_code=code,
            provenance=provenance,
            description=value,
            temporality=clean_text(resource.get("effectiveDateTime")) or None,
        )
    )


def _add_medication(
    profile: PatientProfile,
    resource: Mapping[str, Any],
    provenance: Provenance,
) -> None:
    medication = (
        resource.get("medicationCodeableConcept")
        or resource.get("medication")
        or resource.get("contained")
        or {}
    )
    code = code_from_fhir_codeable(medication) if isinstance(medication, Mapping) else None
    label = (
        label_from_fhir_codeable(medication)
        if isinstance(medication, Mapping)
        else clean_text(medication)
    )
    if not label:
        return
    profile.medications.append(
        make_fact(
            category="medication",
            label=label,
            original_code=code,
            provenance=provenance,
            description=clean_text(resource.get("dosageInstruction")) or None,
        )
    )


def _add_procedure(
    profile: PatientProfile,
    resource: Mapping[str, Any],
    provenance: Provenance,
) -> None:
    code = code_from_fhir_codeable(resource.get("code") or {})
    label = label_from_fhir_codeable(resource.get("code") or {})
    if label:
        profile.procedures.append(
            make_fact(
                category="procedure",
                label=label,
                original_code=code,
                provenance=provenance,
                temporality=clean_text(resource.get("performedDateTime")) or None,
            )
        )


def _add_diagnostic_report(
    profile: PatientProfile,
    resource: Mapping[str, Any],
    provenance: Provenance,
) -> None:
    code = code_from_fhir_codeable(resource.get("code") or {})
    label = label_from_fhir_codeable(resource.get("code") or {}) or clean_text(
        resource.get("conclusion")
    )
    if label:
        profile.diagnostic_reports.append(
            make_fact(
                category="diagnostic_report",
                label=label,
                original_code=code,
                provenance=provenance,
                description=clean_text(resource.get("conclusion")) or None,
            )
        )


def _add_document_reference(
    profile: PatientProfile,
    resource: Mapping[str, Any],
    provenance: Provenance,
) -> None:
    profile.source_documents.append(
        SourceDocument(
            document_id=clean_text(resource.get("id")) or provenance.source_resource or "document",
            title=clean_text(resource.get("description") or resource.get("docStatus")) or None,
            document_type=label_from_fhir_codeable(resource.get("type") or {}) or None,
            url=_document_url(resource),
            provenance=provenance,
        )
    )


def _add_allergy(
    profile: PatientProfile,
    resource: Mapping[str, Any],
    provenance: Provenance,
) -> None:
    code = code_from_fhir_codeable(resource.get("code") or {})
    label = label_from_fhir_codeable(resource.get("code") or {})
    if label:
        profile.conditions.append(
            make_fact(
                category="condition",
                label=f"Allergy: {label}",
                original_code=code,
                provenance=provenance,
                extra={"clinical_status": resource.get("clinicalStatus")},
            )
        )


def _add_family_history(
    profile: PatientProfile,
    resource: Mapping[str, Any],
    provenance: Provenance,
) -> None:
    relationship = label_from_fhir_codeable(resource.get("relationship") or {})
    conditions = resource.get("condition") or []
    label = relationship or clean_text(resource.get("id"))
    if conditions:
        labels = [
            label_from_fhir_codeable(condition.get("code") or {})
            for condition in conditions
            if isinstance(condition, Mapping)
        ]
        label = f"{relationship}: {', '.join(item for item in labels if item)}"
    if label:
        profile.family_history.append(
            make_fact(
                category="family_history",
                label=label,
                provenance=provenance,
                extra={"relationship": relationship, "conditions": conditions},
            )
        )


def _add_genomic(
    profile: PatientProfile,
    resource: Mapping[str, Any],
    provenance: Provenance,
) -> None:
    label = clean_text(resource.get("id") or resource.get("type") or "Genomic finding")
    profile.genomic_findings.append(
        make_fact(
            category="genomic_finding",
            label=label,
            provenance=provenance,
            extra=dict(resource),
        )
    )


def _add_specimen(
    profile: PatientProfile,
    resource: Mapping[str, Any],
    provenance: Provenance,
) -> None:
    label = label_from_fhir_codeable(resource.get("type") or {}) or clean_text(
        resource.get("id")
    )
    if label:
        profile.diagnostic_reports.append(
            make_fact(
                category="diagnostic_report",
                label=f"Specimen: {label}",
                provenance=provenance,
                extra=dict(resource),
            )
        )


def _observation_value(resource: Mapping[str, Any]) -> str | None:
    if resource.get("valueQuantity"):
        value = resource["valueQuantity"]
        return clean_text(
            f"{value.get('value', '')} {value.get('unit') or value.get('code') or ''}"
        )
    if resource.get("valueCodeableConcept"):
        return label_from_fhir_codeable(resource.get("valueCodeableConcept") or {})
    for key in ("valueString", "valueBoolean", "valueInteger", "valueDateTime"):
        if key in resource:
            return clean_text(resource.get(key))
    return None


def _is_negated(resource: Mapping[str, Any]) -> bool:
    verification = resource.get("verificationStatus") or {}
    text = label_from_fhir_codeable(verification).casefold()
    return "refuted" in text or "entered-in-error" in text


def _is_genomic_observation(resource: Mapping[str, Any]) -> bool:
    categories = resource.get("category") or []
    text = clean_text(categories).casefold()
    return "genetic" in text or "genomic" in text


def _document_url(resource: Mapping[str, Any]) -> str | None:
    for content in resource.get("content") or []:
        attachment = (content or {}).get("attachment") or {}
        if attachment.get("url"):
            return clean_text(attachment.get("url"))
    return None
