from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from trialmatchai.interop.models import (
    Demographics,
    Location,
    PatientProfile,
    Provenance,
    SourceDocument,
)
from trialmatchai.interop.utils import (
    age_years_from_birth_date,
    clean_text,
    codes_from_fhir_codeable,
    label_from_fhir_codeable,
    make_fact,
    normalize_gender,
    parse_date,
    safe_patient_id,
    source_path_string,
)
from trialmatchai.utils.logging_config import setup_logging

logger = setup_logging(__name__)


def import_fhir(
    path: str | Path,
    *,
    input_format: str = "fhir",
    strict: bool = False,
) -> list[PatientProfile]:
    source_path = Path(path)
    resources = (
        _load_ndjson(source_path, strict=strict)
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
            # Attribute the orphan to the first profile only (avoid duplicating it
            # across every profile in a multi-patient bundle).
            profiles[0].unsupported.append(
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
        except Exception as exc:
            if strict:
                raise
            logger.warning(
                "FHIR mapping failed for %s/%s: %s",
                resource.get("resourceType"),
                resource.get("id"),
                exc,
            )
            profile.unsupported.append(
                {
                    "resourceType": resource.get("resourceType"),
                    "id": resource.get("id"),
                    "reason": f"resource mapping failed: {exc}",
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
        location=_patient_location(patient.get("address")),
        provenance=[provenance],
    )


def _patient_location(address: Any) -> Location | None:
    """Extract a coarse location from a FHIR Patient.address list (first entry)."""
    if isinstance(address, dict):
        address = [address]
    if not isinstance(address, list):
        return None
    for entry in address:
        if not isinstance(entry, dict):
            continue
        country = (entry.get("country") or "").strip() or None
        state = (entry.get("state") or "").strip() or None
        city = (entry.get("city") or "").strip() or None
        if country or state or city:
            return Location(country=country, state=state, city=city)
    return None


# --------------------------------------------------------------------------- I/O


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


def _load_ndjson(path: Path, *, strict: bool = False) -> list[dict[str, Any]]:
    resources: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                if strict:
                    raise
                logger.warning(
                    "Skipping malformed NDJSON line %d in %s: %s",
                    line_number,
                    path.name,
                    exc,
                )
                continue
            if isinstance(obj, dict):
                resources.append(obj)
    return resources


# ------------------------------------------------------------ reference resolution


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
            if full_url.casefold().startswith("urn:uuid:"):
                mapping[full_url.split(":")[-1]] = profile
    return mapping


def _profile_for_resource(
    resource: Mapping[str, Any],
    profiles: list[PatientProfile],
    profiles_by_reference: Mapping[str, PatientProfile],
) -> PatientProfile | None:
    reference = _patient_reference(resource)
    if reference:
        for candidate in _reference_candidates(reference):
            if candidate in profiles_by_reference:
                return profiles_by_reference[candidate]
    if len(profiles) == 1:
        return profiles[0]
    return None


def _reference_candidates(reference: str) -> list[str]:
    """All key forms a patient reference might match (relative, absolute URL,
    urn:uuid, contained)."""
    ref = reference.strip()
    candidates = [ref]
    if ref.casefold().startswith("urn:uuid:"):
        candidates.append(ref.split(":")[-1])
    if ref.startswith("#"):
        candidates.append(ref[1:])
    if "/" in ref:
        parts = ref.rstrip("/").split("/")
        tail = parts[-1]
        candidates.append(tail)
        if len(parts) >= 2 and parts[-2].casefold() == "patient":
            candidates.append(f"Patient/{tail}")
    # de-duplicate, preserve order
    seen: set[str] = set()
    return [c for c in candidates if c and not (c in seen or seen.add(c))]


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


# ------------------------------------------------------------------ status / dates

# Statuses that mean the resource is an error or did not happen -> drop entirely.
_STATUS_DROP = {"entered-in-error", "cancelled", "not-done", "nullified", "declined"}
# clinicalStatus values that mean the item is no longer present -> negate.
_INACTIVE_CLINICAL = {"resolved", "inactive", "remission"}


def _status_code(value: Any) -> str:
    if isinstance(value, Mapping):
        for coding in value.get("coding") or []:
            code = str((coding or {}).get("code") or "").strip().casefold()
            if code:
                return code
        return clean_text(value.get("text")).casefold()
    return str(value or "").strip().casefold()


def _resource_disposition(resource_type: str, resource: Mapping[str, Any]) -> tuple[str, bool]:
    """Return ("drop"|"keep", negated) from FHIR status fields.

    - entered-in-error / cancelled / not-done -> drop (error or did not happen)
    - verificationStatus refuted -> keep but negated
    - clinicalStatus resolved/inactive on Condition/Allergy -> keep but negated
      (medications stay un-negated: a completed/stopped drug is real prior exposure)
    """
    status = _status_code(resource.get("status"))
    verification = _status_code(resource.get("verificationStatus"))
    clinical = _status_code(resource.get("clinicalStatus"))

    if status in _STATUS_DROP or verification == "entered-in-error":
        return "drop", False
    if verification == "refuted":
        return "keep", True
    if resource_type in {"Condition", "AllergyIntolerance"} and clinical in _INACTIVE_CLINICAL:
        return "keep", True
    return "keep", False


def _temporality(resource: Mapping[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = resource.get(key)
        if not value:
            continue
        if isinstance(value, str):
            return clean_text(value) or None
        if isinstance(value, Mapping):
            if value.get("start") or value.get("end"):
                span = f"{value.get('start', '')} to {value.get('end', '')}"
                return clean_text(span).strip(" to") or None
            if value.get("value") is not None:
                return clean_text(f"{value.get('value')} {value.get('unit', '')}") or None
    return None


def _annotations_text(value: Any) -> str | None:
    if isinstance(value, Mapping):
        value = [value]
    if not isinstance(value, list):
        return None
    texts = [
        clean_text((note or {}).get("text"))
        for note in value
        if isinstance(note, Mapping)
    ]
    joined = "; ".join(text for text in texts if text)
    return joined or None


# ----------------------------------------------------------------- resource router


def _add_resource(
    profile: PatientProfile,
    resource: Mapping[str, Any],
    base_provenance: Provenance,
) -> None:
    resource_type = resource.get("resourceType")
    if resource_type in {None, "Patient"}:
        return

    disposition, negated = _resource_disposition(str(resource_type), resource)
    provenance = base_provenance.model_copy(
        update={"source_resource": f"{resource_type}/{resource.get('id', 'unknown')}"}
    )
    if disposition == "drop":
        profile.unsupported.append(
            {
                "resourceType": resource_type,
                "id": resource.get("id"),
                "reason": "dropped: status indicates error or did-not-happen",
            }
        )
        return

    if resource_type == "Condition":
        _add_condition(profile, resource, provenance, negated)
    elif resource_type == "Observation":
        _add_observation(profile, resource, provenance, negated)
    elif resource_type in {
        "MedicationRequest",
        "MedicationStatement",
        "MedicationAdministration",
    }:
        _add_medication(profile, resource, provenance, negated)
    elif resource_type == "Procedure":
        _add_procedure(profile, resource, provenance, negated)
    elif resource_type == "DiagnosticReport":
        _add_diagnostic_report(profile, resource, provenance)
    elif resource_type == "DocumentReference":
        _add_document_reference(profile, resource, provenance)
    elif resource_type == "AllergyIntolerance":
        _add_allergy(profile, resource, provenance, negated)
    elif resource_type == "FamilyMemberHistory":
        _add_family_history(profile, resource, provenance)
    elif resource_type in {"MolecularSequence", "GenomicStudy"}:
        _add_genomic(profile, resource, provenance, negated)
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
    negated: bool,
) -> None:
    codeable = resource.get("code") or {}
    codes = codes_from_fhir_codeable(codeable)
    label = label_from_fhir_codeable(codeable)
    if not label:
        return
    profile.conditions.append(
        make_fact(
            category="condition",
            label=label,
            original_code=codes[0] if codes else None,
            normalized_codes=codes or None,
            provenance=provenance,
            description=_annotations_text(resource.get("note")),
            temporality=_temporality(
                resource,
                ("onsetDateTime", "onsetPeriod", "onsetAge", "onsetString", "recordedDate"),
            ),
            negated=negated,
        )
    )


def _add_observation(
    profile: PatientProfile,
    resource: Mapping[str, Any],
    provenance: Provenance,
    negated: bool,
) -> None:
    codeable = resource.get("code") or {}
    codes = codes_from_fhir_codeable(codeable)
    label = label_from_fhir_codeable(codeable)
    if not label:
        return
    category = "genomic_finding" if _is_genomic_observation(resource) else "observation"
    profile.add_fact(
        make_fact(
            category=category,
            label=label,
            original_code=codes[0] if codes else None,
            normalized_codes=codes or None,
            provenance=provenance,
            description=_observation_value(resource),
            temporality=_temporality(
                resource, ("effectiveDateTime", "effectivePeriod", "effectiveInstant")
            ),
            negated=negated,
        )
    )


def _add_medication(
    profile: PatientProfile,
    resource: Mapping[str, Any],
    provenance: Provenance,
    negated: bool,
) -> None:
    codes, label = _medication_codes_label(resource)
    if not label:
        return
    profile.medications.append(
        make_fact(
            category="medication",
            label=label,
            original_code=codes[0] if codes else None,
            normalized_codes=codes or None,
            provenance=provenance,
            description=_dosage_text(resource.get("dosageInstruction")),
            temporality=_temporality(
                resource,
                ("authoredOn", "effectiveDateTime", "effectivePeriod"),
            ),
            negated=negated,
        )
    )


def _add_procedure(
    profile: PatientProfile,
    resource: Mapping[str, Any],
    provenance: Provenance,
    negated: bool,
) -> None:
    codeable = resource.get("code") or {}
    codes = codes_from_fhir_codeable(codeable)
    label = label_from_fhir_codeable(codeable)
    if label:
        profile.procedures.append(
            make_fact(
                category="procedure",
                label=label,
                original_code=codes[0] if codes else None,
                normalized_codes=codes or None,
                provenance=provenance,
                temporality=_temporality(
                    resource, ("performedDateTime", "performedPeriod", "performedString")
                ),
                negated=negated,
            )
        )


def _add_diagnostic_report(
    profile: PatientProfile,
    resource: Mapping[str, Any],
    provenance: Provenance,
) -> None:
    codeable = resource.get("code") or {}
    codes = codes_from_fhir_codeable(codeable)
    label = label_from_fhir_codeable(codeable) or clean_text(resource.get("conclusion"))
    if label:
        profile.diagnostic_reports.append(
            make_fact(
                category="diagnostic_report",
                label=label,
                original_code=codes[0] if codes else None,
                normalized_codes=codes or None,
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
    negated: bool,
) -> None:
    codeable = resource.get("code") or {}
    codes = codes_from_fhir_codeable(codeable)
    label = label_from_fhir_codeable(codeable)
    if label:
        profile.conditions.append(
            make_fact(
                category="condition",
                label=f"Allergy: {label}",
                original_code=codes[0] if codes else None,
                normalized_codes=codes or None,
                provenance=provenance,
                negated=negated,
                extra={"clinical_status": _status_code(resource.get("clinicalStatus")) or None},
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
                extra={"relationship": relationship},
            )
        )


def _add_genomic(
    profile: PatientProfile,
    resource: Mapping[str, Any],
    provenance: Provenance,
    negated: bool = False,
) -> None:
    label = (
        label_from_fhir_codeable(resource.get("type") or {})
        or clean_text(resource.get("id"))
        or "Genomic finding"
    )
    profile.genomic_findings.append(
        make_fact(
            category="genomic_finding",
            label=label,
            provenance=provenance,
            negated=negated,
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
            )
        )


# ------------------------------------------------------------------- value helpers


def _medication_codes_label(resource: Mapping[str, Any]):
    """Resolve a medication's codes + label across R4/R5 shapes and references."""
    codeable = resource.get("medicationCodeableConcept")
    if not isinstance(codeable, Mapping):
        medication = resource.get("medication")
        if isinstance(medication, Mapping):
            # R5 wraps as medication.concept; otherwise treat as the concept itself.
            concept = medication.get("concept")
            codeable = concept if isinstance(concept, Mapping) else medication
    if isinstance(codeable, Mapping) and (codeable.get("coding") or codeable.get("text")):
        label = label_from_fhir_codeable(codeable)
        if label:
            return codes_from_fhir_codeable(codeable), label

    # medicationReference (R4) / medication.reference (R5), incl. contained Medication.
    reference = resource.get("medicationReference")
    if not isinstance(reference, Mapping) and isinstance(resource.get("medication"), Mapping):
        reference = resource["medication"].get("reference")
    if isinstance(reference, Mapping):
        contained = _resolve_contained(resource, reference.get("reference"))
        if contained is not None:
            contained_code = contained.get("code") or {}
            label = label_from_fhir_codeable(contained_code) or clean_text(
                reference.get("display")
            )
            if label:
                return codes_from_fhir_codeable(contained_code), label
        display = clean_text(reference.get("display"))
        if display:
            return [], display
    return [], None


def _resolve_contained(resource: Mapping[str, Any], reference: Any) -> Mapping[str, Any] | None:
    if not isinstance(reference, str) or not reference.startswith("#"):
        return None
    target = reference[1:]
    for item in resource.get("contained") or []:
        if isinstance(item, Mapping) and str(item.get("id")) == target:
            return item
    return None


def _dosage_text(value: Any) -> str | None:
    if isinstance(value, Mapping):
        value = [value]
    if not isinstance(value, list):
        return None
    texts = [
        clean_text((dosage or {}).get("text"))
        for dosage in value
        if isinstance(dosage, Mapping)
    ]
    joined = "; ".join(text for text in texts if text)
    return joined or None


def _observation_value(resource: Mapping[str, Any]) -> str | None:
    parts: list[str] = []
    main = _value_x(resource)
    if main:
        parts.append(main)
    interpretation = _interpretation_text(resource.get("interpretation"))
    if interpretation:
        parts.append(f"[{interpretation}]")
    for component in resource.get("component") or []:
        if not isinstance(component, Mapping):
            continue
        component_label = label_from_fhir_codeable(component.get("code") or {})
        component_value = _value_x(component)
        if component_label and component_value:
            parts.append(f"{component_label}: {component_value}")
        elif component_label:
            parts.append(component_label)
        elif component_value:
            parts.append(component_value)
    return clean_text(" ".join(parts)) or None


def _value_x(node: Mapping[str, Any]) -> str | None:
    quantity = node.get("valueQuantity")
    if isinstance(quantity, Mapping):
        comparator = clean_text(quantity.get("comparator"))
        unit = clean_text(quantity.get("unit") or quantity.get("code"))
        value = quantity.get("value")
        if value is not None or comparator:
            number = f"{comparator}{value if value is not None else ''}".strip()
            return clean_text(f"{number} {unit}") or None
    concept = node.get("valueCodeableConcept")
    if isinstance(concept, Mapping):
        return label_from_fhir_codeable(concept) or None
    value_range = node.get("valueRange")
    if isinstance(value_range, Mapping):
        low = (value_range.get("low") or {}).get("value")
        high = (value_range.get("high") or {}).get("value")
        span = f"{'' if low is None else low}-{'' if high is None else high}"
        return clean_text(span).strip("-") or None
    ratio = node.get("valueRatio")
    if isinstance(ratio, Mapping):
        numerator = (ratio.get("numerator") or {}).get("value")
        denominator = (ratio.get("denominator") or {}).get("value")
        return clean_text(f"{numerator}/{denominator}") or None
    for key in ("valueString", "valueBoolean", "valueInteger", "valueDateTime", "valueTime"):
        if key in node:
            return clean_text(node.get(key)) or None
    return None


def _interpretation_text(value: Any) -> str | None:
    if isinstance(value, Mapping):
        value = [value]
    if not isinstance(value, list):
        return None
    labels = [
        label_from_fhir_codeable(item) for item in value if isinstance(item, Mapping)
    ]
    joined = ", ".join(label for label in labels if label)
    return joined or None


_GENOMIC_HINTS = (
    "genetic",
    "genomic",
    "variant",
    "mutation",
    "sequence variant",
    "molecular",
)


def _is_genomic_observation(resource: Mapping[str, Any]) -> bool:
    haystack = " ".join(
        [
            clean_text(resource.get("category")).casefold(),
            label_from_fhir_codeable(resource.get("code") or {}).casefold(),
        ]
    )
    return any(hint in haystack for hint in _GENOMIC_HINTS)


def _document_url(resource: Mapping[str, Any]) -> str | None:
    for content in resource.get("content") or []:
        attachment = (content or {}).get("attachment") or {}
        if attachment.get("url"):
            return clean_text(attachment.get("url"))
    return None
