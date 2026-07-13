from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from trialmatchai.interop.models import Demographics, PatientProfile, Provenance, SourceDocument
from trialmatchai.interop.utils import (
    clean_text,
    code_from_ontology_class,
    make_fact,
    normalize_gender,
    parse_date,
    parse_iso8601_age_years,
    safe_patient_id,
    source_path_string,
)
from trialmatchai.schemas.phenopacket import Phenopacket
from trialmatchai.utils.logging_config import setup_logging

logger = setup_logging(__name__)


def import_phenopacket(
    path: str | Path,
    *,
    strict: bool = False,
) -> PatientProfile:
    packet_path = Path(path)
    data = json.loads(packet_path.read_text(encoding="utf-8"))
    try:
        Phenopacket.model_validate(data)
    except Exception:
        if strict:
            raise
    patient_id = safe_patient_id(data.get("id"), packet_path.stem)
    provenance = Provenance(
        source_format="phenopacket",
        source_id=patient_id,
        source_path=source_path_string(packet_path),
    )
    profile = PatientProfile(
        patient_id=patient_id,
        demographics=_demographics(data.get("subject") or {}),
        provenance=[provenance],
    )
    # Isolate each section: one malformed item must not abort the whole import in
    # non-strict mode (only re-raise when strict).
    for add_section in (
        _add_phenotypes,
        _add_diseases,
        _add_biosamples,
        _add_measurements,
        _add_actions,
        _add_interpretations,
        _add_family,
        _add_files,
    ):
        try:
            add_section(profile, data, provenance)
        except Exception:
            logger.warning(
                "phenopacket: section %s failed for %s; skipping",
                add_section.__name__,
                patient_id,
                exc_info=True,
            )
            if strict:
                raise
    return profile


def _demographics(subject: Mapping[str, Any]) -> Demographics:
    age_years = None
    encounter = subject.get("timeAtLastEncounter") or {}
    if isinstance(encounter, Mapping):
        age = encounter.get("age") or {}
        if isinstance(age, Mapping):
            age_years = parse_iso8601_age_years(age.get("iso8601duration"))
    birth_date = parse_date(subject.get("dateOfBirth"))
    return Demographics(
        sex=normalize_gender(subject.get("sex")),
        gender=normalize_gender(subject.get("gender")),
        birth_date=birth_date,
        age_years=age_years,
        species=clean_text((subject.get("taxonomy") or {}).get("label")),
        description=clean_text(subject.get("description")) or None,
    )


def _add_phenotypes(
    profile: PatientProfile,
    data: Mapping[str, Any],
    provenance: Provenance,
) -> None:
    for index, item in enumerate(data.get("phenotypicFeatures") or []):
        feature = item.get("type") or {}
        code = code_from_ontology_class(feature)
        label = clean_text(feature.get("label") or feature.get("id"))
        if not label:
            continue
        profile.phenotypes.append(
            make_fact(
                category="phenotype",
                label=label,
                original_code=code,
                provenance=provenance.model_copy(
                    update={"source_resource": f"phenotypicFeatures[{index}]"}
                ),
                description=clean_text(item.get("description")) or None,
                negated=bool(item.get("excluded", False)),
                temporality=clean_text(item.get("onset")) or None,
                extra={"severity": item.get("severity"), "modifiers": item.get("modifiers")},
            )
        )


def _add_diseases(
    profile: PatientProfile,
    data: Mapping[str, Any],
    provenance: Provenance,
) -> None:
    for index, item in enumerate(data.get("diseases") or []):
        term = item.get("term") or {}
        code = code_from_ontology_class(term)
        label = clean_text(term.get("label") or term.get("id"))
        if not label:
            continue
        category = "cancer" if item.get("tnmFinding") or item.get("diseaseStage") else "condition"
        profile.add_fact(
            make_fact(
                category=category,
                label=label,
                original_code=code,
                provenance=provenance.model_copy(
                    update={"source_resource": f"diseases[{index}]"}
                ),
                description=clean_text(item.get("description")) or None,
                temporality=clean_text(item.get("onset")) or None,
                extra={
                    "disease_stage": item.get("diseaseStage") or [],
                    "tnm_finding": item.get("tnmFinding") or [],
                },
            )
        )


def _add_biosamples(
    profile: PatientProfile,
    data: Mapping[str, Any],
    provenance: Provenance,
) -> None:
    for index, sample in enumerate(data.get("biosamples") or []):
        label = clean_text(
            (sample.get("histologicalDiagnosis") or {}).get("label")
            or (sample.get("sampledTissue") or {}).get("label")
            or (sample.get("sampleType") or {}).get("label")
            or sample.get("id")
        )
        if not label:
            continue
        profile.diagnostic_reports.append(
            make_fact(
                category="diagnostic_report",
                label=f"Biosample: {label}",
                provenance=provenance.model_copy(
                    update={"source_resource": f"biosamples[{index}]"}
                ),
                description=clean_text(sample.get("description")) or None,
                extra=sample,
            )
        )


def _add_measurements(
    profile: PatientProfile,
    data: Mapping[str, Any],
    provenance: Provenance,
) -> None:
    for index, item in enumerate(data.get("measurements") or []):
        assay = item.get("assay") or {}
        code = code_from_ontology_class(assay)
        label = clean_text(assay.get("label") or assay.get("id"))
        if not label:
            continue
        value = item.get("value") or {}
        profile.observations.append(
            make_fact(
                category="observation",
                label=label,
                original_code=code,
                provenance=provenance.model_copy(
                    update={"source_resource": f"measurements[{index}]"}
                ),
                description=clean_text(value or item.get("description")) or None,
                extra={"value": value},
            )
        )


def _add_actions(
    profile: PatientProfile,
    data: Mapping[str, Any],
    provenance: Provenance,
) -> None:
    for index, action in enumerate(data.get("medicalActions") or []):
        if action.get("treatment"):
            tx = action["treatment"]
            agent = tx.get("agent") or {}
            code = code_from_ontology_class(agent)
            label = clean_text(agent.get("label") or agent.get("id") or tx.get("description"))
            if label:
                profile.medications.append(
                    make_fact(
                        category="medication",
                        label=label,
                        original_code=code,
                        provenance=provenance.model_copy(
                            update={"source_resource": f"medicalActions[{index}].treatment"}
                        ),
                        description=clean_text(tx.get("description")) or None,
                        extra=tx,
                    )
                )
        if action.get("procedure"):
            proc = action["procedure"]
            codeable = proc.get("code") or {}
            code = code_from_ontology_class(codeable)
            label = clean_text(codeable.get("label") or codeable.get("id") or proc.get("description"))
            if label:
                profile.procedures.append(
                    make_fact(
                        category="procedure",
                        label=label,
                        original_code=code,
                        provenance=provenance.model_copy(
                            update={"source_resource": f"medicalActions[{index}].procedure"}
                        ),
                        description=clean_text(proc.get("description")) or None,
                        temporality=clean_text(proc.get("performed")) or None,
                        extra=proc,
                    )
                )


def _add_interpretations(
    profile: PatientProfile,
    data: Mapping[str, Any],
    provenance: Provenance,
) -> None:
    for index, item in enumerate(data.get("interpretations") or []):
        diagnosis = item.get("diagnosis") or {}
        for gi_index, interpretation in enumerate(diagnosis.get("genomicInterpretations") or []):
            variant = (interpretation.get("variantInterpretation") or {}).get(
                "variationDescriptor"
            ) or {}
            gene = (variant.get("geneContext") or {}).get("symbol")
            label = clean_text(" ".join(part for part in [gene, variant.get("label")] if part))
            if label:
                profile.genomic_findings.append(
                    make_fact(
                        category="genomic_finding",
                        label=label,
                        provenance=provenance.model_copy(
                            update={
                                "source_resource": (
                                    f"interpretations[{index}].diagnosis."
                                    f"genomicInterpretations[{gi_index}]"
                                )
                            }
                        ),
                        description=clean_text(diagnosis.get("description")) or None,
                        extra=variant,
                    )
                )


def _add_family(
    profile: PatientProfile,
    data: Mapping[str, Any],
    provenance: Provenance,
) -> None:
    family = data.get("family") or {}
    for index, relative in enumerate(family.get("relatives") or []):
        label = clean_text(relative.get("description") or relative.get("id"))
        if not label:
            continue
        profile.family_history.append(
            make_fact(
                category="family_history",
                label=label,
                provenance=provenance.model_copy(
                    update={"source_resource": f"family.relatives[{index}]"}
                ),
                extra=relative,
            )
        )


def _add_files(
    profile: PatientProfile,
    data: Mapping[str, Any],
    provenance: Provenance,
) -> None:
    for index, file_ref in enumerate(data.get("files") or []):
        uri = clean_text(file_ref.get("uri") or file_ref.get("path"))
        profile.source_documents.append(
            SourceDocument(
                document_id=clean_text(file_ref.get("individualToFileIdentifiers")) or f"file-{index}",
                title=clean_text(file_ref.get("description")) or None,
                document_type=clean_text(file_ref.get("fileAttributes")) or None,
                url=uri or None,
                provenance=provenance.model_copy(
                    update={"source_resource": f"files[{index}]"}
                ),
            )
        )
