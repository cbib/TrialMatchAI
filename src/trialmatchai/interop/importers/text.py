from __future__ import annotations

from pathlib import Path
from typing import Any

from trialmatchai.interop.models import Demographics, PatientNote, PatientProfile, Provenance
from trialmatchai.interop.utils import make_fact, safe_patient_id, source_path_string


ENTITY_GROUP_TO_CATEGORY = {
    "disease": "condition",
    "condition": "condition",
    "drug": "medication",
    "medication": "medication",
    "procedure": "procedure",
    "diagnostic_test": "observation",
    "laboratory_test": "observation",
    "radiology": "diagnostic_report",
    "sign_symptom": "phenotype",
    "gene": "genomic_finding",
    "cell_type": "phenotype",
    "species": "phenotype",
}


def import_text_note(
    path: str | Path,
    *,
    entity_annotator: Any | None = None,
) -> PatientProfile:
    note_path = Path(path)
    text = note_path.read_text(encoding="utf-8")
    patient_id = safe_patient_id(note_path.stem, note_path.stem)
    provenance = Provenance(
        source_format="text",
        source_id=patient_id,
        source_path=source_path_string(note_path),
        source_field="note_text",
    )
    entities = _annotate(text, entity_annotator)
    profile = PatientProfile(
        patient_id=patient_id,
        demographics=Demographics(),
        notes=[
            PatientNote(
                note_id=f"{patient_id}-note",
                text=text,
                entities=entities,
                provenance=provenance,
            )
        ],
        provenance=[provenance],
    )
    for entity in entities:
        fact = _entity_to_fact(entity, provenance)
        if fact is not None:
            profile.add_fact(fact)
    return profile


def _annotate(text: str, entity_annotator: Any | None) -> list[dict]:
    if entity_annotator is None:
        return []
    if hasattr(entity_annotator, "annotate_texts_in_parallel"):
        result = entity_annotator.annotate_texts_in_parallel([text], max_workers=1)
        return list(result[0]) if result else []
    if hasattr(entity_annotator, "annotate_texts"):
        result = entity_annotator.annotate_texts([text])
        annotations = result[0] if result else []
        return [
            annotation.to_dict() if hasattr(annotation, "to_dict") else dict(annotation)
            for annotation in annotations
        ]
    return []


def _entity_to_fact(entity: dict, provenance: Provenance):
    if entity.get("error_code"):
        return None
    group = str(entity.get("entity_group") or entity.get("class") or "").casefold()
    category = ENTITY_GROUP_TO_CATEGORY.get(group, "observation")
    text = str(entity.get("text") or entity.get("entity") or "").strip()
    if not text:
        return None
    normalized_codes = []
    for normalized_id in entity.get("normalized_id") or []:
        if normalized_id == "CUI-less" or ":" not in normalized_id:
            continue
        vocabulary, code = normalized_id.split(":", 1)
        normalized_codes.append(
            {
                "vocabulary": vocabulary,
                "code": code,
                "label": text,
                "confidence": entity.get("linker_score") or entity.get("score"),
                "mapping_status": "normalized",
            }
        )
    from trialmatchai.interop.models import NormalizedCode

    return make_fact(
        category=category,
        label=text,
        provenance=provenance,
        normalized_codes=[NormalizedCode.model_validate(code) for code in normalized_codes],
        evidence_text=text,
        evidence_start=entity.get("start"),
        evidence_end=entity.get("end"),
        confidence=entity.get("score"),
        extra={
            "entity_group": entity.get("entity_group"),
            "synonyms": entity.get("synonyms") or [],
            "concept_candidates": entity.get("concept_candidates") or [],
            "linker_status": entity.get("linker_status"),
        },
    )
