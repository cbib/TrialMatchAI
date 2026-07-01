from __future__ import annotations

import json

import pandas as pd

from trialmatchai.interop import detect_patient_input_format, import_patient_path
from trialmatchai.interop.exporters import (
    profile_to_fhir_bundle,
    profile_to_matching_summary,
    profile_to_phenopacket,
)
from trialmatchai.interop.models import PatientProfile
from trialmatchai.interop.narrative import render_patient_narrative


def test_space_separated_entity_group_maps_like_underscore_form():
    # Recognizer/schema groups can arrive space-separated ("sign symptom", "cell type") while
    # ENTITY_GROUP_TO_CATEGORY keys are underscore-joined; without normalization they fall
    # through to the "observation" default and mis-type the fact.
    from trialmatchai.interop.importers.text import _entity_to_fact
    from trialmatchai.interop.models import Provenance

    prov = Provenance(
        source_format="text", source_id="p1", source_path="note.txt", source_field="note_text"
    )
    assert _entity_to_fact({"entity_group": "sign symptom", "text": "fatigue"}, prov).category == "phenotype"
    assert _entity_to_fact({"entity_group": "cell type", "text": "T cell"}, prov).category == "phenotype"
    assert _entity_to_fact({"entity_group": "sign_symptom", "text": "fatigue"}, prov).category == "phenotype"
    assert _entity_to_fact({"entity_group": "totally unknown", "text": "x"}, prov).category == "observation"


class FakeAnnotator:
    def annotate_texts_in_parallel(self, texts, max_workers=1, retries=1, delay=0):
        del max_workers, retries, delay
        text = texts[0]
        start = text.index("melanoma")
        return [
            [
                {
                    "entity_group": "disease",
                    "text": "melanoma",
                    "start": start,
                    "end": start + len("melanoma"),
                    "score": 0.97,
                    "normalized_id": ["SNOMED:372244006"],
                    "synonyms": ["malignant melanoma"],
                    "linker_score": 0.91,
                    "linker_status": "linked",
                }
            ]
        ]


def test_patient_profile_schema_and_summary():
    schema = PatientProfile.model_json_schema()
    assert "patient_id" in schema["properties"]

    profile = PatientProfile(patient_id="p1")
    summary = profile_to_matching_summary(profile)
    assert summary["patient_id"] == "p1"
    assert summary["age"] == "all"
    assert summary["patient_narrative"]


def test_text_importer_preserves_offsets_and_entities(tmp_path):
    note = tmp_path / "patient-note.txt"
    note.write_text("Patient has metastatic melanoma.", encoding="utf-8")

    profile = import_patient_path(
        note,
        input_format="text",
        entity_annotator=FakeAnnotator(),
    )[0]

    assert detect_patient_input_format(note) == "text"
    assert profile.conditions[0].label == "melanoma"
    assert profile.conditions[0].evidence_start == 23
    assert profile.conditions[0].normalized_codes[0].vocabulary == "SNOMED"
    assert profile.notes[0].entities[0]["linker_status"] == "linked"


def test_phenopacket_importer_maps_core_sections(tmp_path):
    packet = {
        "id": "patient-phen",
        "metaData": {},
        "subject": {
            "sex": "FEMALE",
            "timeAtLastEncounter": {"age": {"iso8601duration": "P42Y"}},
        },
        "phenotypicFeatures": [
            {"type": {"id": "HP:0001250", "label": "Seizure"}}
        ],
        "diseases": [
            {"term": {"id": "MONDO:0007254", "label": "breast cancer"}}
        ],
        "measurements": [
            {"assay": {"id": "LOINC:718-7", "label": "Hemoglobin"}}
        ],
        "medicalActions": [
            {"treatment": {"agent": {"id": "RxNorm:123", "label": "trastuzumab"}}}
        ],
    }
    path = tmp_path / "patient.json"
    path.write_text(json.dumps(packet), encoding="utf-8")

    profile = import_patient_path(path)[0]

    assert detect_patient_input_format(path) == "phenopacket"
    assert profile.demographics.sex == "Female"
    assert profile.demographics.age_years == 42
    assert profile.conditions[0].label == "breast cancer"
    assert profile.phenotypes[0].label == "Seizure"
    assert profile.medications[0].label == "trastuzumab"


def test_fhir_bundle_importer_and_exporter(tmp_path):
    bundle = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {"resource": {"resourceType": "Patient", "id": "fhir-p1", "gender": "female"}},
            {
                "resource": {
                    "resourceType": "Condition",
                    "id": "c1",
                    "code": {
                        "coding": [
                            {
                                "system": "http://snomed.info/sct",
                                "code": "254637007",
                                "display": "Non-small cell lung cancer",
                            }
                        ]
                    },
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "id": "o1",
                    "code": {"coding": [{"system": "http://loinc.org", "code": "718-7"}]},
                    "valueQuantity": {"value": 12.5, "unit": "g/dL"},
                }
            },
        ],
    }
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(bundle), encoding="utf-8")

    profile = import_patient_path(path)[0]
    exported = profile_to_fhir_bundle(profile)

    assert detect_patient_input_format(path) == "fhir"
    assert profile.patient_id == "fhir-p1"
    assert profile.conditions[0].normalized_codes[0].vocabulary == "SNOMED"
    assert profile.observations[0].description == "12.5 g/dL"
    assert exported["resourceType"] == "Bundle"


def test_fhir_bundle_groups_resources_by_patient_reference(tmp_path):
    bundle = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {
                "fullUrl": "urn:uuid:patient-1",
                "resource": {"resourceType": "Patient", "id": "p1"},
            },
            {"resource": {"resourceType": "Patient", "id": "p2"}},
            {
                "resource": {
                    "resourceType": "Condition",
                    "id": "c1",
                    "subject": {"reference": "urn:uuid:patient-1"},
                    "code": {"text": "melanoma"},
                }
            },
            {
                "resource": {
                    "resourceType": "Condition",
                    "id": "c2",
                    "subject": {"reference": "Patient/p2"},
                    "code": {"text": "sarcoma"},
                }
            },
        ],
    }
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(bundle), encoding="utf-8")

    profiles = import_patient_path(path)

    by_id = {profile.patient_id: profile for profile in profiles}
    assert sorted(by_id) == ["p1", "p2"]
    assert by_id["p1"].conditions[0].label == "melanoma"
    assert by_id["p2"].conditions[0].label == "sarcoma"


def test_fhir_ndjson_detection_and_import(tmp_path):
    path = tmp_path / "patient.ndjson"
    path.write_text(
        "\n".join(
            [
                json.dumps({"resourceType": "Patient", "id": "ndjson-p1"}),
                json.dumps(
                    {
                        "resourceType": "Procedure",
                        "id": "p1",
                        "code": {"text": "bone marrow biopsy"},
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    profile = import_patient_path(path)[0]

    assert detect_patient_input_format(path) == "fhir-ndjson"
    assert profile.procedures[0].label == "bone marrow biopsy"


def test_fhir_jsonl_detection_and_import(tmp_path):
    path = tmp_path / "patient.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"resourceType": "Patient", "id": "jsonl-p1"}),
                json.dumps(
                    {
                        "resourceType": "Condition",
                        "id": "c1",
                        "subject": {"reference": "Patient/jsonl-p1"},
                        "code": {"text": "glioma"},
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    profile = import_patient_path(path)[0]

    assert detect_patient_input_format(path) == "fhir-ndjson"
    assert profile.conditions[0].label == "glioma"


def test_omop_importer_from_csv_extract(tmp_path):
    omop = tmp_path / "omop"
    omop.mkdir()
    pd.DataFrame(
        [{"person_id": 1, "gender_source_value": "F", "year_of_birth": 1980}]
    ).to_csv(omop / "PERSON.csv", index=False)
    pd.DataFrame(
        [
            {
                "person_id": 1,
                "condition_concept_id": 10,
                "condition_start_date": "2026-01-01",
            }
        ]
    ).to_csv(omop / "CONDITION_OCCURRENCE.csv", index=False)
    pd.DataFrame(
        [
            {
                "concept_id": 10,
                "vocabulary_id": "SNOMED",
                "concept_code": "44054006",
                "concept_name": "Diabetes mellitus",
                "domain_id": "Condition",
            }
        ]
    ).to_csv(omop / "CONCEPT.csv", index=False)

    profiles = import_patient_path(omop)

    assert detect_patient_input_format(omop) == "omop"
    assert len(profiles) == 1
    assert profiles[0].conditions[0].label == "Diabetes mellitus"
    assert profiles[0].conditions[0].normalized_codes[0].code == "44054006"


def test_omop_join_survives_null_person_id_float_promotion(tmp_path):
    # A NULL person_id in a child table promotes the whole column to float64, so
    # person_id 1 serializes as "1.0". The join must still match the PERSON row
    # (previously every child record was silently dropped).
    omop = tmp_path / "omop"
    omop.mkdir()
    pd.DataFrame(
        [{"person_id": 1, "gender_source_value": "F", "year_of_birth": 1980}]
    ).to_csv(omop / "PERSON.csv", index=False)
    pd.DataFrame(
        [
            {"person_id": 1, "condition_concept_id": 10},
            {"person_id": None, "condition_concept_id": 11},
        ]
    ).to_csv(omop / "CONDITION_OCCURRENCE.csv", index=False)
    pd.DataFrame(
        [
            {
                "concept_id": 10,
                "vocabulary_id": "SNOMED",
                "concept_code": "44054006",
                "concept_name": "Diabetes mellitus",
                "domain_id": "Condition",
            }
        ]
    ).to_csv(omop / "CONCEPT.csv", index=False)

    profiles = import_patient_path(omop)

    assert len(profiles) == 1
    assert [c.label for c in profiles[0].conditions] == ["Diabetes mellitus"]


def test_omop_join_uses_raw_person_id_not_sanitized_profile_id(tmp_path):
    # The profile id is sanitized ("pat 01" -> "pat-01"), but child-table joins
    # must key off the raw person_id, not the sanitized profile id.
    omop = tmp_path / "omop"
    omop.mkdir()
    pd.DataFrame(
        [{"person_id": "pat 01", "gender_source_value": "M", "year_of_birth": 1970}]
    ).to_csv(omop / "PERSON.csv", index=False)
    pd.DataFrame(
        [{"person_id": "pat 01", "condition_concept_id": 10}]
    ).to_csv(omop / "CONDITION_OCCURRENCE.csv", index=False)
    pd.DataFrame(
        [
            {
                "concept_id": 10,
                "vocabulary_id": "SNOMED",
                "concept_code": "44054006",
                "concept_name": "Diabetes mellitus",
                "domain_id": "Condition",
            }
        ]
    ).to_csv(omop / "CONCEPT.csv", index=False)

    profiles = import_patient_path(omop)

    assert len(profiles) == 1
    assert profiles[0].patient_id == "pat-01"
    assert [c.label for c in profiles[0].conditions] == ["Diabetes mellitus"]


def test_narrative_and_phenopacket_export_are_deterministic(tmp_path):
    packet = {
        "id": "patient-export",
        "metaData": {},
        "subject": {},
        "diseases": [{"term": {"id": "MONDO:0004992", "label": "cancer"}}],
    }
    path = tmp_path / "patient.json"
    path.write_text(json.dumps(packet), encoding="utf-8")
    profile = import_patient_path(path)[0]

    narrative = render_patient_narrative(profile)
    packet_out = profile_to_phenopacket(profile)

    assert narrative[0].startswith("Diagnoses:")
    assert packet_out["id"] == "patient-export"
    assert packet_out["diseases"][0]["term"]["label"] == "cancer"


def test_omop_note_nlp_negation_recognizes_no(tmp_path):
    """OMOP note_nlp.term_exists 'N'/'No'/'0' means the term is negated (e.g.
    'no metastasis'); only literal 'false' was honored before (audit P0)."""
    from trialmatchai.interop.importers.omop import import_omop_extract

    omop = tmp_path / "omop"
    omop.mkdir()
    pd.DataFrame([{"person_id": 1, "gender_source_value": "F", "year_of_birth": 1980}]).to_csv(
        omop / "PERSON.csv", index=False
    )
    pd.DataFrame(
        [
            {"person_id": 1, "note_nlp_concept_id": 10, "term_exists": "N", "snippet": "no metastasis"},
            {"person_id": 1, "note_nlp_concept_id": 11, "term_exists": "Y", "snippet": "diabetes present"},
        ]
    ).to_csv(omop / "NOTE_NLP.csv", index=False)
    pd.DataFrame(
        [
            {"concept_id": 10, "vocabulary_id": "SNOMED", "concept_code": "1", "concept_name": "Metastasis", "domain_id": "Condition"},
            {"concept_id": 11, "vocabulary_id": "SNOMED", "concept_code": "2", "concept_name": "Diabetes", "domain_id": "Condition"},
        ]
    ).to_csv(omop / "CONCEPT.csv", index=False)

    profile = import_omop_extract(omop)[0]
    by_label = {c.label: c for c in profile.conditions}
    assert by_label["Metastasis"].negated is True
    assert by_label["Diabetes"].negated is False


def test_omop_import_isolates_unreadable_table(tmp_path):
    """A single unreadable OMOP table must not abort the whole import in
    non-strict mode, but must raise in strict mode (audit P0)."""
    import pytest

    from trialmatchai.interop.importers.omop import import_omop_extract

    omop = tmp_path / "omop"
    omop.mkdir()
    pd.DataFrame([{"person_id": 1, "gender_source_value": "F", "year_of_birth": 1980}]).to_csv(
        omop / "PERSON.csv", index=False
    )
    (omop / "EXTRA.parquet").write_bytes(b"not a real parquet file")

    profiles = import_omop_extract(omop)  # non-strict: person imports, bad table skipped
    assert len(profiles) == 1
    with pytest.raises(Exception):  # strict: malformed table aborts
        import_omop_extract(omop, strict=True)


def test_phenopacket_import_isolates_malformed_section(tmp_path):
    """A malformed phenotypicFeatures item must not abort the whole import in
    non-strict mode, but must raise in strict mode (audit P0)."""
    import pytest

    from trialmatchai.interop.importers.phenopacket import import_phenopacket

    packet = {
        "id": "P1",
        "subject": {"sex": "FEMALE"},
        "phenotypicFeatures": ["should be an object, not a string"],
        "diseases": [{"term": {"id": "DOID:1612", "label": "breast cancer"}}],
    }
    path = tmp_path / "p.json"
    path.write_text(json.dumps(packet), encoding="utf-8")

    profile = import_phenopacket(path)  # non-strict: bad section skipped, rest imports
    assert profile.patient_id == "P1"
    assert any(c.label == "breast cancer" for c in profile.conditions)
    with pytest.raises(Exception):
        import_phenopacket(path, strict=True)
