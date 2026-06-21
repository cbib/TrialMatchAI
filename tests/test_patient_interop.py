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
    assert summary["expanded_sentences"]


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
