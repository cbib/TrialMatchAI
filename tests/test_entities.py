from __future__ import annotations

import csv
from pathlib import Path

from trialmatchai.entities.annotator import SchemaEntityAnnotator
from trialmatchai.entities.builder import (
    build_dictionary_rows,
    build_omop_concept_rows,
    write_lancedb_table,
)
from trialmatchai.entities.linker import ConceptLinker, InMemoryConceptStore
from trialmatchai.entities.recognizers import (
    RegexSchemaRecognizer,
    _parse_model_entities,
    resolve_overlaps,
)
from trialmatchai.entities.schemas import load_entity_schemas
from trialmatchai.entities.types import ConceptCandidate, EntityAnnotation, NO_ENTITY_ID

ROOT = Path(__file__).resolve().parents[1]


def test_default_schema_validates_vocab_routing():
    schemas = load_entity_schemas()
    by_id = {schema.id: schema for schema in schemas}

    assert by_id["disease"].target_vocabularies == ("SNOMED", "ICD10", "ICD10CM")
    assert by_id["laboratory_test"].target_vocabularies == ("LOINC",)
    assert by_id["medication"].target_vocabularies == ("RxNorm", "ATC")
    assert by_id["disease"].query_expansion is True


def test_regex_backend_returns_current_output_shape():
    schemas = [schema for schema in load_entity_schemas() if schema.id == "disease"]
    annotator = SchemaEntityAnnotator(RegexSchemaRecognizer(), schemas)

    result = annotator.annotate_texts_in_parallel(["metastatic cancer"], max_workers=1)

    assert result[0][0]["entity_group"] == "disease"
    assert result[0][0]["text"] == "cancer"
    assert result[0][0]["normalized_id"] == [NO_ENTITY_ID]
    assert "concept_candidates" in result[0][0]


def test_gliner2_entity_mapping_response_parses_to_annotations():
    schemas = [
        schema for schema in load_entity_schemas() if schema.id in {"disease", "gene"}
    ]
    label_map = {
        label.casefold(): schema
        for schema in schemas
        for label in schema.recognizer_labels
    }
    text = "Patient has lung cancer with EGFR mutation."

    annotations = _parse_model_entities(
        {
            "entities": {
                "disease": [
                    {
                        "text": "lung cancer",
                        "start": 12,
                        "end": 23,
                        "confidence": 0.99,
                    }
                ],
                "gene": [{"text": "EGFR", "start": 29, "end": 33, "confidence": 0.99}],
            }
        },
        text,
        label_map,
    )

    assert [annotation.text for annotation in annotations] == ["lung cancer", "EGFR"]


def test_overlap_resolution_keeps_higher_confidence_span():
    annotations = [
        EntityAnnotation("disease", "lung cancer", 0, 11, 0.91, schema_id="disease"),
        EntityAnnotation("disease", "cancer", 5, 11, 0.95, schema_id="disease"),
    ]

    resolved = resolve_overlaps(annotations)

    assert len(resolved) == 1
    assert resolved[0].text == "cancer"


def test_concept_linker_accepts_rejects_and_marks_ambiguous():
    schemas = [schema for schema in load_entity_schemas() if schema.id == "disease"]
    store = InMemoryConceptStore(
        [
            ConceptCandidate(
                concept_id="1",
                vocabulary_id="SNOMED",
                concept_code="363346000",
                concept_name="Malignant neoplastic disease",
                domain_id="Condition",
                synonyms=("cancer", "malignancy"),
            ),
            ConceptCandidate(
                concept_id="2",
                vocabulary_id="SNOMED",
                concept_code="73211009",
                concept_name="Diabetes mellitus",
                domain_id="Condition",
            ),
        ]
    )
    linker = ConceptLinker(store, schemas, accept_threshold=0.8, reject_threshold=0.3)

    accepted = linker.link_annotation(
        EntityAnnotation("disease", "cancer", 0, 6, 0.95, schema_id="disease")
    )
    ambiguous = linker.link_annotation(
        EntityAnnotation(
            "disease",
            "neoplastic disorder",
            0,
            19,
            0.95,
            schema_id="disease",
        )
    )
    rejected = linker.link_annotation(
        EntityAnnotation("disease", "unrelated words", 0, 15, 0.95, schema_id="disease")
    )

    assert accepted.linker_status == "accepted"
    assert accepted.normalized_id == ("SNOMED:363346000",)
    assert "malignancy" in accepted.synonyms
    assert ambiguous.linker_status == "ambiguous"
    assert ambiguous.normalized_id == (NO_ENTITY_ID,)
    assert rejected.linker_status == "rejected"


def test_concept_builders_import_omop_and_dictionary_rows(tmp_path):
    concept_csv = tmp_path / "CONCEPT.csv"
    synonym_csv = tmp_path / "CONCEPT_SYNONYM.csv"
    with concept_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "concept_id",
                "concept_name",
                "domain_id",
                "vocabulary_id",
                "concept_class_id",
                "standard_concept",
                "concept_code",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "concept_id": "1",
                "concept_name": "Hemoglobin measurement",
                "domain_id": "Measurement",
                "vocabulary_id": "LOINC",
                "concept_class_id": "Lab Test",
                "standard_concept": "S",
                "concept_code": "718-7",
            }
        )
    with synonym_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["concept_id", "concept_synonym_name"],
        )
        writer.writeheader()
        writer.writerow({"concept_id": "1", "concept_synonym_name": "Hgb"})

    rows = build_omop_concept_rows(concept_csv, synonym_csv, vocabularies=("LOINC",))
    dictionary = tmp_path / "dict_Gene.txt"
    dictionary.write_text("EntrezGene:1956||EGFR|ERBB1\n")

    dictionary_rows = build_dictionary_rows(
        dictionary,
        vocabulary_id="EntrezGene",
        domain_id="Gene",
    )

    assert rows[0]["concept_code"] == "718-7"
    assert rows[0]["synonyms"] == ["Hgb"]
    assert dictionary_rows[0]["concept_code"] == "1956"
    assert dictionary_rows[0]["synonyms"] == ["EGFR", "ERBB1"]


def test_concept_table_no_recreate_appends_rows(tmp_path):
    db_path = tmp_path / "concepts"
    first = [
        {
            "concept_id": "SNOMED:1",
            "vocabulary_id": "SNOMED",
            "concept_code": "1",
            "concept_name": "Melanoma",
            "domain_id": "Condition",
            "concept_class_id": "Clinical Finding",
            "standard_concept": "S",
            "synonyms": ["malignant melanoma"],
            "fts_text": "Melanoma malignant melanoma",
        }
    ]
    second = [
        {
            "concept_id": "SNOMED:2",
            "vocabulary_id": "SNOMED",
            "concept_code": "2",
            "concept_name": "Sarcoma",
            "domain_id": "Condition",
            "concept_class_id": "Clinical Finding",
            "standard_concept": "S",
            "synonyms": ["soft tissue sarcoma"],
            "fts_text": "Sarcoma soft tissue sarcoma",
        }
    ]

    write_lancedb_table(first, db_path=db_path, table_name="concepts", recreate=True)
    write_lancedb_table(second, db_path=db_path, table_name="concepts", recreate=False)

    import lancedb

    table = lancedb.connect(str(db_path)).open_table("concepts")
    codes = {row["concept_code"] for row in table.to_arrow().to_pylist()}
    assert codes == {"1", "2"}


def test_runtime_replacement_has_no_old_daemon_references():
    assert not (ROOT / "source/Parser").exists()
    assert not (ROOT / "src/Matcher").exists()
    runtime_files = sorted((ROOT / "src/trialmatchai").rglob("*.py"))
    forbidden = [
        "18888",
        "18892",
        "18894",
        "18783",
        "BioMedNER",
        "GNormPlus",
        "disease_normalizer_21.jar",
        "java -Xmx",
        "import socket",
    ]
    for path in runtime_files:
        content = path.read_text()
        for term in forbidden:
            assert term not in content
