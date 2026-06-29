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

    assert by_id["disease"].target_vocabularies == (
        "SNOMED", "ICD10", "ICD10CM", "MeSH", "OMOP Extension", "DOID",
    )
    assert by_id["laboratory_test"].target_vocabularies == ("LOINC",)
    assert by_id["medication"].target_vocabularies == (
        "RxNorm", "RxNorm Extension", "ATC", "OMOP Extension", "ChEBI",
    )
    assert by_id["disease"].query_expansion is True
    # the variant schema links mutation entities to the cancer-genetics vocabularies
    assert by_id["variant"].target_vocabularies == ("CIViC", "ClinVar", "OncoKB")
    assert by_id["variant"].is_linkable


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
            delimiter="\t",
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
            delimiter="\t",
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


def test_omop_concept_rows_parse_tab_separated_athena(tmp_path):
    """Athena CONCEPT/CONCEPT_SYNONYM files are TAB-separated: ingestion must
    parse them, keep only target vocabularies, skip deprecated rows, and merge
    synonyms."""
    concept = tmp_path / "CONCEPT.csv"
    concept.write_text(
        "concept_id\tconcept_name\tdomain_id\tvocabulary_id\tconcept_class_id\t"
        "standard_concept\tconcept_code\tvalid_start_date\tvalid_end_date\tinvalid_reason\n"
        "201826\tType 2 diabetes mellitus\tCondition\tSNOMED\tDisorder\tS\t44054006\t20020101\t20991231\t\n"
        "1503297\tmetformin\tDrug\tRxNorm\tIngredient\tS\t6809\t20020101\t20991231\t\n"
        "999999\tDeprecated thing\tCondition\tSNOMED\tDisorder\t\t000\t20020101\t20051231\tD\n"
        "888888\tSome LOINC thing\tMeasurement\tLOINC\tLab Test\tS\t1234-5\t20020101\t20991231\t\n",
        encoding="utf-8",
    )
    synonym = tmp_path / "CONCEPT_SYNONYM.csv"
    synonym.write_text(
        "concept_id\tconcept_synonym_name\tlanguage_concept_id\n"
        "201826\tT2DM\t4180186\n"
        "201826\tdiabetes mellitus type 2\t4180186\n",
        encoding="utf-8",
    )
    rows = build_omop_concept_rows(concept, synonym, vocabularies=("SNOMED", "RxNorm"))
    by_code = {r["concept_code"]: r for r in rows}
    assert set(by_code) == {"44054006", "6809"}  # only target vocabularies
    assert "000" not in by_code  # deprecated (invalid_reason) skipped
    assert "1234-5" not in by_code  # non-target vocabulary (LOINC) skipped
    assert "T2DM" in by_code["44054006"]["synonyms"]  # synonyms merged from synonym file
