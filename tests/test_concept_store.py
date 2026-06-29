"""P2: exercise the PRODUCTION concept-retrieval path (LanceDBConceptStore + the
RRF merge + the SQL filter), which the suite previously tested only through the
lexical-only InMemoryConceptStore stand-in (test/codebase audit)."""

from __future__ import annotations

from trialmatchai.entities.linker import (
    LanceDBConceptStore,
    _lancedb_filter,
    _rrf_merge,
    _sql_escape,
)
from trialmatchai.entities.types import ConceptCandidate


def _candidate(concept_id, vocab="", code="", name="x", domain=""):
    return ConceptCandidate(
        concept_id=concept_id,
        vocabulary_id=vocab,
        concept_code=code,
        concept_name=name,
        domain_id=domain,
    )


def test_rrf_merge_keeps_distinct_cui_less_candidates():
    # Two distinct concepts with empty vocab/code share the constant normalized_id
    # "CUI-less"; the old dedup key collapsed them to one. Now keyed on concept_id.
    a = _candidate("C1", name="alpha")
    b = _candidate("C2", name="beta")
    merged = _rrf_merge([a, b], [], limit=10)
    assert {c.concept_id for c in merged} == {"C1", "C2"}


def test_rrf_merge_combines_same_concept_across_channels():
    fts = _candidate("SNOMED:1", vocab="SNOMED", code="1", name="diabetes", domain="Condition")
    vec = _candidate("SNOMED:1", vocab="SNOMED", code="1", name="diabetes", domain="Condition")
    merged = _rrf_merge([fts], [vec], limit=10)
    assert len(merged) == 1  # same concept_id -> one combined result
    assert set(merged[0].source_scores) == {"fts", "vector"}


def test_lancedb_filter_and_sql_escape():
    assert _lancedb_filter(["DOID", "SNOMED"], ["Disease"]) == (
        "vocabulary_id IN ('DOID', 'SNOMED') AND domain_id IN ('Disease')"
    )
    assert _lancedb_filter(["DOID"], []) == "vocabulary_id IN ('DOID')"
    assert _lancedb_filter([], []) == ""
    assert _sql_escape("O'Brien") == "O''Brien"  # quote is escaped, not injected


def test_lancedb_concept_store_search_applies_vocabulary_filter(tmp_path):
    from trialmatchai.entities.builder import write_lancedb_table

    rows = [
        {
            "concept_id": "DOID:1", "vocabulary_id": "DOID", "concept_code": "1",
            "concept_name": "diabetes mellitus", "domain_id": "Disease",
            "concept_class_id": "", "standard_concept": "", "synonyms": ["diabetes"],
            "fts_text": "diabetes mellitus diabetes",
        },
        {
            "concept_id": "RxNorm:1", "vocabulary_id": "RxNorm", "concept_code": "1",
            "concept_name": "metformin", "domain_id": "Drug",
            "concept_class_id": "", "standard_concept": "", "synonyms": [],
            "fts_text": "metformin",
        },
    ]
    db = str(tmp_path / "concepts")
    write_lancedb_table(rows, db_path=db, table_name="concepts", embeddings=[[1.0, 0, 0, 0], [0, 1.0, 0, 0]])

    class FakeEmbedder:
        def embed_text(self, text):
            return [1.0, 0, 0, 0]  # nearest the DOID row

    store = LanceDBConceptStore(db, table_name="concepts", embedder=FakeEmbedder())

    # The disease vocabulary is searched and returns the disease concept.
    hits = store.search("diabetes", vocabularies=["DOID"], domain_hints=["Disease"], limit=5)
    assert hits and hits[0].concept_id == "DOID:1"
    # Filtering to a different vocabulary excludes it entirely.
    other = store.search("diabetes", vocabularies=["RxNorm"], domain_hints=["Drug"], limit=5)
    assert all(h.concept_id != "DOID:1" for h in other)
