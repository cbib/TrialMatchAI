from trialmatchai.matching.retrieval.trial_retrieval import ClinicalTrialSearch
from trialmatchai.search import InMemorySearchBackend, build_criteria_record, build_trial_record


def test_first_level_query_describes_backend_search():
    search = ClinicalTrialSearch(
        search_backend=InMemorySearchBackend(),
        embedder=None,
        entity_annotator=None,
    )
    query = search.create_query(
        synonyms=["lung cancer"],
        embeddings={},
        other_conditions=["smoking"],
    )

    assert query["primary_terms"] == ["lung cancer"]
    assert query["other_terms"] == ["smoking"]
    assert query["embeddings"] == {}
    # Filters are passed to the backend directly, not via the query dict.
    assert set(query) == {"primary_terms", "other_terms", "embeddings"}


def test_build_trial_record_flattens_search_text_and_vector():
    record = build_trial_record(
        {
            "nct_id": "N1",
            "condition": ["Lung cancer", "NSCLC"],
            "brief_title": "Targeted therapy trial",
            "condition_vector": [1.0, 0.0],
            "brief_title_vector": [0.0, 1.0],
        }
    )

    assert "Lung cancer" in record["search_text"]
    assert record["search_vector"] == [0.5, 0.5]


def test_build_criteria_record_flattens_entity_synonyms():
    record = build_criteria_record(
        {
            "criteria_id": "C1",
            "criterion": "Documented malignancy",
            "entities": [
                {
                    "text": "malignancy",
                    "synonyms": ["cancer", "neoplasm"],
                    "concept_candidates": [{"concept_name": "Malignant neoplasm"}],
                }
            ],
        }
    )

    assert "cancer" in record["entity_synonyms_text"]
    assert "Malignant neoplasm" in record["search_text"]
