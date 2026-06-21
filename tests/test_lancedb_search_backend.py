from __future__ import annotations

import pytest

from Matcher.search import LanceDBSearchBackend


pytest.importorskip("lancedb")


def test_lancedb_backend_indexes_and_searches_trials_and_criteria(tmp_path):
    backend = LanceDBSearchBackend(
        tmp_path / "search",
        trials_table="trials",
        criteria_table="criteria",
        candidate_limit=25,
    )

    backend.index_trials(
        [
            {
                "nct_id": "N1",
                "condition": "lung cancer",
                "brief_title": "Targeted therapy for lung carcinoma",
                "eligibility_criteria": "Adults with lung cancer",
                "condition_vector": [1.0, 0.0],
                "eligibility_criteria_vector": [1.0, 0.0],
                "gender": "All",
                "overall_status": "Recruiting",
            },
            {
                "nct_id": "N2",
                "condition": "diabetes mellitus",
                "brief_title": "Diabetes prevention",
                "eligibility_criteria": "Adults with diabetes",
                "condition_vector": [0.0, 1.0],
                "eligibility_criteria_vector": [0.0, 1.0],
                "gender": "All",
                "overall_status": "Recruiting",
            },
        ]
    )
    backend.index_criteria(
        [
            {
                "criteria_id": "C1",
                "nct_id": "N1",
                "criterion": "Confirmed malignant neoplasm",
                "criterion_vector": [1.0, 0.0],
                "entities": [{"text": "malignant neoplasm", "synonyms": ["cancer"]}],
                "eligibility_type": "Inclusion Criteria",
            }
        ]
    )

    issues = backend.health(require_tables=True)
    trials, scores = backend.search_trials(
        primary_terms=["lung cancer"],
        embeddings={"lung cancer": [1.0, 0.0]},
        sex="ALL",
        overall_status="Recruiting",
        search_mode="hybrid",
        vector_score_threshold=0.0,
    )
    criteria_hits = backend.search_criteria(
        query="cancer",
        nct_ids=["N1"],
        search_mode="bm25",
        use_entity_synonyms=True,
    )

    assert issues == []
    assert trials[0]["nct_id"] == "N1"
    assert scores[0] > 0
    assert criteria_hits[0]["_source"]["criteria_id"] == "C1"
