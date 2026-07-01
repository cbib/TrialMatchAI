from __future__ import annotations

import json

import pytest

from trialmatchai.search import LanceDBSearchBackend
from trialmatchai.search.lancedb_backend import _nct_where


pytest.importorskip("lancedb")


def test_trial_vector_secondary_terms_field_weighted_and_sparsity_independent():
    from trialmatchai.search.lancedb_backend import _trial_vector_score

    q = [0.3, 0.9539392]  # unit vector, cosine 0.3 with the field axis -> uncapped score
    field = [1.0, 0.0]
    emb = {"primary": q, "other": q}
    s_one = _trial_vector_score({"condition_vector": field}, ["primary"], ["other"], emb)
    s_all = _trial_vector_score(
        {k: field for k in ("condition_vector", "brief_title_vector",
                            "brief_summary_vector", "eligibility_criteria_vector")},
        ["primary"], ["other"], emb,
    )
    # the fix: the secondary-term contribution no longer depends on how many fields are populated
    assert abs(s_one - s_all) < 1e-9
    assert 0.0 < s_one < 1.0  # uncapped, so the equality is a real property (not both clamped to 1)


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


def test_scan_rows_fallback_applies_nct_filter(tmp_path):
    # The fallback scan (used when FTS and vector both return nothing) must honor
    # the nct_id filter; otherwise it returns arbitrary rows that may exclude the
    # requested trials entirely.
    backend = LanceDBSearchBackend(
        tmp_path / "search",
        trials_table="trials",
        criteria_table="criteria",
        candidate_limit=25,
    )
    backend.index_criteria(
        [
            {
                "criteria_id": "C1",
                "nct_id": "N1",
                "criterion": "alpha",
                "criterion_vector": [1.0, 0.0],
                "entities": [],
                "eligibility_type": "Inclusion Criteria",
            },
            {
                "criteria_id": "C2",
                "nct_id": "N2",
                "criterion": "beta",
                "criterion_vector": [0.0, 1.0],
                "entities": [],
                "eligibility_type": "Inclusion Criteria",
            },
        ]
    )
    table = backend._open_table("criteria")

    rows = backend._scan_rows(table, where=_nct_where(["N1"]), limit=25)

    assert rows
    assert {row["nct_id"] for row in rows} == {"N1"}


def test_lancedb_backend_serializes_variable_entity_payloads(tmp_path):
    backend = LanceDBSearchBackend(
        tmp_path / "search",
        trials_table="trials",
        criteria_table="criteria",
        candidate_limit=25,
    )

    backend.index_criteria(
        [
            {
                "criteria_id": "C1",
                "nct_id": "N1",
                "criterion": "Confirmed EGFR mutation",
                "criterion_vector": [1.0, 0.0],
                "entities": [
                    {
                        "entity_group": "gene",
                        "text": "EGFR",
                        "start": 10,
                        "end": 14,
                        "score": 0.99,
                        "normalized_id": ["CUI-less"],
                        "synonyms": ["ERBB1"],
                        "concept_candidates": [
                            {
                                "concept_id": "EntrezGene:1956",
                                "concept_name": "epidermal growth factor receptor",
                                "source_scores": {"fts": 1.0},
                            }
                        ],
                        "linker_score": None,
                        "linker_status": "not_linked",
                    }
                ],
                "eligibility_type": "Inclusion Criteria",
            },
            {
                "criteria_id": "C2",
                "nct_id": "N1",
                "criterion": "No active autoimmune disease",
                "criterion_vector": [0.0, 1.0],
                "entities": [
                    {
                        "entity_group": "disease",
                        "text": "autoimmune disease",
                        "start": 10,
                        "end": 28,
                        "score": 0.95,
                    }
                ],
                "eligibility_type": "Exclusion Criteria",
            },
        ]
    )

    hits = backend.search_criteria(
        query="ERBB1",
        nct_ids=["N1"],
        search_mode="bm25",
        use_entity_synonyms=True,
    )

    assert hits[0]["_source"]["criteria_id"] == "C1"
    assert json.loads(hits[0]["_source"]["entities"])[0]["text"] == "EGFR"
