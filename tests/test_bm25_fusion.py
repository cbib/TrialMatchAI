"""Tests for the best-of-both BM25 + lexical-heuristic text fusion.

The existing search tests exercise InMemorySearchBackend, which carries no engine
``_score`` and therefore only covers the heuristic fallback path. These tests
target the fusion path directly: the pure helpers (`_fuse_text`, `_bm25_norms`,
`_merge_candidate`) and the ranker behaviour when an engine BM25 score is present.
"""

from __future__ import annotations

import pytest

from trialmatchai.search.lancedb_backend import (
    BM25_FUSION_WEIGHT,
    _bm25_norms,
    _fuse_text,
    _merge_candidate,
    _rank_criteria_rows,
    _rank_trial_rows,
)


# --- _fuse_text -------------------------------------------------------------


def test_fuse_text_falls_back_to_heuristic_when_no_bm25():
    assert _fuse_text(None, 0.42) == 0.42


def test_fuse_text_blends_at_configured_weight():
    assert _fuse_text(1.0, 0.0) == pytest.approx(BM25_FUSION_WEIGHT)
    assert _fuse_text(0.0, 1.0) == pytest.approx(1.0 - BM25_FUSION_WEIGHT)
    assert _fuse_text(0.5, 0.5) == pytest.approx(0.5)
    assert _fuse_text(1.0, 1.0) == pytest.approx(1.0)


def test_fuse_text_clamps_to_unit_interval():
    # Inputs are normally [0, 1]; the clamp is defensive.
    assert _fuse_text(2.0, 2.0) == 1.0
    assert _fuse_text(-1.0, -1.0) == 0.0


# --- _bm25_norms ------------------------------------------------------------


def test_bm25_norms_empty_when_no_engine_score():
    assert _bm25_norms([{"nct_id": "A"}, {"nct_id": "B"}]) == {}


def test_bm25_norms_min_max_normalizes_keyed_by_row():
    rows = [
        {"nct_id": "A", "_score": 2.0},
        {"nct_id": "B", "_score": 4.0},
        {"nct_id": "C", "_score": 3.0},
    ]
    norms = _bm25_norms(rows)
    assert norms == {"A": pytest.approx(0.0), "B": pytest.approx(1.0), "C": pytest.approx(0.5)}


def test_bm25_norms_all_equal_scores_map_to_one():
    norms = _bm25_norms([{"nct_id": "A", "_score": 5.0}, {"nct_id": "B", "_score": 5.0}])
    assert norms == {"A": 1.0, "B": 1.0}


def test_bm25_norms_ignores_non_numeric_and_bool_scores():
    rows = [
        {"nct_id": "A", "_score": 2.0},
        {"nct_id": "B", "_score": True},  # bool must not be treated as a score
        {"nct_id": "C", "_score": "x"},
        {"nct_id": "D", "_score": 4.0},
    ]
    norms = _bm25_norms(rows)
    assert set(norms) == {"A", "D"}
    assert norms["A"] == pytest.approx(0.0)
    assert norms["D"] == pytest.approx(1.0)


# --- _merge_candidate -------------------------------------------------------


def test_merge_candidate_preserves_fts_score_when_vector_row_overwrites():
    rows_by_key: dict = {}
    _merge_candidate(rows_by_key, {"nct_id": "X", "_score": 3.0, "search_text": "a"})
    _merge_candidate(rows_by_key, {"nct_id": "X", "_distance": 0.1, "search_text": "a"})
    merged = rows_by_key["X"]
    assert merged["_score"] == 3.0  # BM25 score survived the overwrite
    assert merged["_distance"] == 0.1  # vector row's payload is present


def test_merge_candidate_vector_only_row_has_no_score():
    rows_by_key: dict = {}
    _merge_candidate(rows_by_key, {"nct_id": "Y", "_distance": 0.2})
    assert "_score" not in rows_by_key["Y"]


# --- ranker behaviour: fusion path vs fallback ------------------------------


def _trial_rank(rows):
    return _rank_trial_rows(
        rows,
        primary_terms=["melanoma"],
        other_terms=[],
        embeddings={},
        age=None,
        sex="ALL",
        overall_status=None,
        pre_selected_nct_ids=None,
        size=10,
        vector_score_threshold=0.0,
        search_mode="bm25",
    )


def test_bm25_breaks_heuristic_ties_in_trial_ranking():
    # Identical text => identical heuristic; only the engine BM25 differs.
    rows = [
        {"nct_id": "LOW", "search_text": "melanoma", "condition": "melanoma", "_score": 1.0},
        {"nct_id": "HIGH", "search_text": "melanoma", "condition": "melanoma", "_score": 5.0},
    ]
    hits = _trial_rank(rows)
    assert [h.source["nct_id"] for h in hits] == ["HIGH", "LOW"]
    assert hits[0].score > hits[1].score


def test_trial_ranking_without_engine_score_stays_pure_heuristic():
    # No _score on any row => fusion must reduce exactly to the heuristic, so
    # text-identical trials keep tying (unchanged legacy behaviour).
    rows = [
        {"nct_id": "A", "search_text": "melanoma", "condition": "melanoma"},
        {"nct_id": "B", "search_text": "melanoma", "condition": "melanoma"},
    ]
    hits = _trial_rank(rows)
    assert len(hits) == 2
    assert hits[0].score == pytest.approx(hits[1].score)


def test_bm25_breaks_heuristic_ties_in_criteria_ranking():
    rows = [
        {"criteria_id": "LOW", "nct_id": "N1", "search_text": "diabetes", "criterion": "diabetes", "_score": 1.0},
        {"criteria_id": "HIGH", "nct_id": "N1", "search_text": "diabetes", "criterion": "diabetes", "_score": 9.0},
    ]
    hits = _rank_criteria_rows(
        rows,
        query="diabetes",
        nct_ids=[],
        query_vector=None,
        size=10,
        search_mode="bm25",
        use_entity_synonyms=False,
        vector_score_threshold=0.0,
    )
    assert [h.source["criteria_id"] for h in hits] == ["HIGH", "LOW"]
    assert hits[0].score > hits[1].score
