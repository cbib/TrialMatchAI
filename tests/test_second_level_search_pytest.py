from trialmatchai.constraints.models import (
    Constraint,
    ConstraintSet,
    PatientConstraintContext,
    PatientConstraintFact,
)
from trialmatchai.main import _fuse_shortlist_scores
from trialmatchai.matching.retrieval.criteria_retrieval import SecondStageRetriever
from trialmatchai.search import InMemorySearchBackend


def test_score_criteria_without_llm_weights():
    retriever = SecondStageRetriever(
        search_backend=InMemorySearchBackend(),
        llm_reranker=None,
        embedder=None,
        inclusion_weight=1.0,
        exclusion_weight=0.25,
    )
    # Use the real stored short-form values ("inclusion"/"exclusion"/"unknown"), not the
    # full-phrase headers — the old full-phrase fixtures masked the weighting never firing.
    criteria = [
        {"_score": 2.0, "_source": {"eligibility_type": "inclusion"}},
        {"_score": 1.0, "_source": {"eligibility_type": "exclusion"}},
        {"_score": 2.0, "_source": {"eligibility_type": "unknown"}},
    ]
    scored = retriever.score_criteria_without_llm(criteria)
    assert scored[0]["llm_score"] == 1.0     # inclusion: (2/2) * 1.0
    assert scored[1]["llm_score"] == 0.125   # exclusion: (1/2) * 0.25
    assert scored[2]["llm_score"] == 1.0     # unknown: (2/2), no downweight (ES-era fall-through)


def test_rerank_criteria_applies_exclusion_weight():
    class _StubReranker:
        def rank_pairs(self, pairs):
            return [1.0 for _ in pairs]

    retriever = SecondStageRetriever(
        search_backend=InMemorySearchBackend(),
        llm_reranker=_StubReranker(),
        embedder=None,
        inclusion_weight=1.0,
        exclusion_weight=0.25,
    )
    criteria = [
        {"query": "q", "_source": {"criterion": "c", "eligibility_type": "inclusion"}},
        {"query": "q", "_source": {"criterion": "c", "eligibility_type": "exclusion"}},
        {"query": "q", "_source": {"criterion": "c", "eligibility_type": "unknown"}},
    ]
    scored = retriever.rerank_criteria(criteria)
    assert scored[0]["llm_score"] == 1.0     # inclusion
    assert scored[1]["llm_score"] == 0.25    # exclusion downweighted
    assert scored[2]["llm_score"] == 1.0     # unknown untouched


def test_aggregate_to_trials_weighted():
    retriever = SecondStageRetriever(
        search_backend=InMemorySearchBackend(),
        llm_reranker=None,
        embedder=None,
    )
    criteria = [
        {"llm_score": 0.6, "_source": {"nct_id": "N1"}},
        {"llm_score": 0.5, "_source": {"nct_id": "N1"}},
        {"llm_score": 0.9, "_source": {"nct_id": "N2"}},
    ]
    trials = retriever.aggregate_to_trials(criteria, threshold=0.5, method="weighted")
    assert trials[0]["nct_id"] == "N2"


def test_shortlist_fusion_rrf_keeps_first_level_trial_the_reranker_dropped():
    """A trial retrieval ranked high but the reranker never scored (no criterion cleared the
    aggregation threshold, so it is absent from second_level_results) must still receive a
    fused score under RRF -- retrieval keeps it a floor instead of it being silently evicted."""
    nct_ids = ["N1", "N2", "N3", "N4"]  # first-level order
    second_level_results = [  # reranker only produced N2 and N3
        {"nct_id": "N2", "score": 9.0},
        {"nct_id": "N3", "score": 8.0},
    ]
    fused = _fuse_shortlist_scores(
        nct_ids=nct_ids,
        second_level_results=second_level_results,
        first_level_scores={"N1": 0.9, "N2": 0.4, "N3": 0.3, "N4": 0.2},
        search_config={"shortlist_fusion": "rrf", "shortlist_rrf_k": 60},
    )
    # Every first-level trial is present (floor preserved), including the reranker-absent N1.
    assert set(fused) == {"N1", "N2", "N3", "N4"}
    assert fused["N1"] > 0.0
    # N1 (first-level #1) outranks N4 (first-level #4) despite neither being reranked.
    assert fused["N1"] > fused["N4"]


def test_shortlist_fusion_rrf_rewards_agreement_across_both_rankings():
    nct_ids = ["N1", "N2", "N3"]
    second_level_results = [
        {"nct_id": "N1", "score": 5.0},
        {"nct_id": "N2", "score": 4.0},
        {"nct_id": "N3", "score": 3.0},
    ]
    fused = _fuse_shortlist_scores(
        nct_ids=nct_ids,
        second_level_results=second_level_results,
        first_level_scores={"N1": 0.1, "N2": 0.1, "N3": 0.1},
        search_config={"shortlist_fusion": "rrf"},
    )
    # Both rankings agree on N1 > N2 > N3, so the fused order matches.
    assert [nct for nct, _ in sorted(fused.items(), key=lambda x: x[1], reverse=True)] == [
        "N1",
        "N2",
        "N3",
    ]


def test_shortlist_fusion_second_level_weight_zero_follows_first_level_order():
    fused = _fuse_shortlist_scores(
        nct_ids=["A", "B", "C"],
        second_level_results=[{"nct_id": "C", "score": 99.0}],  # reranker loves C
        first_level_scores={},
        search_config={
            "shortlist_fusion": "rrf",
            "shortlist_second_level_weight": 0.0,
        },
    )
    # With the reranker weight zeroed, order is purely first-level: A > B > C.
    assert fused["A"] > fused["B"] > fused["C"]


def test_shortlist_fusion_score_sum_is_legacy_second_level_only():
    """score_sum reproduces the earlier behaviour: only reranked trials, raw score add,
    no floor for first-level-only trials."""
    fused = _fuse_shortlist_scores(
        nct_ids=["N1", "N2", "N3"],
        second_level_results=[
            {"nct_id": "N2", "score": 2.0},
            {"nct_id": "N3", "score": 1.0},
        ],
        first_level_scores={"N1": 0.9, "N2": 0.5, "N3": 0.5},
        search_config={"shortlist_fusion": "score_sum"},
    )
    assert set(fused) == {"N2", "N3"}  # N1 (first-level-only) is NOT carried
    assert fused["N2"] == 2.5
    assert fused["N3"] == 1.5


def test_retrieve_criteria_uses_entity_synonyms():
    backend = InMemorySearchBackend(
        criteria=[
            {
                "criteria_id": "C1",
                "nct_id": "N1",
                "criterion": "Documented malignant neoplasm",
                "eligibility_type": "Inclusion Criteria",
                "entities": [{"text": "malignant neoplasm", "synonyms": ["cancer"]}],
            },
            {
                "criteria_id": "C2",
                "nct_id": "N2",
                "criterion": "Documented diabetes mellitus",
                "eligibility_type": "Inclusion Criteria",
                "entities": [],
            },
        ]
    )
    retriever = SecondStageRetriever(
        search_backend=backend,
        llm_reranker=None,
        embedder=None,
        entity_annotator=object(),
        search_mode="bm25",
    )

    hits = retriever.retrieve_criteria(["N1"], ["cancer"])

    assert hits["cancer"][0]["_source"]["criteria_id"] == "C1"


def test_constraint_adjustments_penalize_without_hard_exclusion():
    backend = InMemorySearchBackend(criteria=_constraint_test_criteria())
    retriever = SecondStageRetriever(
        search_backend=backend,
        llm_reranker=None,
        embedder=None,
        inclusion_weight=1.0,
        exclusion_weight=1.0,
        search_mode="bm25",
    )

    trials = retriever.retrieve_and_rank(
        ["lung cancer"],
        ["N1", "N2"],
        top_n=2,
        patient_context=_constraint_context(),
        constraints_config={"enabled": True, "score_weight": 1.0},
    )

    scores = {trial["nct_id"]: trial["score"] for trial in trials}
    assert trials[0]["nct_id"] == "N1"
    assert "N2" in scores
    assert scores["N2"] == 0
    assert any(
        evaluation.nct_id == "N2" and evaluation.violated_count == 1
        for evaluation in retriever.last_constraint_evaluations
    )


def test_constraints_disabled_preserves_ranking_behavior():
    backend = InMemorySearchBackend(criteria=_constraint_test_criteria())
    retriever = SecondStageRetriever(
        search_backend=backend,
        llm_reranker=None,
        embedder=None,
        inclusion_weight=1.0,
        exclusion_weight=1.0,
        search_mode="bm25",
    )

    disabled = retriever.retrieve_and_rank(
        ["lung cancer"],
        ["N1", "N2"],
        top_n=2,
        patient_context=_constraint_context(),
        constraints_config={"enabled": False, "score_weight": 1.0},
    )
    assert retriever.last_constraint_evaluations == []

    baseline = retriever.retrieve_and_rank(["lung cancer"], ["N1", "N2"], top_n=2)

    assert disabled == baseline


def _constraint_test_criteria():
    return [
        {
            "criteria_id": "C1",
            "nct_id": "N1",
            "criterion": "Adults with lung cancer.",
            "eligibility_type": "Inclusion Criteria",
            "entities": [],
            "constraints": _constraint_payload(
                "N1",
                "C1",
                "inclusion",
                [Constraint(kind="condition", label="lung cancer")],
            ),
        },
        {
            "criteria_id": "C2",
            "nct_id": "N2",
            "criterion": "Adults with lung cancer. Prior osimertinib is excluded.",
            "eligibility_type": "Exclusion Criteria",
            "entities": [],
            "constraints": _constraint_payload(
                "N2",
                "C2",
                "exclusion",
                [
                    Constraint(
                        kind="medication",
                        label="osimertinib",
                        comparator="prior",
                    )
                ],
            ),
        },
    ]


def _constraint_payload(
    nct_id: str,
    criteria_id: str,
    polarity: str,
    constraints: list[Constraint],
):
    return ConstraintSet(
        nct_id=nct_id,
        criteria_id=criteria_id,
        polarity=polarity,
        source_text="constraint fixture",
        constraints=constraints,
    ).model_dump(mode="json")


def _constraint_context() -> PatientConstraintContext:
    return PatientConstraintContext(
        patient_id="P1",
        facts=[
            PatientConstraintFact(
                kind="condition",
                label="lung cancer",
                evidence_text="Patient has lung cancer.",
            ),
            PatientConstraintFact(
                kind="medication",
                label="osimertinib",
                temporality="prior",
                evidence_text="Prior osimertinib documented.",
            ),
        ],
    )
