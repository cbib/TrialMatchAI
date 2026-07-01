from trialmatchai.constraints.models import (
    Constraint,
    ConstraintSet,
    PatientConstraintContext,
    PatientConstraintFact,
)
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
