from Matcher.pipeline.trial_search.second_level_search import SecondStageRetriever
from Matcher.search import InMemorySearchBackend


def test_score_criteria_without_llm_weights():
    retriever = SecondStageRetriever(
        search_backend=InMemorySearchBackend(),
        llm_reranker=None,
        embedder=None,
        inclusion_weight=1.0,
        exclusion_weight=0.25,
    )
    criteria = [
        {"_score": 2.0, "_source": {"eligibility_type": "Inclusion Criteria"}},
        {"_score": 1.0, "_source": {"eligibility_type": "Exclusion Criteria"}},
    ]
    scored = retriever.score_criteria_without_llm(criteria)
    assert scored[0]["llm_score"] == 1.0
    assert scored[1]["llm_score"] == 0.125


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
