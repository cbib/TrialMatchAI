from trialmatchai.matching.retrieval.trial_retrieval import ClinicalTrialSearch
from trialmatchai.search import InMemorySearchBackend


def test_search_trials_bm25_returns_hits():
    backend = InMemorySearchBackend(
        trials=[
            {
                "nct_id": "N1",
                "condition": "lung cancer",
                "brief_title": "Lung carcinoma treatment",
                "eligibility_criteria": "Adults with lung cancer",
                "minimum_age": 18,
                "maximum_age": 80,
                "gender": "All",
                "overall_status": "Recruiting",
            },
            {
                "nct_id": "N2",
                "condition": "diabetes mellitus",
                "brief_title": "Diabetes lifestyle trial",
                "eligibility_criteria": "Adults with diabetes",
                "overall_status": "Recruiting",
            },
        ]
    )
    search = ClinicalTrialSearch(
        search_backend=backend,
        embedder=None,
        bio_med_ner=None,
    )
    trials, scores = search.search_trials(
        condition="lung cancer",
        age_input="all",
        sex="ALL",
        size=5,
        synonyms=["lung carcinoma"],
        search_mode="bm25",
    )
    assert len(trials) == 1
    assert trials[0]["nct_id"] == "N1"
    assert scores[0] > 0
