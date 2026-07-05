import json

import pytest

from trialmatchai.interop.models import ClinicalFact, PatientProfile, Provenance
from trialmatchai.main import run_first_level_search
from trialmatchai.matching.retrieval.first_level_planner import (
    FirstLevelQueryPlanner,
    parse_llm_query_expansion,
)
from trialmatchai.matching.retrieval.trial_retrieval import ClinicalTrialSearch
from trialmatchai.search import InMemorySearchBackend


def test_planner_builds_deterministic_channels_and_skips_negated_facts():
    planner = FirstLevelQueryPlanner(entity_annotator=FakeAnnotator())
    profile = PatientProfile(
        patient_id="P1",
        conditions=[
            _fact("condition-1", "condition", "lung cancer"),
            _fact("condition-2", "condition", "asthma", negated=True),
        ],
        genomic_findings=[_fact("gene-1", "genomic_finding", "EGFR mutation")],
        medications=[
            _fact("med-1", "medication", "osimertinib", temporality="prior")
        ],
    )

    plan = planner.build(
        profile=profile,
        matching_summary={
            "main_conditions": ["lung cancer"],
            "other_conditions": [],
            "patient_narrative": ["Patient has EGFR-mutated lung cancer."],
        },
        config={"llm_expansion_enabled": False},
        age=64,
        sex="female",
        overall_status="All",
    )

    assert plan.terms_for("primary_condition") == ["lung cancer"]
    assert "lung carcinoma" in plan.terms_for("concept_synonym")
    assert "solid tumor" in plan.terms_for("broader_disease")
    assert "EGFR mutation" in plan.terms_for("biomarker")
    assert "prior osimertinib" in plan.terms_for("therapy")
    assert "asthma" not in plan.terms_for("primary_condition", "broader_disease")


def test_planner_builds_per_condition_other_condition_channels():
    """Each comorbidity gets its OWN focused first-level channel instead of only leaking into
    the narrative blob; a blended channel over many distinct conditions dilutes to noise, which
    collapsed recall on multi-morbid patients."""
    planner = FirstLevelQueryPlanner(entity_annotator=FakeAnnotator())
    plan = planner.build(
        profile=PatientProfile(patient_id="P2"),
        matching_summary={
            "main_conditions": ["neurogenic bladder"],
            "other_conditions": [
                "Urinary tract infections",
                "Non-Hodgkin's Lymphoma",
                "neurogenic bladder",  # duplicate of the primary term -> must be dropped
            ],
            "patient_narrative": ["x"],
        },
        config={"llm_expansion_enabled": False},
    )
    oc = [c for c in plan.channels if c.kind == "other_condition"]
    assert len(oc) == 2  # one channel per distinct non-primary comorbidity
    assert all(len(c.terms) == 1 for c in oc)  # each is a single focused query, not a blob
    terms = {c.terms[0] for c in oc}
    assert terms == {"Urinary tract infections", "Non-Hodgkin's Lymphoma"}
    assert all(0.0 < c.weight < 1.0 for c in oc)  # below the primary condition


def test_planner_no_other_condition_channel_when_empty():
    planner = FirstLevelQueryPlanner(entity_annotator=FakeAnnotator())
    plan = planner.build(
        profile=PatientProfile(patient_id="P3"),
        matching_summary={
            "main_conditions": ["lung cancer"],
            "other_conditions": [],
            "patient_narrative": ["x"],
        },
        config={"llm_expansion_enabled": False},
    )
    assert [c for c in plan.channels if c.kind == "other_condition"] == []


def test_llm_query_expansion_is_strict_and_capped():
    parsed = parse_llm_query_expansion(
        {
            "primary_queries": ["lung cancer", "lung cancer"],
            "disease_aliases": ["NSCLC"],
            "broader_queries": ["solid tumor"],
            "biomarker_queries": ["EGFR mutation"],
            "treatment_queries": ["osimertinib"],
            "discarded_or_uncertain": ["random drift"],
        },
        max_terms=3,
    )

    assert parsed.primary_queries == ["lung cancer"]
    assert parsed.disease_aliases == ["NSCLC"]
    assert parsed.broader_queries == ["solid tumor"]
    assert parsed.biomarker_queries == []
    assert parsed.treatment_queries == []
    with pytest.raises(Exception):
        parse_llm_query_expansion("{bad json", max_terms=3)
    with pytest.raises(Exception):
        parse_llm_query_expansion({"primary_queries": [], "extra": []}, max_terms=3)


def test_planned_search_fuses_multi_channel_hits_above_single_channel_hits():
    backend = InMemorySearchBackend(
        trials=[
            {
                "nct_id": "N1",
                "condition": "lung cancer",
                "brief_title": "EGFR lung cancer osimertinib trial",
                "brief_summary": "Study for EGFR mutation after prior osimertinib.",
                "eligibility_criteria": "Adults with lung cancer or solid tumor.",
                "overall_status": "Recruiting",
                "gender": "All",
            },
            {
                "nct_id": "N2",
                "condition": "lung cancer",
                "brief_title": "General lung cancer trial",
                "eligibility_criteria": "Adults with lung cancer.",
                "overall_status": "Recruiting",
                "gender": "All",
            },
            {
                "nct_id": "N3",
                "condition": "solid tumor",
                "brief_title": "Advanced solid tumor study",
                "eligibility_criteria": "Adults with solid tumor.",
                "overall_status": "Recruiting",
                "gender": "All",
            },
        ]
    )
    search = ClinicalTrialSearch(
        search_backend=backend,
        embedder=None,
        entity_annotator=FakeAnnotator(),
    )
    profile = PatientProfile(
        patient_id="P1",
        conditions=[_fact("condition-1", "condition", "lung cancer")],
        genomic_findings=[_fact("gene-1", "genomic_finding", "EGFR mutation")],
        medications=[
            _fact("med-1", "medication", "osimertinib", temporality="prior")
        ],
    )
    plan = search.build_query_plan(
        profile=profile,
        matching_summary={
            "main_conditions": ["lung cancer"],
            "other_conditions": [],
            "patient_narrative": ["Patient has lung cancer with EGFR mutation."],
        },
        config={},
        age="all",
        sex="ALL",
        overall_status="All",
    )

    trials, scores, evidence = search.search_trials_with_plan(
        query_plan=plan,
        age_input="all",
        sex="ALL",
        overall_status="All",
        size=3,
        per_channel_size=3,
        search_mode="bm25",
    )

    assert trials[0]["nct_id"] == "N1"
    assert scores[0] > scores[-1]
    assert "N3" in {trial["nct_id"] for trial in trials}
    n1 = next(item for item in evidence if item.nct_id == "N1")
    assert len({channel["channel"] for channel in n1.channels}) > 1


def test_run_first_level_search_uses_narrative_channel_and_writes_artifacts(tmp_path):
    backend = InMemorySearchBackend(
        trials=[
            {
                "nct_id": "N1",
                "condition": "rare sarcoma",
                "brief_title": "Rare sarcoma trial",
                "eligibility_criteria": "Adults with rare sarcoma.",
                "overall_status": "Recruiting",
                "gender": "All",
            }
        ]
    )
    profile = PatientProfile(
        patient_id="P1",
        conditions=[_fact("condition-1", "condition", "unknown condition")],
    )

    result = run_first_level_search(
        {
            "main_conditions": ["unknown condition"],
            "other_conditions": [],
            "patient_narrative": ["Patient has rare sarcoma."],
        },
        str(tmp_path),
        {"age": "all", "gender": "ALL"},
        None,
        None,
        _config(enabled=True),
        backend,
        patient_profile=profile,
    )

    assert result is not None
    nct_ids, *_ = result
    assert nct_ids == ["N1"]
    assert (tmp_path / "first_level_query_plan.json").exists()
    candidates = json.loads((tmp_path / "first_level_candidates.json").read_text())
    assert candidates["candidates"][0]["nct_id"] == "N1"
    assert (tmp_path / "nct_ids.txt").read_text().strip() == "N1"


def test_run_first_level_search_disabled_uses_single_query_path(tmp_path):
    backend = InMemorySearchBackend(
        trials=[
            {
                "nct_id": "N1",
                "condition": "lung cancer",
                "brief_title": "Lung cancer trial",
                "eligibility_criteria": "Adults with lung cancer.",
                "overall_status": "Recruiting",
                "gender": "All",
            }
        ]
    )

    result = run_first_level_search(
        {
            "main_conditions": ["lung cancer"],
            "other_conditions": [],
            "patient_narrative": ["Patient has lung cancer."],
        },
        str(tmp_path),
        {"age": "all", "gender": "ALL"},
        None,
        None,
        _config(enabled=False),
        backend,
        patient_profile=PatientProfile(patient_id="P1"),
    )

    assert result is not None
    nct_ids, *_ = result
    assert nct_ids == ["N1"]
    assert not (tmp_path / "first_level_query_plan.json").exists()


class _RecordingBackend:
    def __init__(self):
        self.calls = []

    def search_trials(self, **kwargs):
        self.calls.append(kwargs)
        return [], []


def test_hard_filters_config_controls_applied_filters():
    profile = PatientProfile(
        patient_id="P1", conditions=[_fact("c1", "condition", "lung cancer")]
    )
    summary = {"main_conditions": ["lung cancer"], "other_conditions": [], "patient_narrative": []}

    # hard_filters=[] disables age/sex/status filtering entirely.
    backend = _RecordingBackend()
    search = ClinicalTrialSearch(
        search_backend=backend, embedder=None, entity_annotator=FakeAnnotator()
    )
    plan = search.build_query_plan(
        profile=profile, matching_summary=summary, config={"hard_filters": []},
        age=64, sex="female", overall_status="Recruiting",
    )
    search.search_trials_with_plan(
        query_plan=plan, age_input=64, sex="female", overall_status="Recruiting",
        size=10, per_channel_size=10, search_mode="bm25",
    )
    assert backend.calls
    for call in backend.calls:
        assert call["age"] is None
        assert call["sex"] == "all"
        assert call["overall_status"] is None

    # hard_filters=["overall_status"] applies only the recruitment-status filter.
    backend2 = _RecordingBackend()
    search2 = ClinicalTrialSearch(
        search_backend=backend2, embedder=None, entity_annotator=FakeAnnotator()
    )
    plan2 = search2.build_query_plan(
        profile=profile, matching_summary=summary,
        config={"hard_filters": ["overall_status"]},
        age=64, sex="female", overall_status="Recruiting",
    )
    search2.search_trials_with_plan(
        query_plan=plan2, age_input=64, sex="female", overall_status="Recruiting",
        size=10, per_channel_size=10, search_mode="bm25",
    )
    assert backend2.calls
    for call in backend2.calls:
        assert call["age"] is None
        assert call["sex"] == "all"
        assert call["overall_status"] == "Recruiting"


class FakeAnnotator:
    def annotate_texts_in_parallel(self, texts, max_workers=1, retries=1, delay=0):
        return [
            [
                {
                    "entity_group": "disease",
                    "text": text,
                    "synonyms": ["NSCLC", "lung carcinoma"],
                }
            ]
            for text in texts
        ]


def _fact(
    fact_id: str,
    category: str,
    label: str,
    *,
    negated: bool = False,
    temporality: str | None = None,
) -> ClinicalFact:
    return ClinicalFact(
        fact_id=fact_id,
        category=category,
        label=label,
        negated=negated,
        temporality=temporality,
        provenance=Provenance(source_format="test"),
    )


def _config(*, enabled: bool) -> dict:
    return {
        "search": {
            "mode": "bm25",
            "vector_score_threshold": 0.5,
            "max_trials_first_level": 1000,
            "first_level": {
                "enabled": enabled,
                "max_trials": 1000,
                "per_channel_size": 300,
                "rrf_k": 60,
                "vector_score_threshold": 0.0,
                "llm_expansion_enabled": False,
                "write_reports": True,
            },
        }
    }
