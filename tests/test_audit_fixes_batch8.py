"""Regression tests for batch-8 audit fixes (interop/fhir + misc).

Generated from the per-file fix agents; each test pins a specific finding's corrected
behaviour. Grouped by source file via '# ====' banners.
"""

from __future__ import annotations
from trialmatchai.interop.importers.fhir import _value_x, _genomic_label
from dataclasses import replace
from trialmatchai.entities.linker import (
    ConceptLinker,
    LanceDBConceptStore,
    _lexical_score,
)
from trialmatchai.entities.schemas import load_entity_schemas
from trialmatchai.entities.types import ConceptCandidate, EntityAnnotation
import pytest
from trialmatchai.interop.models import PatientProfile
from trialmatchai.matching.retrieval.trial_retrieval import ClinicalTrialSearch
from trialmatchai.search import InMemorySearchBackend
from trialmatchai.models.embedding import HashingTextEmbedder
import json
from trialmatchai.trec.qrels import _retrieved_for_patient
from trialmatchai import orchestration as orch


# ==== src/trialmatchai/interop/importers/fhir.py ====


def test_value_range_preserves_negative_low_sign():
    # low=-2.5, high=3.0 must not have its leading minus stripped.
    assert _value_x({"valueRange": {"low": {"value": -2.5}, "high": {"value": 3.0}}}) == "-2.5-3.0"
    # single bounds emit the raw value, not a mangled dash string.
    assert _value_x({"valueRange": {"high": {"value": 5}}}) == "5"
    assert _value_x({"valueRange": {"low": {"value": 5}}}) == "5"
    # both-bounds behavior unchanged.
    assert _value_x({"valueRange": {"low": {"value": 1}, "high": {"value": 5}}}) == "1-5"


def test_value_ratio_does_not_emit_literal_none():
    assert _value_x({"valueRatio": {"numerator": {"value": 5}, "denominator": {"value": 10}}}) == "5/10"
    # missing denominator must not produce '5/None'.
    assert _value_x({"valueRatio": {"numerator": {"value": 5}}}) == "5"
    # empty ratio yields no value (None), never 'None/None'.
    assert _value_x({"valueRatio": {}}) is None


def test_genomic_label_handles_str_list_and_mapping_types():
    # MolecularSequence R4 'type' is a code string.
    assert _genomic_label("dna") == "dna"
    # GenomicStudy 'type' is a list of CodeableConcept.
    assert _genomic_label([{"coding": [{"display": "Whole genome sequencing"}]}]) == "Whole genome sequencing"
    # A single CodeableConcept mapping still works.
    assert _genomic_label({"text": "variant"}) == "variant"
    # Unusable input falls back to empty string (caller supplies id / default).
    assert _genomic_label(None) == ""


def test_genomic_resource_not_dropped_to_unsupported(tmp_path):
    import json

    from trialmatchai.interop.importers.fhir import import_fhir

    path = tmp_path / "bundle.json"
    path.write_text(
        json.dumps(
            {
                "resourceType": "Bundle",
                "entry": [
                    {"resource": {"resourceType": "Patient", "id": "p1"}},
                    {"resource": {"resourceType": "MolecularSequence", "id": "ms1", "type": "dna", "subject": {"reference": "Patient/p1"}}},
                    {"resource": {"resourceType": "GenomicStudy", "id": "gs1", "type": [{"coding": [{"display": "WGS"}]}], "subject": {"reference": "Patient/p1"}}},
                ],
            }
        )
    )
    profile = import_fhir(path, input_format="fhir")[0]
    labels = {g.label for g in profile.genomic_findings}
    assert {"dna", "WGS"} <= labels
    assert not any(u.get("resourceType") in {"MolecularSequence", "GenomicStudy"} for u in profile.unsupported)


def test_family_history_without_relationship_has_no_leading_colon(tmp_path):
    import json

    from trialmatchai.interop.importers.fhir import import_fhir

    path = tmp_path / "bundle.json"
    path.write_text(
        json.dumps(
            {
                "resourceType": "Bundle",
                "entry": [
                    {"resource": {"resourceType": "Patient", "id": "p1"}},
                    {
                        "resource": {
                            "resourceType": "FamilyMemberHistory",
                            "id": "fh1",
                            "patient": {"reference": "Patient/p1"},
                            "condition": [{"code": {"text": "Breast cancer"}}],
                        }
                    },
                ],
            }
        )
    )
    profile = import_fhir(path, input_format="fhir")[0]
    fh = profile.family_history[0]
    assert fh.label == "Breast cancer"
    assert not fh.label.startswith(":")
    assert fh.extra.get("relationship") is None


# ==== src/trialmatchai/interop/utils.py ====
def test_make_fact_folds_temporality_into_fact_id_so_dated_occurrences_dont_collide():
    from trialmatchai.interop.models import Provenance
    from trialmatchai.interop.utils import make_fact

    # Same patient/table/concept, no per-row source_resource (as OMOP produces),
    # differing only by date. These must not collapse to one fact_id.
    prov = Provenance(
        source_format="omop",
        source_id="pt1",
        source_path="/data/omop",
        source_table="CONDITION_OCCURRENCE",
    )
    early = make_fact(category="condition", label="Diabetes mellitus", provenance=prov, temporality="2020-01-01")
    late = make_fact(category="condition", label="Diabetes mellitus", provenance=prov, temporality="2022-05-01")
    same = make_fact(category="condition", label="Diabetes mellitus", provenance=prov, temporality="2020-01-01")

    assert early.fact_id != late.fact_id  # distinct occurrences stay distinct
    assert early.fact_id == same.fact_id  # true duplicates still dedup to one id

    # Backward compatible: when no temporality exists the id is unchanged
    # (stable_id drops None/empty parts before hashing).
    no_temp = make_fact(category="condition", label="X", provenance=prov)
    explicit_none = make_fact(category="condition", label="X", provenance=prov, temporality=None)
    assert no_temp.fact_id == explicit_none.fact_id


# ==== src/trialmatchai/services/preflight.py ====
def test_require_patient_inputs_flags_empty_profile_dir():
    from trialmatchai.services.preflight import _require_patient_inputs

    issues = []
    _require_patient_inputs(issues, {"patient_inputs": {"profile_dir": ""}})
    assert issues == ["patient_inputs.profile_dir is not configured."]


def test_require_patient_inputs_flags_missing_profile_dir_key():
    from trialmatchai.services.preflight import _require_patient_inputs

    issues = []
    _require_patient_inputs(issues, {"patient_inputs": {}})
    assert issues == ["patient_inputs.profile_dir is not configured."]


def test_require_patient_inputs_flags_none_profile_dir():
    from trialmatchai.services.preflight import _require_patient_inputs

    issues = []
    _require_patient_inputs(issues, {"patient_inputs": {"profile_dir": None}})
    assert issues == ["patient_inputs.profile_dir is not configured."]


def test_require_patient_inputs_reports_missing_dir(tmp_path):
    from trialmatchai.services.preflight import _require_patient_inputs

    missing = tmp_path / "no-such-profiles"
    issues = []
    _require_patient_inputs(issues, {"patient_inputs": {"profile_dir": str(missing)}})
    assert issues == [f"patient_inputs.profile_dir does not exist: {missing}"]


def test_require_patient_inputs_passes_for_existing_dir(tmp_path):
    from trialmatchai.services.preflight import _require_patient_inputs

    profiles = tmp_path / "profiles"
    profiles.mkdir()
    issues = []
    _require_patient_inputs(issues, {"patient_inputs": {"profile_dir": str(profiles)}})
    assert issues == []


# ==== src/trialmatchai/registry/criteria_chunking.py ====
def test_midline_ordinal_in_prose_not_split():
    from trialmatchai.registry.criteria_chunking import _split_inline
    # A bare "N." following a word in prose is a decimal/ordinal, not a list marker,
    # so the line must not be split at "3.".
    assert _split_inline("Patient seen on day 3. Blood work required") == [
        "Patient seen on day 3. Blood work required"
    ]


def test_midline_numeric_list_still_splits_in_list_context():
    from trialmatchai.registry.criteria_chunking import _split_inline
    # When the line already starts with a marker it is a list; packed single-period
    # numeric items still split (intended behavior).
    assert _split_inline("1. Pregnancy 2. Active infection 3. Prior therapy") == [
        "1. Pregnancy ",
        "2. Active infection ",
        "3. Prior therapy",
    ]


def test_unbalanced_open_paren_does_not_swallow_semicolon():
    from trialmatchai.registry.criteria_chunking import _split_semicolons
    # A stray "(" must not mask the rest of the line; the ';' stays a split point.
    assert _split_semicolons("Age >= 18 (years and older; no upper limit") == [
        "Age >= 18 (years and older",
        "no upper limit",
    ]


def test_balanced_parens_still_masked():
    from trialmatchai.registry.criteria_chunking import _mask_parens
    # A closed span is still blanked so markers/semicolons inside are not split points.
    masked = _mask_parens("a (b; c) d")
    assert ";" not in masked
    assert masked == "a        d"


# ==== src/trialmatchai/entities/linker.py ====



def test_lexical_score_substring_no_longer_flatscores_to_accept():
    # A bare mention that is a proper subset of a more specific concept name must not
    # clear the default accept gate (0.7); the old code flat-scored it 0.86.
    concept = ConceptCandidate(
        concept_id="DOID:1", vocabulary_id="DOID", concept_code="1",
        concept_name="hepatocellular carcinoma", domain_id="Disease",
    )
    assert _lexical_score("carcinoma", concept) < 0.7
    # mid-word substring ("cell" inside "hepatocellular") is not a whole-token match
    assert _lexical_score("cell", concept) == 0.0
    # exact name still scores 1.0
    assert _lexical_score("hepatocellular carcinoma", concept) == 1.0
    # empty/whitespace query -> 0.0
    assert _lexical_score("   ", concept) == 0.0


def test_linker_abstains_on_partial_substring_only_candidate():
    only = ConceptCandidate(
        concept_id="DOID:1", vocabulary_id="DOID", concept_code="1",
        concept_name="hepatocellular carcinoma", domain_id="Disease",
    )

    class _RRFStore:
        # Mimics an RRF store: returns the (wrong, more-specific) candidate at score 1.0.
        def search(self, query, **_kwargs):
            return [replace(only, score=1.0)]

    linker = ConceptLinker(
        _RRFStore(), load_entity_schemas(None),
        accept_threshold=0.7, reject_threshold=0.5,
    )
    out = linker.link_annotation(
        EntityAnnotation(entity_group="disease", text="carcinoma", start=0, end=9, score=0.9)
    )
    assert out.linker_status in ("rejected", "ambiguous")
    assert out.normalized_id == ("CUI-less",)


def test_embed_query_empty_string_degrades_instead_of_raising():
    class _RaisingEmbedder:
        def embed_text(self, text):
            if not text.strip():
                raise ValueError("Cannot embed empty text")
            return [1.0, 0.0]

    # Bypass __init__ (which needs lancedb + a real table); only self.embedder is used.
    store = LanceDBConceptStore.__new__(LanceDBConceptStore)
    store.embedder = _RaisingEmbedder()
    assert store._embed_query("   ") is None
    assert store._embed_query("") is None
    assert store._embed_query("diabetes") == [1.0, 0.0]


# ==== src/trialmatchai/main.py ====



def _make_scored_retriever(scored):
    class _FakeRetriever:
        last_constraint_evaluations = []

        def get_synonyms(self, query):
            return []

        def retrieve_and_rank(self, queries, nct_ids, top_n, patient_context=None, constraints_config=None):
            return [{"nct_id": nid, "score": s} for nid, s in scored][:top_n]

    return _FakeRetriever()


def test_run_second_level_search_returns_pure_second_level_scores(tmp_path):
    from trialmatchai.main import run_second_level_search

    retriever = _make_scored_retriever([("NCT1", 0.2), ("NCT2", 0.9)])
    config = {
        "search": {"max_trials_second_level": 100, "second_level_keep_divisor": 1},
        "constraints": {"enabled": False},
        "rag": {"enabled": False},
        "use_cot_reasoning": False,
    }
    semi_final, _path, second_level_scores = run_second_level_search(
        str(tmp_path),
        ["NCT1", "NCT2"],
        ["lung cancer"], [], [],
        retriever,
        {"NCT1": 10.0, "NCT2": 0.0},  # first-level dominates the combined shortlist key
        config,
    )
    combined = dict(semi_final)
    # The shortlist key fuses the first-level and reranker RANKINGS (RRF) and is used only
    # for selection; NCT1 leads both rankings, so it leads the fused key...
    assert combined["NCT1"] > combined["NCT2"]
    # ...but the scores threaded to rank_trials are the PURE reranker scores.
    assert second_level_scores == {"NCT1": pytest.approx(0.2), "NCT2": pytest.approx(0.9)}


def test_run_second_level_search_caps_shortlist_to_rag_budget(tmp_path):
    from trialmatchai.main import run_second_level_search

    scored = [(f"NCT{i}", 1.0 - i * 0.05) for i in range(9)]
    retriever = _make_scored_retriever(scored)
    base_config = {
        "search": {"max_trials_second_level": 100, "second_level_keep_divisor": 3},
        "constraints": {"enabled": False},
    }
    nct_ids = [nid for nid, _ in scored]

    rag_on = dict(base_config, rag={"enabled": True, "max_trials_rag": 1}, use_cot_reasoning=True)
    semi_final_on, _p, _s = run_second_level_search(
        str(tmp_path / "on"), nct_ids, ["q"], [], [], retriever, {}, rag_on
    )
    assert len(semi_final_on) == 1  # capped to rag.max_trials_rag

    rag_off = dict(base_config, rag={"enabled": False}, use_cot_reasoning=False)
    semi_final_off, _p2, _s2 = run_second_level_search(
        str(tmp_path / "off"), nct_ids, ["q"], [], [], retriever, {}, rag_off
    )
    assert len(semi_final_off) == 3  # 9 // keep_divisor(3), uncapped


def _setup_pipeline(monkeypatch, tmp_path, patient_inputs, first_level):
    import trialmatchai.main as main_module
    import trialmatchai.models.embedding as embedding_module

    class _Backend:
        def health(self, *, require_tables=False):
            return []

    config = {
        "paths": {
            "output_dir": str(tmp_path / "results"),
            "trials_json_folder": str(tmp_path / "trials"),
        },
        "search_backend": {"backend": "lancedb"},
        "patient_inputs": {},
        "search": {"mode": "bm25"},
        "constraints": {"enabled": False},
        "LLM_reranker": {"enabled": False},
        "rag": {"enabled": False},
        "use_cot_reasoning": False,
        "reporting": {"emit_html": False},
    }
    monkeypatch.setattr(main_module, "load_config", lambda config_path=None: config)
    monkeypatch.setattr(main_module, "run_preflight_checks", lambda *a, **k: [])
    monkeypatch.setattr(main_module, "build_search_backend", lambda cfg: _Backend())
    monkeypatch.setattr(embedding_module, "build_embedder", lambda cfg: object())
    monkeypatch.setattr(main_module, "build_entity_annotator", lambda cfg, embedder: None)
    monkeypatch.setattr(main_module, "_load_patient_inputs", lambda cfg: patient_inputs)
    monkeypatch.setattr(main_module, "run_first_level_search", first_level)
    return main_module


def _summary(pid):
    return {
        "patient_id": pid, "main_conditions": ["lung cancer"],
        "other_conditions": [], "patient_narrative": ["x"],
        "age": "all", "gender": "all",
    }


def test_resume_all_pending_fail_returns_nonzero(tmp_path, monkeypatch):
    profile = PatientProfile.model_validate({"patient_id": "pfail", "demographics": {}})
    done = PatientProfile.model_validate({"patient_id": "pdone", "demographics": {}})
    out = tmp_path / "results" / "pdone"
    out.mkdir(parents=True)
    (out / "ranked_trials.json").write_text("[]", encoding="utf-8")  # valid marker -> skipped

    def _boom_first_level(*a, **k):
        raise RuntimeError("forced failure")

    main_module = _setup_pipeline(
        monkeypatch, tmp_path,
        [(done, _summary("pdone")), (profile, _summary("pfail"))],
        _boom_first_level,
    )
    # Previously returned 0 ("all already matched"), masking the failed pending patient.
    assert main_module.main_pipeline("config.json", resume=True) == 1


def test_first_level_none_counted_as_failed_on_resume(tmp_path, monkeypatch):
    profile = PatientProfile.model_validate({"patient_id": "pnone", "demographics": {}})
    done = PatientProfile.model_validate({"patient_id": "pdone2", "demographics": {}})
    out = tmp_path / "results" / "pdone2"
    out.mkdir(parents=True)
    (out / "ranked_trials.json").write_text("[]", encoding="utf-8")

    main_module = _setup_pipeline(
        monkeypatch, tmp_path,
        [(done, _summary("pdone2")), (profile, _summary("pnone"))],
        lambda *a, **k: None,  # first-level returns None
    )
    # None result must be counted as failed; with a skipped patient present the run
    # must still report failure (return 1), not "all already matched -> 0".
    assert main_module.main_pipeline("config.json", resume=True) == 1


# ==== src/trialmatchai/matching/retrieval/trial_retrieval.py ====


def _make_search():
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
            }
        ]
    )
    return ClinicalTrialSearch(
        search_backend=backend, embedder=None, entity_annotator=None
    )


def test_search_trials_mixed_case_age_wildcard_does_not_raise():
    # 'aLL' is a wildcard casing that used to raise ValueError; it must now be
    # treated as the no-age-filter case, consistent with the plan path.
    search = _make_search()
    trials, scores = search.search_trials(
        condition="lung cancer",
        age_input="aLL",
        sex="all",
        size=5,
        search_mode="bm25",
    )
    assert len(trials) == 1
    assert trials[0]["nct_id"] == "N1"


def test_search_trials_unparseable_age_warns_and_continues(caplog):
    # An unparseable age must warn and proceed without an age filter instead of
    # aborting the whole first-level search with ValueError.
    import logging

    search = _make_search()
    with caplog.at_level(logging.WARNING):
        trials, scores = search.search_trials(
            condition="lung cancer",
            age_input="not-an-age",
            sex="all",
            size=5,
            search_mode="bm25",
        )
    assert len(trials) == 1
    assert any("Could not parse age" in rec.getMessage() for rec in caplog.records)


# ==== src/trialmatchai/models/embedding/text_embedder.py ====


def test_hashing_embedder_non_alphanumeric_is_nonzero_and_normalized():
    embedder = HashingTextEmbedder(dimensions=16)
    vector = embedder.embed_text("!!!")
    # Non-empty punctuation-only text must not collapse to an all-zero
    # (undefined-cosine) embedding.
    assert any(value != 0.0 for value in vector)
    assert round(sum(value * value for value in vector), 6) == 1.0
    # Distinct punctuation strings should not collide to the same bucket.
    assert embedder.embed_text("!!!") != embedder.embed_text("???")


def test_hashing_embedder_non_alphanumeric_nonzero_without_normalize():
    embedder = HashingTextEmbedder(dimensions=16, normalize=False)
    vector = embedder.embed_text("###")
    assert any(value != 0.0 for value in vector)


# ==== src/trialmatchai/trec/qrels.py ====



def test_retrieved_for_patient_ranked_trials_dict_fallback(tmp_path):
    """ranked_trials.json is a dict {"RankedTrials": [...]}; the fallback must
    unwrap it (not iterate the dict's keys, which raised AttributeError)."""
    pdir = tmp_path / "trec-1"
    pdir.mkdir()
    # No nct_ids.txt -> force the ranked_trials.json fallback path.
    (pdir / "ranked_trials.json").write_text(
        json.dumps(
            {
                "RankedTrials": [
                    {"TrialID": "NCT1", "Score": 1.0},
                    {"TrialID": "NCT2", "Score": 0.5},
                ]
            }
        )
    )
    assert _retrieved_for_patient(pdir) == ["NCT1", "NCT2"]


def test_retrieved_for_patient_ranked_trials_bare_list_and_malformed(tmp_path):
    """Legacy bare-list shape still works, and non-dict / missing-TrialID
    entries are skipped rather than crashing."""
    pdir = tmp_path / "trec-2"
    pdir.mkdir()
    (pdir / "ranked_trials.json").write_text(
        json.dumps(["garbage", {"NoTrial": 1}, {"TrialID": "NCT9"}])
    )
    assert _retrieved_for_patient(pdir) == ["NCT9"]


# ==== src/trialmatchai/constraints/evaluation.py ====
def test_inclusion_violation_dominates_matched_credits_not_averaged():
    # Conjunctive inclusion criterion: patient matches the condition (+1.0) but violates a
    # hard numeric bound (ANC 800 < required 1500 -> -0.6). Old mean-averaging gave
    # (1.0 - 0.6)/2 = +0.2, a trial-boosting sign-flip; worst-case min must keep it negative.
    from trialmatchai.constraints.evaluation import (
        INCLUSION_VIOLATION_SIGNAL,
        evaluate_constraint_set,
    )
    from trialmatchai.constraints.models import (
        Constraint,
        ConstraintSet,
        PatientConstraintContext,
        PatientConstraintFact,
    )

    context = PatientConstraintContext(
        patient_id="P1",
        facts=[
            PatientConstraintFact(
                kind="condition",
                label="non-small cell lung cancer",
                evidence_text="metastatic non-small cell lung cancer",
            ),
            PatientConstraintFact(
                kind="lab",
                label="absolute neutrophil count",
                value=800.0,
                unit="/mm3",
                evidence_text="ANC 800/mm3",
            ),
        ],
    )
    constraint_set = ConstraintSet(
        nct_id="N1",
        criteria_id="C1",
        polarity="inclusion",
        source_text="Adults with NSCLC and ANC >= 1500/mm3.",
        constraints=[
            Constraint(kind="condition", label="non-small cell lung cancer"),
            Constraint(
                kind="lab",
                label="absolute neutrophil count",
                comparator="ge",
                value=1500,
                unit="/mm3",
            ),
        ],
    )
    evaluation = evaluate_constraint_set(constraint_set, context)
    assert evaluation.matched_count == 1
    assert evaluation.violated_count == 1
    # Worst-case aggregation: the violated inclusion bound dominates.
    assert evaluation.constraint_signal == INCLUSION_VIOLATION_SIGNAL
    assert evaluation.constraint_signal < 0


# ==== src/trialmatchai/orchestration.py ====


def test_build_index_validates_criteria_before_writing_trials(tmp_path, monkeypatch):
    """orchestration.build_index — an empty-criteria corpus must fail fast BEFORE the
    trials table is written, so a data error never leaves a half-populated index."""
    trials_dir = tmp_path / "processed_trials"
    trials_dir.mkdir()
    (trials_dir / "N1.json").write_text(json.dumps({"nct_id": "N1"}))
    criteria_dir = tmp_path / "processed_criteria"
    criteria_dir.mkdir()  # empty -> no criteria docs

    class FakeBackend:
        db_path = tmp_path / "search"

        def __init__(self):
            self.indexed_trials = False

        def table_exists(self, name):
            return False

        def index_trials(self, docs, recreate=True):
            self.indexed_trials = True
            return len(list(docs))

        def index_criteria(self, docs, recreate=True):
            return len(list(docs))

    fake = FakeBackend()
    monkeypatch.setattr(orch, "build_search_backend", lambda config: fake)

    config = {"search_backend": {"trials_table": "trials", "criteria_table": "criteria"}}
    with pytest.raises(RuntimeError, match="No criteria documents"):
        orch.build_index(
            config,
            processed_trials_folder=trials_dir,
            processed_criteria_folder=criteria_dir,
        )

    # Precondition failed BEFORE any table write: no half-populated index.
    assert fake.indexed_trials is False
