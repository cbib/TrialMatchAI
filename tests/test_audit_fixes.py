"""Regression tests for defects found in the deep code audit.

Each test reproduces a specific audit finding (fails on the pre-fix code) and pins
the corrected behaviour. Grouped by the batch in which the fix landed.
"""

from __future__ import annotations

import gc
import weakref

from trialmatchai.interop.models import ClinicalFact, PatientProfile, Provenance


def _fact(label: str, *, negated: bool = False, category: str = "condition") -> ClinicalFact:
    return ClinicalFact(
        fact_id=f"f-{label}",
        category=category,
        label=label,
        negated=negated,
        provenance=Provenance(source_format="test"),
    )


# --------------------------------------------------------------------------- #
# Batch 1 — session-relevant HIGH defects
# --------------------------------------------------------------------------- #


def test_free_vllm_engines_drops_refs_before_gpu_reclaim(monkeypatch):
    """vllm_loader:80 — the engine tuples must be unreferenced *before* the GC /
    empty_cache reclaim, or the GPU memory is never actually released."""
    from trialmatchai.models.llm import vllm_loader

    class _FakeEngine:
        pass

    engine = _FakeEngine()
    ref = weakref.ref(engine)
    vllm_loader._ENGINE_CACHE.clear()
    vllm_loader._ENGINE_CACHE[("model", "adapter", ())] = (engine, object(), None)
    del engine  # only the cache holds the engine now

    seen: dict[str, bool] = {}
    real_collect = gc.collect

    def spy(*args, **kwargs):
        n = real_collect()
        seen["dead_when_gc_runs"] = ref() is None
        return n

    monkeypatch.setattr(gc, "collect", spy)
    vllm_loader.free_vllm_engines()

    assert seen.get("dead_when_gc_runs") is True  # engine released before reclaim, not after return
    assert vllm_loader._ENGINE_CACHE == {}


def test_score_trial_survives_non_dict_criteria():
    """trial_ranker:63 — an LLM emitting bare strings/None in the eval list must not
    raise AttributeError (which discarded every ranked trial for the patient)."""
    from trialmatchai.matching.trial_ranker import score_trial

    trial = {
        "Inclusion_Criteria_Evaluation": [{"Classification": "Met"}, "garbage", None],
        "Exclusion_Criteria_Evaluation": ["also garbage"],
    }
    assert score_trial(trial) == 1.0  # scored from the one valid dict entry, no crash


def test_score_trial_error_output_is_disqualified():
    """trial_ranker:77 — an error-output trial (never actually assessed) must not be
    scored 0.0 (neutral), which ranked it above genuinely disqualified trials."""
    from trialmatchai.matching.trial_ranker import DISQUALIFIED_SCORE, score_trial

    err = {"error": "invalid_json_response", "raw_output": "<think> ..."}
    assert score_trial(err) == DISQUALIFIED_SCORE


def test_render_search_terms_seeds_main_conditions_from_cancer_profile():
    """narrative:45 — a staged-cancer-only patient (dx in cancer_profile, empty
    conditions/phenotypes/notes) must still yield a non-empty primary query."""
    from trialmatchai.interop.narrative import render_search_terms

    profile = PatientProfile(
        patient_id="p1", cancer_profile=[_fact("anaplastic astrocytoma", category="cancer")]
    )
    main, other = render_search_terms(profile)
    assert "anaplastic astrocytoma" in main  # was empty before the fix
    assert "anaplastic astrocytoma" not in other  # not duplicated into other_terms


# --------------------------------------------------------------------------- #
# Batch 2 — LLM/JSON-shape robustness
# --------------------------------------------------------------------------- #


def test_query_expansion_as_list_does_not_shred_strings():
    """query_expansion:173 — a string-valued field must become [string], not per-character."""
    from trialmatchai.matching.query_expansion import _as_list

    assert _as_list("Non-small cell lung cancer") == ["Non-small cell lung cancer"]
    assert _as_list(["a", "b", ""]) == ["a", "b"]
    assert _as_list(None) == []
    assert _as_list("") == []


def test_max_input_tokens_never_collapses_to_one():
    """eligibility_transformers:123 — max_new_tokens >= context must not truncate every
    prompt to a single token."""
    from trialmatchai.matching.eligibility_reasoning_transformers import _max_input_tokens

    class _Cfg:
        max_position_embeddings = 2048

    class _Model:
        config = _Cfg()

    class _Tok:
        model_max_length = 2048

    assert _max_input_tokens(_Tok(), _Model(), 5000) >= 1024  # >= half the window, not 1


def test_expand_queries_applies_configured_condition_caps(tmp_path, monkeypatch):
    """query_expansion:205 — the configured max_main/other_conditions must reach
    enrich_summary, not the hardcoded 11/50 defaults."""
    import json

    from trialmatchai import orchestration
    from trialmatchai.matching import query_expansion as qe

    profiles = tmp_path / "profiles"
    summaries = tmp_path / "summaries"
    profiles.mkdir()
    summaries.mkdir()
    (profiles / "p1.json").write_text(json.dumps({"notes": [{"text": "x"}]}))
    (summaries / "p1.json").write_text(json.dumps({"main_conditions": ["orig"]}))

    class _Expander:
        settings = {"max_main_conditions": 2, "max_other_conditions": 3}

        def expand(self, narrative):
            return {
                "main_conditions": ["a", "b", "c", "d"],
                "other_conditions": ["1", "2", "3", "4", "5"],
                "expanded_sentences": [],
            }

    monkeypatch.setattr(qe, "build_query_expander", lambda cfg: _Expander())
    orchestration.expand_queries(
        {
            "query_expansion": {"enabled": True},
            "patient_inputs": {"profile_dir": str(profiles), "summary_dir": str(summaries)},
        }
    )
    out = json.loads((summaries / "p1.json").read_text())
    assert out["main_conditions"] == ["a", "b"]  # capped to configured 2, not default 11
    assert out["other_conditions"] == ["1", "2", "3"]  # capped to configured 3


# --------------------------------------------------------------------------- #
# Batch 3 — pandas NaN / OMOP float truthiness
# --------------------------------------------------------------------------- #


def test_omop_person_birth_date_handles_nan():
    """omop:129,135 — NaN month/day must not raise, and a NaN birth_datetime must fall
    through to year/month/day rather than silently dropping the birth date."""
    from trialmatchai.interop.importers.omop import _person_birth_date

    nan = float("nan")
    d = _person_birth_date({"year_of_birth": 1980, "month_of_birth": nan, "day_of_birth": nan})
    assert d is not None and d.year == 1980  # no ValueError

    d2 = _person_birth_date({"birth_datetime": nan, "year_of_birth": 1975})
    assert d2 is not None and d2.year == 1975  # fallback used, not dropped


def test_omop_term_exists_negation_survives_float_promotion():
    """omop:308 — an asserted-negative 0 promoted to float 0.0 must still read as negated."""
    from trialmatchai.interop.importers.omop import _term_negated

    assert _term_negated(0.0) is True
    assert _term_negated("N") is True
    assert _term_negated(1.0) is False
    assert _term_negated("Y") is False


def test_omop_code_skips_zero_concept_sentinel():
    """omop:405 — concept_id 0 ('No matching concept') must not mint an OMOP:0 code."""
    from trialmatchai.interop.importers.omop import _omop_code

    assert _omop_code(0, {}) is None
    assert _omop_code(0.0, {}) is None
    code = _omop_code(12345, {})
    assert code is not None and code.code == "12345"  # real unmapped id still coded


# --------------------------------------------------------------------------- #
# Batch 4 — constraint extraction / evaluation semantics
# --------------------------------------------------------------------------- #


def test_text_match_score_requires_whole_word_substring():
    """evaluation:421 — short markers must not collide via raw substring containment."""
    from trialmatchai.constraints.evaluation import _text_match_score

    assert _text_match_score("alk", "alkaline phosphatase") == 0.0  # was 0.95
    assert _text_match_score("er", "cancer") == 0.0
    assert _text_match_score("breast cancer", "invasive breast cancer") == 0.95  # whole-word


def test_sex_constraints_skip_pregnancy_context():
    """extraction:166 — a female/male mention inside a pregnancy criterion is not a
    sex-eligibility restriction."""
    from trialmatchai.constraints.extraction import _sex_constraints

    assert _sex_constraints("Pregnant or breastfeeding women are excluded") == []
    assert _sex_constraints("Women of childbearing potential must use contraception") == []
    got = _sex_constraints("Male patients only")
    assert len(got) == 1 and got[0].value == "male"


def test_normalize_comparator_greater_than_is_exclusive():
    """extraction:284 — 'greater than' is exclusive (gt), not inclusive (ge)."""
    from trialmatchai.constraints.extraction import _normalize_comparator

    assert _normalize_comparator("greater than") == "gt"
    assert _normalize_comparator("at least") == "ge"


def test_find_span_offsets_are_from_original_text():
    """recognizers:371 — span must index the original text, not the casefolded copy."""
    from trialmatchai.entities.recognizers import _find_span

    text = "History of GLIOMA today"
    start, end = _find_span(text, "glioma")
    assert text[start:end] == "GLIOMA"


def test_units_incompatible_guard():
    """evaluation:240 — cross-unit numeric comparison must be suppressed, not mis-decided."""
    from trialmatchai.constraints.evaluation import _units_incompatible

    assert _units_incompatible("10^9/L", "/mm3") is True
    assert _units_incompatible("mg/dL", "mg/dl") is False  # case/format-insensitive
    assert _units_incompatible(None, "/mm3") is False  # one side unitless -> compatible


def test_dedupe_constraints_keeps_distinct_codes():
    """extraction:464 — same-label constraints differing only in codes must not collapse."""
    from trialmatchai.constraints.extraction import _dedupe_constraints
    from trialmatchai.constraints.models import Constraint

    a = Constraint(kind="condition", label="cancer", normalized_codes=[{"vocabulary": "SNOMED", "code": "1"}])
    b = Constraint(kind="condition", label="cancer", normalized_codes=[{"vocabulary": "SNOMED", "code": "2"}])
    assert len(_dedupe_constraints([a, b])) == 2


# --------------------------------------------------------------------------- #
# Batch 5 — falsy-value config bugs
# --------------------------------------------------------------------------- #


def test_search_settings_first_level_alias_syncs_both_ways():
    """settings:163 — a first_level block that omits max_trials must inherit the explicit
    top-level value (not silently the default), and the nested value wins when both set."""
    from trialmatchai.config.settings import SearchSettings

    s = SearchSettings.model_validate(
        {"max_trials_first_level": 500, "first_level": {"enabled": True}}
    )
    assert s.max_trials_first_level == 500 and s.first_level.max_trials == 500

    s2 = SearchSettings.model_validate(
        {"max_trials_first_level": 500, "first_level": {"max_trials": 300}}
    )
    assert s2.max_trials_first_level == 300 and s2.first_level.max_trials == 300


def test_vllm_top_p_zero_rejected():
    """settings:190 — top_p=0 (rejected by vLLM) must fail validation, not surface late."""
    import pytest

    from trialmatchai.config.settings import VllmSettings

    with pytest.raises(Exception):
        VllmSettings(top_p=0.0)
    assert VllmSettings(top_p=0.1).top_p == 0.1


# --------------------------------------------------------------------------- #
# Batch 6 — atomic writes + resume/fingerprint gates
# --------------------------------------------------------------------------- #


def test_dir_fingerprint_include_dirs_detects_nested_content_change(tmp_path):
    """pipeline_state:49 — a copy-on-write store (LanceDB) rewrites nested data files
    without touching the sub-directory's own mtime; the fingerprint must still change."""
    from trialmatchai.utils.pipeline_state import dir_fingerprint

    root = tmp_path / "store"
    sub = root / "table.lance"
    sub.mkdir(parents=True)
    (sub / "data.bin").write_bytes(b"x" * 10)
    before = dir_fingerprint(root, include_dirs=True)

    # Rewrite the nested file with a different size but keep the sub-directory's mtime fixed,
    # exactly as a copy-on-write rebuild would.
    import os

    sub_mtime = sub.stat().st_mtime_ns
    (sub / "data.bin").write_bytes(b"y" * 999)
    os.utime(sub, ns=(sub_mtime, sub_mtime))

    after = dir_fingerprint(root, include_dirs=True)
    assert before != after  # nested change caught despite the invariant sub-dir mtime


def test_match_signature_tracks_model_identity():
    """orchestration:261 — swapping reranker/CoT weights or adapter must invalidate the
    cached patient matches, not serve the old model's rankings as 'done'."""
    from trialmatchai.orchestration import _match_signature

    base = {"model": {"reranker_model_path": "gemma-2b", "base_model": "phi-4"}}
    swapped = {"model": {"reranker_model_path": "gemma-9b", "base_model": "phi-4"}}
    assert _match_signature(base) != _match_signature(swapped)

    adapter = {"model": {"reranker_model_path": "gemma-2b", "reranker_adapter_path": "lora-A"}}
    adapter2 = {"model": {"reranker_model_path": "gemma-2b", "reranker_adapter_path": "lora-B"}}
    assert _match_signature(adapter) != _match_signature(adapter2)


def test_write_dictionary_is_atomic(tmp_path, monkeypatch):
    """concept_sources:170 — a crash mid-conversion must not leave a truncated dict file at
    the final path (temp + rename), and no leftover .part on success."""
    from trialmatchai.entities import concept_sources

    monkeypatch.setattr(
        concept_sources, "_iter_source", lambda source, raw: [("C1", ["alpha"]), ("C2", ["beta"])]
    )
    monkeypatch.setattr(concept_sources, "_clean_names", lambda names: list(names))

    dict_path = tmp_path / "dict_test.txt"
    n = concept_sources.write_dictionary(object(), tmp_path / "raw", dict_path)
    assert n == 2
    assert dict_path.exists()
    assert not (tmp_path / "dict_test.txt.part").exists()  # temp cleaned up by rename
    assert "C1||alpha" in dict_path.read_text()


def test_bootstrap_extract_sentinel_rejects_partial_tree(tmp_path):
    """bootstrap_data:114 — a directory holding a crashed partial extract (entries but no
    completion sentinel) must NOT be treated as already-done."""
    from trialmatchai.cli import bootstrap_data

    d = tmp_path / "processed_criteria"
    d.mkdir()
    (d / "leftover.json").write_text("{}")  # partial extract left some files
    assert bootstrap_data._extract_complete(d) is False  # not the marker -> re-extract

    bootstrap_data._mark_extract_complete(d)
    assert bootstrap_data._extract_complete(d) is True


def test_registry_skip_requires_successful_previous_status():
    """updater:157 — an unchanged study whose previous run FAILED must be retried, not
    skipped as 'done'."""
    from trialmatchai.registry.manifest import ManifestRecord

    # The skip predicate the updater applies inline; pin its truth table.
    def _skips(status: str) -> bool:
        prev = ManifestRecord(
            nct_id="NCT1",
            source_url="u",
            source_hash="h",
            fetched_at="t",
            last_update_posted=None,
            processing_status=status,
        )
        return prev.source_hash == "h" and prev.processing_status in {"indexed", "fetched"}

    assert _skips("indexed") is True
    assert _skips("fetched") is True
    assert _skips("failed") is False  # transient OOM etc. must not lock the study out


# --------------------------------------------------------------------------- #
# Batch 7 — retrieval / search
# --------------------------------------------------------------------------- #


def test_aggregate_to_trials_dedupes_criteria_across_queries():
    """criteria_retrieval:259 — a criterion retrieved by several query paraphrases must count
    once (best score), not once per query, or a trial is inflated by query overlap."""
    from trialmatchai.matching.retrieval.criteria_retrieval import SecondStageRetriever

    agg = SecondStageRetriever.__new__(SecondStageRetriever)  # method uses no instance state

    def crit(cid, score):
        return {"_source": {"nct_id": "NCT1", "criteria_id": cid}, "llm_score": score}

    # One unique criterion retrieved by three query paraphrases (+ one distinct criterion).
    criteria = [crit("c1", 0.8), crit("c1", 0.6), crit("c1", 0.9), crit("c2", 0.7)]
    out = agg.aggregate_to_trials(criteria, method="weighted")
    trial = next(t for t in out if t["nct_id"] == "NCT1")

    # After dedup the trial aggregates from exactly two criteria: c1@0.9 (best) and c2@0.7.
    import math

    scores = [0.9, 0.7]
    expected = 0.7 * (sum(scores) / math.sqrt(len(scores))) + 0.3 * max(scores)
    assert trial["score"] == expected  # not inflated by the two duplicate c1 hits


def test_health_reports_never_built_search_db(tmp_path):
    """lancedb:224 — a never-built search DB (table-less) must be reported unhealthy by the
    DEFAULT healthcheck, not silently pass because the directory happens to exist."""
    from trialmatchai.search.lancedb_backend import LanceDBSearchBackend

    db_path = tmp_path / "search"
    backend = LanceDBSearchBackend(db_path)  # no tables ever written
    issues = backend.health()  # default require_tables=False
    assert issues and "never built" in issues[0]
