"""Regression tests for completeness-re-audit fixes (19 verified findings).

Generated from the per-file fix agents; each pins a specific finding's corrected behaviour.
"""

from __future__ import annotations
import trialmatchai.cli.build_concepts as build_concepts
from trialmatchai.constraints import (
    build_patient_constraint_context,
    evaluate_constraint_set,
)
from trialmatchai.constraints.evaluation import _parse_numeric_value
from trialmatchai.constraints.models import Constraint, ConstraintSet
from trialmatchai.interop.models import (
    ClinicalFact,
    Demographics,
    PatientProfile,
    Provenance,
)
from trialmatchai.constraints.extraction import _lab_constraints, _biomarker_constraints
from trialmatchai.entities.annotator import build_entity_annotator
from trialmatchai.entities.linker import lexical_reranker
import json
import os
import trialmatchai.entities as ent_mod
import trialmatchai.models.embedding as emb_mod
import trialmatchai.registry.preparation as prep_mod
from trialmatchai.orchestration import _trial_needs_prepare, prepare_corpus
from trialmatchai.utils.file_utils import write_json_file, write_text_file


# ==== src/trialmatchai/cli/build_concepts.py ====



def test_concepts_fingerprint_changes_when_embedder_model_swapped():
    base = build_concepts._concepts_fingerprint(
        "open", ["EntrezGene:Gene:/tmp/d.txt"], ["SNOMED"],
        embedder_model="BAAI/bge-m3", embedder_revision=None, skip_embeddings=False,
    )
    swapped = build_concepts._concepts_fingerprint(
        "open", ["EntrezGene:Gene:/tmp/d.txt"], ["SNOMED"],
        embedder_model="intfloat/e5-large", embedder_revision=None, skip_embeddings=False,
    )
    revised = build_concepts._concepts_fingerprint(
        "open", ["EntrezGene:Gene:/tmp/d.txt"], ["SNOMED"],
        embedder_model="BAAI/bge-m3", embedder_revision="deadbeef", skip_embeddings=False,
    )
    assert base != swapped
    assert base != revised


def test_concepts_fingerprint_changes_when_skip_embeddings_flips():
    with_emb = build_concepts._concepts_fingerprint(
        "open", [], [], embedder_model="BAAI/bge-m3", skip_embeddings=False,
    )
    fts_only = build_concepts._concepts_fingerprint(
        "open", [], [], embedder_model="BAAI/bge-m3", skip_embeddings=True,
    )
    assert with_emb != fts_only


def test_run_build_concepts_rebuilds_when_embedder_model_swapped(tmp_path, monkeypatch):
    import numpy as np

    class _FakeEmbedder:
        def embed_texts(self, texts):
            return np.zeros((len(list(texts)), 4), dtype="float32")

    build_calls = {"n": 0}

    def _fake_build_embedder(config):
        build_calls["n"] += 1
        return _FakeEmbedder()

    monkeypatch.setattr(build_concepts, "build_embedder", _fake_build_embedder)

    dict_file = tmp_path / "dict_Gene.txt"
    dict_file.write_text("C1||BRCA1|breast cancer 1\n", encoding="utf-8")
    db_path = str(tmp_path / "concepts")
    dictionary = [f"EntrezGene:Gene:{dict_file}"]

    cfg_a = {
        "embedder": {"model_name": "model/A"},
        "concept_linker": {"db_path": db_path, "table": "concepts"},
    }
    assert build_concepts.run_build_concepts(cfg_a, dictionary=dictionary) == 0
    fp_a = build_concepts._read_concepts_fingerprint(db_path)
    assert build_calls["n"] == 1

    # Re-running with the SAME embedder must hit the skip branch (no re-embed).
    assert build_concepts.run_build_concepts(cfg_a, dictionary=dictionary) == 0
    assert build_calls["n"] == 1, "unchanged config should skip the rebuild"

    # Swapping the embedder model must invalidate the store and force a rebuild.
    cfg_b = {
        "embedder": {"model_name": "model/B"},
        "concept_linker": {"db_path": db_path, "table": "concepts"},
    }
    assert build_concepts.run_build_concepts(cfg_b, dictionary=dictionary) == 0
    fp_b = build_concepts._read_concepts_fingerprint(db_path)
    assert fp_a != fp_b
    assert build_calls["n"] == 2, "model swap must re-embed instead of skipping"


# ==== src/trialmatchai/cli/run.py ====
def test_cli_run_delegates_to_run_matching_for_corpus_invalidation(monkeypatch):
    """`trialmatchai run` must go through orchestration.run_matching (which carries the
    corpus-fingerprint resume invalidation), not call main_pipeline directly — otherwise
    stale ranked_trials.json are served after a reindex."""
    import sys
    from trialmatchai.config import config_loader as cfg_mod
    from trialmatchai import orchestration as orch
    from trialmatchai.cli import run as run_cli

    monkeypatch.setattr(cfg_mod, "load_config", lambda p: {"sentinel": True, "path": p})
    calls = []
    monkeypatch.setattr(orch, "run_matching", lambda config, **kw: calls.append((config, kw)) or 0)
    monkeypatch.setattr(orch, "free_models", lambda: None)
    # main_pipeline must NOT be invoked directly on this path.
    import trialmatchai.main as main_mod

    def _boom(*a, **k):
        raise AssertionError("cli.run must delegate to run_matching, not main_pipeline")

    monkeypatch.setattr(main_mod, "main_pipeline", _boom)
    monkeypatch.setattr(sys, "argv", ["trialmatchai", "--config", "cfg.json"])

    rc = run_cli.main()
    assert rc == 0
    assert len(calls) == 1
    config_arg, kwargs = calls[0]
    assert config_arg == {"sentinel": True, "path": "cfg.json"}
    assert kwargs.get("resume") is True


def test_cli_run_force_disables_resume(monkeypatch):
    """--force must propagate as resume=False to run_matching (re-match every patient)."""
    import sys
    from trialmatchai.config import config_loader as cfg_mod
    from trialmatchai import orchestration as orch
    from trialmatchai.cli import run as run_cli

    monkeypatch.setattr(cfg_mod, "load_config", lambda p: {"cfg": True})
    calls = []
    monkeypatch.setattr(orch, "run_matching", lambda config, **kw: calls.append(kw) or 0)
    monkeypatch.setattr(orch, "free_models", lambda: None)
    monkeypatch.setattr(sys, "argv", ["trialmatchai", "--force"])

    assert run_cli.main() == 0
    assert calls[0].get("resume") is False


# ==== src/trialmatchai/constraints/evaluation.py ====



def _clinical_fact(fact_id, category, label, **kwargs):
    return ClinicalFact(
        fact_id=fact_id,
        category=category,
        label=label,
        provenance=Provenance(source_format="test"),
        **kwargs,
    )


def test_parse_numeric_value_preserves_thousands_separator_and_unit():
    # A thousands comma must not truncate the value nor drop the unit.
    assert _parse_numeric_value("ANC 1,200 cells/mm3") == (1200.0, "cells/mm3")
    assert _parse_numeric_value("ANC 1,500 /mm3") == (1500.0, "/mm3")
    # Plain integers and decimals with units remain unaffected.
    assert _parse_numeric_value("1800 /mm3") == (1800.0, "/mm3")
    assert _parse_numeric_value("1.5 x10^9/L") == (1.5, "x10^9/L")


def test_thousands_separator_lab_value_satisfies_same_unit_constraint():
    profile = PatientProfile(
        patient_id="P1",
        demographics=Demographics(age_years=55),
        observations=[
            _clinical_fact(
                "obs-1",
                "observation",
                "absolute neutrophil count",
                description="1,500 /mm3",
                evidence_text="ANC 1,500 /mm3.",
            )
        ],
    )
    context = build_patient_constraint_context(profile)
    inclusion = ConstraintSet(
        nct_id="N1",
        criteria_id="C1",
        polarity="inclusion",
        source_text="ANC >= 1500 /mm3.",
        constraints=[
            Constraint(
                kind="lab",
                label="absolute neutrophil count",
                comparator="ge",
                value=1500,
                unit="/mm3",
            )
        ],
    )
    evaluation = evaluate_constraint_set(inclusion, context)
    # 1500 >= 1500 must be matched, not violated by a comma-truncated 1.0.
    assert evaluation.matched_count == 1
    assert evaluation.violated_count == 0
    assert evaluation.constraint_signal > 0


def test_family_history_disease_does_not_satisfy_patient_condition_inclusion():
    profile = PatientProfile(
        patient_id="P1",
        demographics=Demographics(age_years=50),
        family_history=[
            _clinical_fact(
                "fh-1",
                "family_history",
                "mother: breast cancer",
                evidence_text="Mother had breast cancer.",
            )
        ],
    )
    context = build_patient_constraint_context(profile)
    # Family history must not be folded into the patient's own condition facts.
    assert not any(fact.kind == "condition" for fact in context.facts)
    inclusion = ConstraintSet(
        nct_id="N1",
        criteria_id="C1",
        polarity="inclusion",
        source_text="Patients with breast cancer.",
        constraints=[Constraint(kind="condition", label="breast cancer")],
    )
    evaluation = evaluate_constraint_set(inclusion, context)
    # A relative's disease must not score the patient as having it.
    assert evaluation.matched_count == 0
    assert evaluation.unknown_count == 1
    assert evaluation.constraint_signal == 0


def test_ecog_year_in_narrative_is_not_taken_as_performance_score():
    profile = PatientProfile(
        patient_id="P1",
        diagnostic_reports=[
            _clinical_fact(
                "dr-1",
                "diagnostic_report",
                "ECOG",
                description="ECOG performance status assessed 2021-03-15: grade 1",
                evidence_text="ECOG performance status assessed 2021-03-15: grade 1",
            )
        ],
    )
    context = build_patient_constraint_context(profile)
    # The year's '2' must not fabricate an ECOG=2 performance fact.
    perf_values = [f.value for f in context.facts if f.kind == "performance_status"]
    assert 2.0 not in perf_values


def test_ecog_score_still_extracted_from_common_phrasings():
    for text, expected in (
        ("ECOG 2", 2.0),
        ("ECOG performance status 2", 2.0),
        ("ECOG performance status of 1", 1.0),
    ):
        profile = PatientProfile(
            patient_id="P1",
            diagnostic_reports=[
                _clinical_fact(
                    "dr-1",
                    "diagnostic_report",
                    "ECOG",
                    description=text,
                    evidence_text=text,
                )
            ],
        )
        context = build_patient_constraint_context(profile)
        perf = [f for f in context.facts if f.kind == "performance_status"]
        assert perf, text
        assert perf[0].value == expected, text


# ==== src/trialmatchai/constraints/extraction.py ====


def test_lab_constraint_parses_thousands_separator_comma():
    # "1,500/mm3" must parse as 1500 with its unit intact, not truncate at the comma to 1/None.
    labs = [c for c in _lab_constraints("Absolute neutrophil count >= 1,500/mm3") if c.kind == "lab"]
    assert labs, "expected a lab constraint for comma-grouped ANC threshold"
    c = labs[0]
    assert c.comparator == "ge"
    assert c.value == 1500.0
    assert c.unit is not None and "mm" in c.unit


def test_lab_constraint_large_comma_grouped_platelet_threshold():
    labs = [c for c in _lab_constraints("platelets >= 100,000/uL") if c.kind == "lab"]
    assert any(c.value == 100000.0 and c.comparator == "ge" for c in labs), labs


def test_lab_constraint_trailing_comma_is_not_a_thousands_separator():
    # A comma not followed by digits must not be swallowed into the number.
    labs = [c for c in _lab_constraints("platelets >= 100, then reassess") if c.kind == "lab"]
    assert labs and labs[0].value == 100.0


def test_lab_constraint_decimal_and_plain_values_unchanged():
    labs = [c for c in _lab_constraints("Creatinine <= 1.5 mg/dL") if c.kind == "lab"]
    assert any(c.value == 1.5 and c.comparator == "le" for c in labs), labs
    plain = [c for c in _lab_constraints("ANC >= 1500/mm3") if c.kind == "lab"]
    assert any(c.value == 1500.0 for c in plain), plain


def test_biomarker_bare_gene_in_therapy_phrase_emits_no_constraint():
    # A gene name inside a drug/therapy phrase (no status token) must not become a biomarker requirement.
    assert _biomarker_constraints("Prior treatment with an EGFR tyrosine kinase inhibitor") == []


def test_biomarker_with_explicit_status_token_still_emitted():
    bm = _biomarker_constraints("EGFR mutated disease")
    assert any(c.kind == "biomarker" and c.label == "EGFR" and c.comparator == "mutated" for c in bm), bm


# ==== src/trialmatchai/entities/annotator.py ====



def test_build_entity_annotator_wires_lexical_reranker_by_default():
    # A concept store need not exist for the linker to be constructed (store falls back to
    # None); we only assert that build_entity_annotator wires the accept-gate's reranker,
    # matching link_corpus's _build_linker so RRF-#2 exact matches are promoted before gating.
    config = {
        "entity_extraction": {"backend": "regex"},
        "concept_linker": {"enabled": True},  # no db_path -> store None, linker still built
    }
    annotator = build_entity_annotator(config)
    assert annotator.linker is not None
    assert annotator.linker.reranker is lexical_reranker


def test_build_entity_annotator_reranker_disabled_when_rerank_not_lexical():
    config = {
        "entity_extraction": {"backend": "regex"},
        "concept_linker": {"enabled": True, "rerank": "none"},
    }
    annotator = build_entity_annotator(config)
    assert annotator.linker is not None
    assert annotator.linker.reranker is None


# ==== src/trialmatchai/entities/concept_sources.py ====
def test_parse_obo_keeps_last_term_before_trailing_typedef(tmp_path):
    """Regression: the final [Term] must survive a trailing [Typedef] stanza.

    Real OBO sources end with [Typedef] stanzas; previously parse_obo let the
    typedef's id:/name: overwrite the pending last term, so it was silently
    dropped at EOF (e.g. DOID:9997 'peripartum cardiomyopathy').
    """
    from trialmatchai.entities.concept_sources import parse_obo

    obo = tmp_path / "t.obo"
    obo.write_text(
        "[Term]\nid: DOID:12930\nname: earlier disease\n\n"
        "[Term]\nid: DOID:9997\nname: peripartum cardiomyopathy\n"
        'synonym: "PPCM" EXACT []\n\n'
        "[Typedef]\nid: IDO:0000664\nname: has_material_basis_in\n\n"
        "[Typedef]\nid: RO:is_a\nname: is a\n"
    )
    rows = dict(parse_obo(obo, "DOID:"))
    # the last real term is emitted with its own name/synonym...
    assert rows.get("DOID:9997") == ["peripartum cardiomyopathy", "PPCM"]
    assert "DOID:12930" in rows
    # ...and no non-Term stanza field leaks in as a concept.
    assert not any(cid.startswith(("IDO:", "RO:")) for cid in rows)


# ==== src/trialmatchai/interop/importers/omop.py ====
def test_omop_note_nlp_resolved_via_note_id_when_no_person_id(tmp_path):
    """Standard OMOP CDM NOTE_NLP has no person_id; it links to a patient only
    through note_id -> NOTE.person_id. Facts must be attributed via that join and
    not silently dropped (audit: note_nlp grouped by a nonexistent person_id)."""
    import pandas as pd

    from trialmatchai.interop.importers.omop import import_omop_extract

    omop = tmp_path / "omop"
    omop.mkdir()
    pd.DataFrame(
        [{"person_id": 1, "gender_source_value": "F", "year_of_birth": 1980}]
    ).to_csv(omop / "PERSON.csv", index=False)
    # Standard NOTE table carries the note_id -> person_id link.
    pd.DataFrame(
        [{"note_id": 100, "person_id": 1, "note_text": "clinical note", "note_type_concept_id": "x"}]
    ).to_csv(omop / "NOTE.csv", index=False)
    # Standard NOTE_NLP: NO person_id column, only note_id.
    pd.DataFrame(
        [
            {"note_nlp_id": 1, "note_id": 100, "note_nlp_concept_id": 10, "term_exists": "N", "snippet": "no metastasis"},
            {"note_nlp_id": 2, "note_id": 100, "note_nlp_concept_id": 11, "term_exists": "Y", "snippet": "diabetes present"},
        ]
    ).to_csv(omop / "NOTE_NLP.csv", index=False)
    pd.DataFrame(
        [
            {"concept_id": 10, "vocabulary_id": "SNOMED", "concept_code": "1", "concept_name": "Metastasis", "domain_id": "Condition"},
            {"concept_id": 11, "vocabulary_id": "SNOMED", "concept_code": "2", "concept_name": "Diabetes", "domain_id": "Condition"},
        ]
    ).to_csv(omop / "CONCEPT.csv", index=False)

    profile = import_omop_extract(omop)[0]
    by_label = {c.label: c for c in profile.conditions}
    # Without the note_id -> person_id resolution these would both be missing.
    assert "Metastasis" in by_label and "Diabetes" in by_label
    assert by_label["Metastasis"].negated is True
    assert by_label["Diabetes"].negated is False


def test_omop_note_nlp_falls_back_to_row_person_id(tmp_path):
    """A non-standard extract that puts person_id directly on NOTE_NLP (and has no
    NOTE table to join through) must still attribute the facts via that fallback."""
    import pandas as pd

    from trialmatchai.interop.importers.omop import import_omop_extract

    omop = tmp_path / "omop"
    omop.mkdir()
    pd.DataFrame(
        [{"person_id": 7, "gender_source_value": "M", "year_of_birth": 1975}]
    ).to_csv(omop / "PERSON.csv", index=False)
    pd.DataFrame(
        [{"person_id": 7, "note_nlp_concept_id": 10, "term_exists": "Y", "snippet": "diabetes"}]
    ).to_csv(omop / "NOTE_NLP.csv", index=False)
    pd.DataFrame(
        [{"concept_id": 10, "vocabulary_id": "SNOMED", "concept_code": "2", "concept_name": "Diabetes", "domain_id": "Condition"}]
    ).to_csv(omop / "CONCEPT.csv", index=False)

    profile = import_omop_extract(omop)[0]
    assert [c.label for c in profile.conditions] == ["Diabetes"]


# ==== src/trialmatchai/interop/utils.py ====
def test_parse_iso8601_age_years_handles_days_and_weeks_components():
    from trialmatchai.interop.utils import parse_iso8601_age_years

    # Pre-fix, any duration with a trailing D/W component failed fullmatch and
    # returned None, discarding even the parseable Y/M part.
    assert parse_iso8601_age_years("P70Y6M15D") == 70.54
    assert parse_iso8601_age_years("P70Y6M") == 70.5
    assert parse_iso8601_age_years("P15D") == 0.04
    assert parse_iso8601_age_years("P2W") == 0.04
    # Y-only and empty/invalid inputs are unaffected.
    assert parse_iso8601_age_years("P42Y") == 42.0
    assert parse_iso8601_age_years(None) is None
    assert parse_iso8601_age_years("not-a-duration") is None


# ==== src/trialmatchai/main.py ====
def test_rag_final_ranking_falls_back_to_shortlist_when_rag_wrote_nothing():
    """RAG-enabled run whose patient_narrative was empty writes no per-trial
    NCTxxxx.json, so load_trial_data yields []. The retrievable shortlist must be
    preserved (retrieval-scored), not discarded as an empty ranked_trials.json."""
    from trialmatchai.main import _rag_final_ranking

    semi_final_trials = [("NCT001", 0.9), ("NCT002", 0.7)]
    ranked = _rag_final_ranking(
        [],  # no per-trial eligibility outputs loaded
        semi_final_trials,
        first_level_scores={},
        second_level_scores={},
    )
    assert ranked == [
        {"TrialID": "NCT001", "Score": 0.9},
        {"TrialID": "NCT002", "Score": 0.7},
    ]


def test_rag_final_ranking_empty_when_no_trials_and_no_shortlist():
    """No per-trial outputs AND nothing retrieved -> genuinely empty result."""
    from trialmatchai.main import _rag_final_ranking

    assert _rag_final_ranking([], []) == []


def test_rag_final_ranking_uses_rank_trials_when_outputs_present():
    """When RAG did write per-trial eligibility outputs, rank_trials drives the
    ranking (EligibilityScore present) and the retrieval fallback is not used."""
    from trialmatchai.main import _rag_final_ranking

    trial_data = [
        {
            "TrialID": "NCT003",
            "Inclusion_Criteria_Evaluation": [{"Classification": "Met"}],
            "Exclusion_Criteria_Evaluation": [],
        }
    ]
    ranked = _rag_final_ranking(trial_data, [("NCT001", 0.9)])
    assert len(ranked) == 1
    assert ranked[0]["TrialID"] == "NCT003"
    assert "EligibilityScore" in ranked[0]


# ==== src/trialmatchai/matching/eligibility_reasoning_vllm.py ====
def test_vllm_trim_fit_check_uses_real_tokens_when_length_bucket_false():
    # With length_bucket=False the prompt-fit trim must still use a real token count. Previously
    # _token_length short-circuited to a len(prompt)//4 char estimate, so the keep-math mixed a
    # char estimate with the real token budget/criterion_ids and returned a prompt still far over
    # budget (~2000 real tok vs a 187 budget). The fix tokenizes the prompt directly for the check.
    from trialmatchai.matching.eligibility_reasoning_vllm import BatchTrialProcessorVLLM

    class _WhitespaceTokenizer:
        """One token per whitespace word; decode joins with spaces."""

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": text.split()}

        def decode(self, ids):
            return " ".join(ids)

    proc = BatchTrialProcessorVLLM.__new__(BatchTrialProcessorVLLM)  # bypass __init__ (no vLLM)
    proc.use_cot = False
    proc.no_think = False
    proc.llm = None
    proc.length_bucket = False  # the previously-broken path
    proc.tokenizer = _WhitespaceTokenizer()

    # single-char "criteria" words -> char estimate (len//4) diverges sharply from real word count
    long_criteria = " ".join("x" for _ in range(4000))

    proc._max_prompt_tokens = None
    overhead = len(proc.tokenizer(proc._format_prompt("", "patient description"))["input_ids"])
    proc._max_prompt_tokens = overhead + 50

    trimmed = proc._format_prompt(long_criteria, "patient description")
    real_len = len(proc.tokenizer(trimmed)["input_ids"])
    # Fixed code trims to fit using the real token count; the old char-estimate path returned a
    # prompt of ~2000 real tokens, far above the budget, so this assertion pins the regression.
    assert real_len <= proc._max_prompt_tokens
    assert real_len > overhead  # criteria were trimmed, not fully dropped


# ==== src/trialmatchai/matching/query_expansion.py ====
def test_query_expansion_strips_thinking_before_json_extraction():
    """query_expansion:183 — the CoT reasoning LoRA emits a <think>...<think> chain that
    echoes the JSON skeleton before the real answer; expand() must strip the thinking
    block so extract_json_object grabs the answer, not the placeholder skeleton (which
    would otherwise poison main_conditions or raise on an unbalanced brace)."""
    from trialmatchai.matching.query_expansion import QueryExpander

    # Reasoning preamble drafts the exact SYSTEM_PROMPT skeleton, then emits the answer.
    raw = (
        '<think> I should return {"main_conditions": ["PrimaryCondition", "Synonym1"], '
        '"other_conditions": [], "expanded_sentences": []} in this format '
        '<think> {"main_conditions": ["Non-small cell lung cancer", "NSCLC"], '
        '"other_conditions": ["EGFR mutation"], '
        '"expanded_sentences": ["Patient has NSCLC."]}'
    )

    class _FakeExpander(QueryExpander):
        def __init__(self):  # bypass model loading
            pass

        def _generate(self, narrative):
            return raw

    out = _FakeExpander().expand(["Patient has lung cancer."])
    # The real post-<think> answer wins, not the skeleton echoed inside the reasoning.
    assert out["main_conditions"] == ["Non-small cell lung cancer", "NSCLC"]
    assert out["other_conditions"] == ["EGFR mutation"]
    assert out["expanded_sentences"] == ["Patient has NSCLC."]
    assert "PrimaryCondition" not in out["main_conditions"]


def test_query_expansion_unbalanced_brace_inside_thinking_does_not_discard_answer():
    """query_expansion:183 — an unbalanced brace inside the <think> chain must not raise
    'Unbalanced JSON object' and silently discard a valid answer that follows it."""
    from trialmatchai.matching.query_expansion import QueryExpander

    raw = (
        '<think> hmm the object opens like { but I got cut off mid-draft '
        '<think> {"main_conditions": ["Melanoma"], "other_conditions": [], '
        '"expanded_sentences": []}'
    )

    class _FakeExpander(QueryExpander):
        def __init__(self):
            pass

        def _generate(self, narrative):
            return raw

    out = _FakeExpander().expand(["Patient has melanoma."])
    assert out["main_conditions"] == ["Melanoma"]  # not the empty fallback


# ==== src/trialmatchai/orchestration.py ====



def test_trial_needs_prepare_reprepares_edited_source(tmp_path):
    src_dir = tmp_path / "trials_jsons"
    pt = tmp_path / "processed_trials"
    src_dir.mkdir()
    pt.mkdir()
    source = src_dir / "NCT123.json"
    source.write_text('{"nct_id": "NCT123"}', encoding="utf-8")
    output = pt / "NCT123.json"
    output.write_text('{"nct_id": "NCT123"}', encoding="utf-8")

    # Fresh prepare: processed output written after the source -> up to date, skip.
    os.utime(source, ns=(1_000_000_000, 1_000_000_000))
    os.utime(output, ns=(2_000_000_000, 2_000_000_000))
    assert _trial_needs_prepare(source, pt) is False

    # In-place edit bumps the source mtime past the processed output -> re-prepare.
    os.utime(source, ns=(3_000_000_000, 3_000_000_000))
    assert _trial_needs_prepare(source, pt) is True

    # Missing / invalid processed output always re-prepares.
    output.unlink()
    assert _trial_needs_prepare(source, pt) is True


def test_prepare_corpus_reprepares_in_place_edited_trial(tmp_path, monkeypatch):
    src = tmp_path / "trials_jsons"
    src.mkdir()
    (src / "NCT1.json").write_text(json.dumps({"nct_id": "NCT1"}), encoding="utf-8")

    pt = tmp_path / "processed_trials"
    pt.mkdir()
    (pt / "NCT1.json").write_text(json.dumps({"nct_id": "NCT1"}), encoding="utf-8")

    # Simulate an in-place edit reusing the same NCT id: the source is now newer
    # than the already-valid processed output.
    os.utime(pt / "NCT1.json", ns=(1_000_000_000, 1_000_000_000))
    os.utime(src / "NCT1.json", ns=(2_000_000_000, 2_000_000_000))

    processed: list[str] = []
    monkeypatch.setattr(emb_mod, "build_embedder", lambda config: "EMB")
    monkeypatch.setattr(ent_mod, "build_entity_annotator", lambda config, embedder=None: None)
    monkeypatch.setattr(prep_mod, "prepare_trial_document", lambda doc, emb: dict(doc))
    monkeypatch.setattr(
        prep_mod,
        "prepare_criteria_documents",
        lambda doc, emb, entity_annotator=None: processed.append(doc["nct_id"]) or [],
    )
    monkeypatch.setattr(prep_mod, "write_prepared_trial", lambda row, folder: None)
    monkeypatch.setattr(prep_mod, "write_prepared_criteria", lambda rows, folder: 0)

    stats = prepare_corpus(
        {},
        trials_json_folder=src,
        processed_trials_folder=pt,
        processed_criteria_folder=tmp_path / "pc",
    )
    # The edited-in-place trial is re-embedded despite a valid processed output.
    assert processed == ["NCT1"]
    assert stats == {"total": 1, "prepared": 1, "skipped": 0, "failed": 0}


# ==== src/trialmatchai/registry/preparation.py ====
def test_to_iso_date_fills_partial_dates_deterministically():
    """Month- and year-precision CT.gov dates must resolve missing components
    to day/month 01 (deterministic) rather than being filled from today's date."""
    from trialmatchai.registry.preparation import _to_iso_date

    # Month-precision: missing day -> 01, not today's day-of-month.
    assert _to_iso_date("2021-03") == "2021-03-01"
    # Year-precision: missing month and day -> 01-01, not today's month/day.
    assert _to_iso_date("2021") == "2021-01-01"
    # Textual month/year also resolves deterministically.
    assert _to_iso_date("March 2021") == "2021-03-01"
    # Full dates and datetimes are preserved.
    assert _to_iso_date("2021-03-15") == "2021-03-15"
    assert _to_iso_date("2021-03-15T00:00:00") == "2021-03-15"
    # Empty and unparseable inputs still return None.
    assert _to_iso_date("") is None
    assert _to_iso_date(None) is None
    assert _to_iso_date("garbage") is None


# ==== src/trialmatchai/search/lancedb_backend.py ====
def test_lexical_score_short_term_does_not_match_within_longer_word():
    # Regression: a short normalized term must not raw-substring-match inside an
    # unrelated longer word and return the spurious 0.95 phrase-match score.
    from trialmatchai.search.lancedb_backend import _lexical_score

    # 'mm' used to match inside 'immune'; 'all' inside 'small'; 't' inside any word.
    assert _lexical_score("MM", "chronic inflammation and immune summary") == 0.0
    assert _lexical_score("ALL", "small cell lung cancer") == 0.0
    assert _lexical_score("T", "patient treatment text") == 0.0


def test_lexical_score_still_matches_whole_token_and_phrase():
    # The token-boundary fix must preserve genuine phrase/whole-token matches.
    from trialmatchai.search.lancedb_backend import _lexical_score

    assert _lexical_score("breast cancer", "patient breast cancer stage") == 0.95
    assert _lexical_score("melanoma", "metastatic melanoma stage") == 0.95
    assert _lexical_score("melanoma", "melanoma") == 1.0
    # A short term that IS a whole token still contributes via token overlap (> 0).
    assert _lexical_score("mm ecog", "mm patient text") > 0.0


# ==== src/trialmatchai/trec/topics.py ====
def test_import_topics_completes_partial_interrupted_import(tmp_path, monkeypatch):
    """A partial/interrupted topic import must be resumed, not treated as done.

    Simulates a crash after only the first of three topics was written: the next
    call must NOT short-circuit on "a profile exists" but re-run the idempotent
    loop and fill in the remaining topics (and overwrite the stale sentinel).
    """
    from trialmatchai.trec import topics as topics_mod

    expected = {
        "trec-2021_1": "A 30-year-old man with cough.",
        "trec-2021_2": "A 45-year-old woman with fever.",
        "trec-2021_3": "A 5-year-old boy with rash.",
    }
    # Bypass network/XML: resolve the track's topics deterministically.
    monkeypatch.setattr(
        topics_mod, "load_track_topics", lambda track, trec_dir: dict(expected)
    )

    profile_dir = tmp_path / "profiles"
    summary_dir = tmp_path / "summaries"
    profile_dir.mkdir(parents=True)
    summary_dir.mkdir(parents=True)
    # Only the first topic made it to disk before the (simulated) kill.
    (profile_dir / "trec-2021_1.json").write_text("SENTINEL", encoding="utf-8")
    (summary_dir / "trec-2021_1.json").write_text("SENTINEL", encoding="utf-8")

    count = topics_mod.import_topics(
        "21",
        trec_dir=tmp_path / "trec",
        profile_dir=profile_dir,
        summary_dir=summary_dir,
    )

    assert count == 3
    for pid in expected:
        assert (profile_dir / f"{pid}.json").exists()
        assert (summary_dir / f"{pid}.json").exists()
    # The stale partial file was overwritten with a real profile (valid JSON).
    assert (profile_dir / "trec-2021_1.json").read_text(encoding="utf-8").lstrip().startswith("{")


def test_import_topics_skips_only_when_every_topic_present(tmp_path, monkeypatch):
    """A fully-complete import is still skipped idempotently (no rewrite)."""
    from trialmatchai.trec import topics as topics_mod

    expected = {
        "trec-2021_1": "A 30-year-old man with cough.",
        "trec-2021_2": "A 45-year-old woman with fever.",
    }
    monkeypatch.setattr(
        topics_mod, "load_track_topics", lambda track, trec_dir: dict(expected)
    )

    profile_dir = tmp_path / "profiles"
    summary_dir = tmp_path / "summaries"

    first = topics_mod.import_topics(
        "21",
        trec_dir=tmp_path / "trec",
        profile_dir=profile_dir,
        summary_dir=summary_dir,
    )
    assert first == 2

    # Mark one file so we can prove the skip path does not rewrite it.
    sentinel = "KEEP-ME"
    (profile_dir / "trec-2021_1.json").write_text(sentinel, encoding="utf-8")

    second = topics_mod.import_topics(
        "21",
        trec_dir=tmp_path / "trec",
        profile_dir=profile_dir,
        summary_dir=summary_dir,
    )
    assert second == 2
    assert (profile_dir / "trec-2021_1.json").read_text(encoding="utf-8") == sentinel


# ==== src/trialmatchai/utils/file_utils.py ====



def test_orphaned_json_temp_not_matched_by_artifact_glob(tmp_path, monkeypatch):
    # Simulate a crash in the fsync->os.replace window: os.replace is a no-op,
    # so the temp file survives as an orphan and the target is never created.
    monkeypatch.setattr(os, "replace", lambda *a, **k: None)

    target = tmp_path / "patient1.json"
    write_json_file({"patient_id": "P1"}, str(target))

    orphans = list(tmp_path.iterdir())
    assert orphans, "expected an orphaned temp file to remain"
    assert all(p.name.startswith(".tmp-") for p in orphans)
    # The crux: resume/enumeration globs must NOT see the orphan as an artifact.
    assert list(tmp_path.glob("*.json")) == []
    assert not target.exists()


def test_orphaned_text_temp_not_matched_by_txt_or_json_glob(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "replace", lambda *a, **k: None)

    target = tmp_path / "note.txt"
    write_text_file(["a", "b"], str(target))

    assert list(tmp_path.glob("*.txt")) == []
    assert list(tmp_path.glob("*.json")) == []
