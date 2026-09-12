from __future__ import annotations

import json
from pathlib import Path

import pytest

from trialmatchai.matching.assessment import assessment_run_info, assessment_settings, match_controls_current, match_is_complete
from trialmatchai.matching.eligibility_base import BaseTrialProcessor
from trialmatchai.utils.file_utils import write_json_file, write_text_file


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("cot", [True, False])
def test_preflight_checks_assessment_even_without_cot(tmp_path, monkeypatch, enabled, cot):
    from trialmatchai.services import preflight

    checked = []
    monkeypatch.setattr(preflight, "check_hf_access", lambda models: checked.extend(models) or [])
    monkeypatch.setattr(preflight.importlib.util, "find_spec", lambda name: object())
    config = {
        "paths": {"output_dir": str(tmp_path)}, "LLM_reranker": {"enabled": False},
        "rag": {"enabled": enabled, "backend": "transformers"}, "use_cot_reasoning": cot,
        "model": {"base_model": "test/assessment"},
    }
    assert preflight.run_preflight_checks(config, require_models=True) == []
    assert checked == (["test/assessment"] if enabled else [])


def test_default_assessment_remains_enabled_with_direct_json_prompt():
    from trialmatchai.main import _rag_enabled

    assert _rag_enabled({"use_cot_reasoning": False})
    proc = BaseTrialProcessor()
    proc.use_cot = False
    prompt = proc._format_prompt("Adults with lung cancer", "Adult with lung cancer")
    assert "Inclusion_Criteria_Evaluation" in prompt and "Exclusion_Criteria_Evaluation" in prompt
    assert "Final Decision" in prompt
    assert "chain of thoughts" not in prompt


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    import trialmatchai.main as main
    import trialmatchai.models.embedding as embedding
    from trialmatchai.interop.models import PatientProfile
    from trialmatchai.matching import eligibility_reasoning_transformers as backend
    from trialmatchai.matching import eligibility_reasoning_vllm as vllm_backend
    from trialmatchai.models.llm import vllm_loader

    config = {
        "paths": {"output_dir": str(tmp_path / "results"), "trials_json_folder": str(tmp_path / "trials")},
        "patient_inputs": {"profile_dir": str(tmp_path / "profiles")},
        "search": {"mode": "bm25"}, "LLM_reranker": {"enabled": False},
        "rag": {"enabled": True, "backend": "transformers"}, "use_cot_reasoning": False,
        "model": {"base_model": "test-model"}, "reporting": {"emit_html": True},
    }
    profile = PatientProfile.model_validate({"patient_id": "P1", "demographics": {}})
    summary = {"patient_id": "P1", "main_conditions": ["lung cancer"], "other_conditions": [],
               "patient_narrative": ["Adult with lung cancer"], "age": "all", "gender": "all"}
    write_json_file(profile.model_dump(mode="json"), str(tmp_path / "profiles/P1.json"))
    write_json_file({"eligibility_criteria": "Inclusion: Adults with lung cancer"}, str(tmp_path / "trials/NCT1.json"))
    monkeypatch.setattr(main, "build_search_backend", lambda cfg: object())
    monkeypatch.setattr(main, "run_preflight_checks", lambda *a, **kw: [])
    monkeypatch.setattr(main, "_load_patient_inputs", lambda cfg: [(profile, summary)])
    monkeypatch.setattr(embedding, "build_embedder", lambda cfg: object())
    monkeypatch.setattr(main, "build_entity_annotator", lambda *a, **kw: None)
    monkeypatch.setattr(main, "SecondStageRetriever", lambda **kw: object())
    monkeypatch.setattr(main, "run_first_level_search", lambda *a, **kw: (["NCT1"], ["lung cancer"], [], summary["patient_narrative"], {"NCT1": 0.7}))

    def second(*args):
        path = str(Path(args[0]) / "top_trials.txt")
        write_text_file(["NCT1"], path)
        return [("NCT1", 0.7)], path, {"NCT1": 0.7}

    monkeypatch.setattr(main, "run_second_level_search", second)
    calls = []

    class FakeProcessor(BaseTrialProcessor):
        def __init__(self, **kwargs):
            self.use_cot = kwargs["use_cot"]

        def _process_batch(self, items, output_folder):
            calls.append(self.use_cot)
            for item in items:
                self._save_outputs(item["nct_id"], json.dumps({
                    "Final Decision": "Eligible",
                    "Inclusion_Criteria_Evaluation": [{"Criterion": "Lung cancer", "Classification": "Met", "Justification": "Recorded."}],
                    "Exclusion_Criteria_Evaluation": [],
                }), output_folder)

    monkeypatch.setattr(backend, "BatchTrialProcessorTransformers", FakeProcessor)
    monkeypatch.setattr(vllm_backend, "BatchTrialProcessorVLLM", FakeProcessor)
    monkeypatch.setattr(vllm_loader, "load_vllm_engine", lambda **kw: (None, None, None))
    return main, config, calls, tmp_path / "results/P1"


@pytest.mark.parametrize("backend_name", ["transformers", "vllm"])
def test_full_pipeline_assesses_when_cot_is_off(pipeline, backend_name):
    main, config, calls, output = pipeline
    config["rag"]["backend"] = backend_name
    assert main.main_pipeline(config=config) == 0
    assert calls == [False]
    ranked = json.loads((output / "ranked_trials.json").read_text())
    assert ranked["Run"]["mode"] == "eligibility_assessment"
    assert ranked["Run"]["assessment_status"] == "outputs_available"
    assert ranked["Run"]["assessed_trial_ids"] == ["NCT1"]
    assert '"final_decision": "Eligible"' in (output / "report.html").read_text()


def test_mode_changes_invalidate_resume_and_hide_old_assessments(pipeline):
    from trialmatchai.interop.exporters.html_report import profile_to_model
    from trialmatchai.matching.trial_ranker import rerank_patient_dir
    from trialmatchai.orchestration import count_pending

    main, config, calls, output = pipeline
    assert main.main_pipeline(config=config) == 0
    assert count_pending(config) == (0, 1)
    (output / "NCT1.txt").write_text("<think>old reasoning</think>")
    config["rag"]["enabled"] = False
    assert count_pending(config) == (1, 0)
    assert main.main_pipeline(config=config, resume=True) == 0
    assert calls == [False]  # Assessment was not invoked again.
    assert (output / "NCT1.json").exists()  # Old output remains but cannot leak into this run.
    payload = json.loads((output / "ranked_trials.json").read_text())
    assert payload["Run"]["mode"] == "retrieval_only"
    assert payload["Run"]["assessment_status"] == "disabled"
    report = profile_to_model(output)
    trial = report["trials"][0]
    assert trial["final_decision"] is None and trial["inclusion"] == [] and trial["cot"] is None
    assert not trial["reasoning_available"]
    assert rerank_patient_dir(str(output)) == 0
    assert json.loads((output / "ranked_trials.json").read_text()) == payload
    config["rag"]["enabled"] = True
    assert count_pending(config) == (1, 0)
    assert main.main_pipeline(config=config, resume=True) == 0
    assert calls == [False, False]  # Re-enabling recomputes, including per-trial outputs.


def test_cot_style_change_recomputes_per_trial_outputs(pipeline):
    main, config, calls, output = pipeline
    assert main.main_pipeline(config=config) == 0
    config["use_cot_reasoning"] = True
    assert not match_controls_current(output / "ranked_trials.json", config)
    assert main.main_pipeline(config=config, resume=True) == 0
    assert calls == [False, True]
    assert main.main_pipeline(config=config, resume=True) == 0
    assert calls == [False, True]  # Unchanged settings preserve resume.


def test_missing_or_error_assessments_produce_retrieval_only_results():
    from trialmatchai.main import _rag_final_ranking

    data = [{"TrialID": "NCT1", "error": "invalid_json_response"}]
    assert _rag_final_ranking(data, [("NCT1", 0.7)]) == [{"TrialID": "NCT1", "Score": 0.7}]
    info = assessment_run_info({}, data, {"NCT1"})
    assert info["mode"] == "retrieval_only" and info["assessment_status"] == "unavailable"
    assert info["assessment"]["enabled"] is True


def test_legacy_results_require_recomputation_for_new_control_semantics(tmp_path):
    path = tmp_path / "ranked_trials.json"
    path.write_text('{"RankedTrials": []}')
    assert not match_controls_current(path, {})
    write_json_file({"RankedTrials": [], "Run": assessment_run_info({}, [], set())}, str(path))
    assert match_controls_current(path, {})
    assert not match_controls_current(path, {"use_cot_reasoning": False})


def test_reranking_assessed_results_preserves_run_metadata(tmp_path):
    from trialmatchai.matching.trial_ranker import rerank_patient_dir

    data = {"TrialID": "NCT1", "Final Decision": "Eligible", "Inclusion_Criteria_Evaluation": []}
    run = assessment_run_info({}, [data], {"NCT1"})
    write_json_file(data, str(tmp_path / "NCT1.json"))
    write_json_file({"RankedTrials": [{"TrialID": "NCT1", "Score": 1}], "Run": run}, str(tmp_path / "ranked_trials.json"))
    assert rerank_patient_dir(str(tmp_path)) == 1
    assert json.loads((tmp_path / "ranked_trials.json").read_text())["Run"] == run


def test_match_signature_tracks_assessment_controls():
    from trialmatchai.orchestration import _match_signature

    assert _match_signature({}) != _match_signature({"rag": {"enabled": False}})
    assert _match_signature({}) != _match_signature({"use_cot_reasoning": False})
    assert _match_signature({}) != _match_signature({"rag": {"max_trials_rag": 5}})
    assert assessment_settings({})["enabled"] is True


def test_shortlist_budget_change_invalidates_resume(pipeline):
    from trialmatchai.orchestration import count_pending

    main, config, calls, output = pipeline
    assert main.main_pipeline(config=config) == 0
    config["rag"]["max_trials_rag"] = 1
    assert not match_controls_current(output / "ranked_trials.json", config)
    assert count_pending(config) == (1, 0)
    assert main.main_pipeline(config=config, resume=True) == 0
    assert calls == [False, False]
    assert json.loads((output / "ranked_trials.json").read_text())["Run"]["assessment"]["max_trials_rag"] == 1


@pytest.mark.parametrize("narrative", [[], ["  ", "\n"]])
def test_aborted_assessment_cannot_reuse_prior_verdicts(pipeline, narrative):
    from trialmatchai.interop.exporters.html_report import profile_to_model
    from trialmatchai.orchestration import count_pending

    main, config, calls, output = pipeline
    assert main.main_pipeline(config=config) == 0
    config["use_cot_reasoning"] = True
    summary = main._load_patient_inputs(config)[0][1]
    original_narrative = summary["patient_narrative"]
    summary["patient_narrative"] = narrative
    assert main.main_pipeline(config=config, resume=True) == 0
    assert calls == [False]
    payload = json.loads((output / "ranked_trials.json").read_text())
    assert payload["RankedTrials"] == [{"TrialID": "NCT1", "Score": 0.7}]
    assert payload["Run"]["assessment_status"] == "unavailable"
    assert payload["Run"]["assessed_trial_ids"] == []
    assert (output / "NCT1.json").exists()
    assert not profile_to_model(output)["trials"][0]["reasoning_available"]
    assert count_pending(config) == (1, 0)
    summary["patient_narrative"] = original_narrative
    assert main.main_pipeline(config=config, resume=True) == 0
    assert calls == [False, True]
    assert count_pending(config) == (0, 1)


def test_failed_forced_write_cannot_revive_old_output(pipeline, monkeypatch):
    from trialmatchai.matching import eligibility_base
    from trialmatchai.orchestration import count_pending

    main, config, calls, output = pipeline
    assert main.main_pipeline(config=config) == 0
    previous = (output / "NCT1.json").read_bytes()
    config["use_cot_reasoning"] = True

    def fail_write(*args, **kwargs):
        raise OSError("simulated write failure")

    with monkeypatch.context() as patch:
        patch.setattr(eligibility_base, "write_json_file", fail_write)
        assert main.main_pipeline(config=config, resume=True) == 0
    payload = json.loads((output / "ranked_trials.json").read_text())
    assert payload["Run"]["assessment_status"] == "unavailable"
    assert payload["RankedTrials"] == [{"TrialID": "NCT1", "Score": 0.7}]
    assert (output / "NCT1.json").read_bytes() == previous
    assert count_pending(config) == (1, 0)
    assert not list(output.glob(".assessment-*"))
    assert main.main_pipeline(config=config, resume=True) == 0
    assert calls == [False, True, True]
    assert count_pending(config) == (0, 1)


@pytest.mark.parametrize("response", ['{"error": "transient"}', '{}', 'not JSON'])
def test_unavailable_assessments_are_retried(pipeline, monkeypatch, response):
    from trialmatchai.orchestration import count_pending

    main, config, calls, output = pipeline
    save = BaseTrialProcessor._save_outputs
    with monkeypatch.context() as patch:
        patch.setattr(BaseTrialProcessor, "_save_outputs", lambda self, tid, text, folder: save(self, tid, response, folder))
        assert main.main_pipeline(config=config) == 0
    assert match_controls_current(output / "ranked_trials.json", config)
    assert not match_is_complete(output / "ranked_trials.json", config)
    assert count_pending(config) == (1, 0)
    assert main.main_pipeline(config=config, resume=True) == 0
    assert calls == [False, False]
    assert count_pending(config) == (0, 1)


def test_partial_assessment_retries_only_missing_trial(pipeline, monkeypatch):
    from trialmatchai.orchestration import count_pending

    main, config, calls, output = pipeline
    write_json_file({"eligibility_criteria": "Adults"}, str(Path(config["paths"]["trials_json_folder"]) / "NCT2.json"))

    def second(*args):
        path = str(Path(args[0]) / "top_trials.txt")
        write_text_file(["NCT1", "NCT2"], path)
        return [("NCT1", 0.7), ("NCT2", 0.6)], path, {"NCT1": 0.7, "NCT2": 0.6}

    monkeypatch.setattr(main, "run_second_level_search", second)
    save = BaseTrialProcessor._save_outputs
    generated = []

    def save_with_failure(self, tid, text, folder):
        generated.append(tid)
        if tid == "NCT2" and generated.count(tid) == 1:
            text = '{"error": "transient"}'
        save(self, tid, text, folder)

    monkeypatch.setattr(BaseTrialProcessor, "_save_outputs", save_with_failure)
    assert main.main_pipeline(config=config) == 0
    assert json.loads((output / "ranked_trials.json").read_text())["Run"]["assessment_status"] == "partial"
    completed_mtime = (output / "NCT1.json").stat().st_mtime_ns
    assert count_pending(config) == (1, 0)
    assert main.main_pipeline(config=config, resume=True) == 0
    assert generated.count("NCT1") == 1 and generated.count("NCT2") == 2
    assert (output / "NCT1.json").stat().st_mtime_ns == completed_mtime
    assert count_pending(config) == (0, 1)


def test_deleted_assessment_is_pending_and_recomputed(pipeline):
    from trialmatchai.orchestration import count_pending

    main, config, calls, output = pipeline
    assert main.main_pipeline(config=config) == 0
    (output / "NCT1.json").unlink()
    assert count_pending(config) == (1, 0)
    assert main.main_pipeline(config=config, resume=True) == 0
    assert calls == [False, False]
    assert count_pending(config) == (0, 1)


def test_report_marks_partial_assessment_and_ignores_unassociated_outputs():
    from trialmatchai.interop.exporters.html_report import build_report_model

    data = {"TrialID": "NCT1", "Final Decision": "Eligible"}
    run = assessment_run_info({}, [data], {"NCT1", "NCT2"})
    model = build_report_model(
        patient_summary={}, ranked={"RankedTrials": [{"TrialID": "NCT1"}, {"TrialID": "NCT2"}], "Run": run},
        eligibility_by_id={"NCT1": data, "NCT2": {"Final Decision": "Eligible"}},
        meta_by_id={}, cot_by_id={"NCT2": "stale text"}, generated_at="x",
    )
    assert model["run"]["assessment_status"] == "partial"
    assert model["trials"][0]["reasoning_available"]
    assert not model["trials"][1]["reasoning_available"]
    assert model["trials"][1]["final_decision"] is None
    assert model["trials"][1]["cot"] is None


def test_report_downgrades_availability_when_saved_assessment_is_missing():
    from trialmatchai.interop.exporters.html_report import build_report_model

    run = assessment_run_info({}, [{"TrialID": "NCT1", "Final Decision": "Eligible"}], {"NCT1"})
    model = build_report_model(
        patient_summary={}, ranked={"RankedTrials": [{"TrialID": "NCT1"}], "Run": run},
        eligibility_by_id={}, meta_by_id={}, generated_at="x",
    )
    assert model["run"]["mode"] == "retrieval_only"
    assert model["run"]["assessment_status"] == "unavailable"
