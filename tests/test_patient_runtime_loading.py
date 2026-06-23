from __future__ import annotations

import json

from trialmatchai.interop.models import PatientProfile
from trialmatchai.main import _load_patient_inputs


def test_runtime_loads_profiles_from_configured_directory(tmp_path):
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    summaries = tmp_path / "summaries"
    summaries.mkdir()
    (profiles / "p1.json").write_text(
        json.dumps({"patient_id": "p1", "demographics": {}}),
        encoding="utf-8",
    )
    (summaries / "p1.json").write_text(
        json.dumps(
            {
                "patient_id": "p1",
                "main_conditions": ["melanoma"],
                "other_conditions": [],
                "patient_narrative": ["Patient has melanoma."],
                "age": "all",
                "gender": "all",
            }
        ),
        encoding="utf-8",
    )

    loaded = _load_patient_inputs(
        {
            "patient_inputs": {
                "profile_dir": str(profiles),
                "summary_dir": str(summaries),
            },
        }
    )

    assert loaded[0][0].patient_id == "p1"
    assert loaded[0][1]["main_conditions"] == ["melanoma"]


def test_runtime_requires_canonical_profiles(tmp_path):
    profiles = tmp_path / "profiles"
    summaries = tmp_path / "summaries"

    loaded = _load_patient_inputs(
        {
            "patient_inputs": {
                "profile_dir": str(profiles),
                "summary_dir": str(summaries),
            },
        }
    )

    assert loaded == []


def test_main_pipeline_returns_nonzero_when_all_patients_fail(tmp_path, monkeypatch):
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
    }
    profile = PatientProfile.model_validate({"patient_id": "p1", "demographics": {}})
    summary = {
        "patient_id": "p1",
        "main_conditions": ["lung cancer"],
        "other_conditions": [],
        "patient_narrative": ["Patient has lung cancer."],
        "age": "all",
        "gender": "all",
    }

    monkeypatch.setattr(main_module, "load_config", lambda config_path=None: config)
    monkeypatch.setattr(main_module, "run_preflight_checks", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        main_module.LanceDBSearchBackend,
        "from_config",
        classmethod(lambda cls, cfg: _Backend()),
    )
    monkeypatch.setattr(embedding_module, "build_embedder", lambda cfg: object())
    monkeypatch.setattr(main_module, "build_entity_annotator", lambda cfg, embedder: None)
    monkeypatch.setattr(
        main_module,
        "_load_patient_inputs",
        lambda cfg: [(profile, summary)],
    )

    def _fail_first_level(*args, **kwargs):
        raise RuntimeError("forced patient failure")

    monkeypatch.setattr(main_module, "run_first_level_search", _fail_first_level)

    assert main_module.main_pipeline("config.json") == 1
