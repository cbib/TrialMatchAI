from __future__ import annotations

import json
from pathlib import Path

from trialmatchai.cli.index_data import _load_nested_json_folder
from trialmatchai.config.config_loader import load_config
from trialmatchai.search import InMemorySearchBackend
from trialmatchai.services import preflight
from trialmatchai.services.preflight import run_preflight_checks


def _base_config(tmp_path):
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    trials = tmp_path / "trials"
    trials.mkdir()
    search_db = tmp_path / "search"
    search_db.mkdir()
    return {
        "paths": {
            "trials_json_folder": str(trials),
            "output_dir": str(tmp_path / "results"),
        },
        "patient_inputs": {
            "profile_dir": str(profiles),
            "summary_dir": str(tmp_path / "summaries"),
        },
        "search_backend": {
            "backend": "lancedb",
            "db_path": str(search_db),
            "trials_table": "trials",
            "criteria_table": "criteria",
        },
        "model": {
            "cot_adapter_path": str(tmp_path / "models" / "cot"),
            "reranker_adapter_path": str(tmp_path / "models" / "reranker"),
        },
    }


def _entity_config(tmp_path):
    cfg = _base_config(tmp_path)
    schema = tmp_path / "schema.yaml"
    schema.write_text("version: 1\nentities: []\n")
    cfg["entity_extraction"] = {
        "backend": "gliner2",
        "schema_path": str(schema),
    }
    cfg["concept_linker"] = {
        "enabled": True,
        "db_path": str(tmp_path / "missing-concepts"),
    }
    return cfg


class FakeSearchBackend:
    def __init__(self, issues: list[str] | None = None):
        self.issues = issues or []

    def health(self, *, require_tables: bool = False):
        return self.issues if require_tables else []


def test_preflight_passes_for_required_paths_and_tables(tmp_path):
    cfg = _base_config(tmp_path)
    issues = run_preflight_checks(
        cfg,
        search_backend=FakeSearchBackend(),
        require_patient_inputs=True,
        require_trials_json=True,
        require_search_tables=True,
    )

    assert issues == []


def test_preflight_reports_missing_search_tables(tmp_path):
    cfg = _base_config(tmp_path)
    issues = run_preflight_checks(
        cfg,
        search_backend=FakeSearchBackend(["Missing LanceDB tables: criteria"]),
        require_search_tables=True,
    )

    assert issues == ["Missing LanceDB tables: criteria"]


def test_preflight_reports_missing_search_db_path(tmp_path):
    cfg = _base_config(tmp_path)
    cfg["search_backend"]["db_path"] = str(tmp_path / "missing-search")

    issues = run_preflight_checks(
        cfg,
        search_backend=FakeSearchBackend(),
        require_search_tables=True,
    )

    assert issues == [
        f"search_backend.db_path does not exist: {tmp_path / 'missing-search'}"
    ]


def test_preflight_reports_missing_vllm_extra(tmp_path, monkeypatch):
    cfg = _base_config(tmp_path)
    Path(cfg["model"]["cot_adapter_path"]).mkdir(parents=True)
    Path(cfg["model"]["reranker_adapter_path"]).mkdir(parents=True)
    monkeypatch.setattr(preflight.importlib.util, "find_spec", lambda name: None)

    issues = run_preflight_checks(cfg, require_models=True)

    assert issues == ["vLLM is required (`uv sync --extra llm --extra gpu`)."]


def test_preflight_allows_cpu_smoke_when_llm_stages_disabled(tmp_path, monkeypatch):
    cfg = _base_config(tmp_path)
    cfg["LLM_reranker"] = {"enabled": False}
    cfg["rag"] = {"enabled": False}
    cfg["use_cot_reasoning"] = False
    monkeypatch.setattr(preflight.importlib.util, "find_spec", lambda name: None)

    issues = run_preflight_checks(cfg, require_models=True)

    assert issues == []


def test_preflight_allows_transformers_cpu_llm_backend(tmp_path, monkeypatch):
    cfg = _base_config(tmp_path)
    cfg["LLM_reranker"] = {"enabled": True, "backend": "transformers"}
    cfg["rag"] = {"enabled": True, "backend": "transformers"}
    cfg["use_cot_reasoning"] = True
    monkeypatch.setattr(
        preflight.importlib.util,
        "find_spec",
        lambda name: object() if name in {"torch", "transformers"} else None,
    )

    issues = run_preflight_checks(cfg, require_models=True)

    assert issues == []


def test_preflight_reports_missing_entity_extra(tmp_path, monkeypatch):
    cfg = _entity_config(tmp_path)
    monkeypatch.setattr(preflight.importlib.util, "find_spec", lambda name: None)

    issues = run_preflight_checks(cfg, require_models=True)

    assert (
        "entity_extraction.backend=gliner2 requires the entity extra "
        "(`uv sync --extra entity`)."
    ) in issues


def test_main_config_resolves_search_paths(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    schema = tmp_path / "schema.yaml"
    schema.write_text("version: 1\nentities: []\n")
    config_path.write_text(
        json.dumps(
            {
                "paths": {
                    "output_dir": "results",
                    "trials_json_folder": "trials",
                },
                "patient_inputs": {
                    "profile_dir": "patients/profiles",
                    "summary_dir": "patients/summaries",
                },
                "search_backend": {
                    "backend": "lancedb",
                    "db_path": "search",
                    "trials_table": "trials",
                    "criteria_table": "criteria",
                },
                "entity_extraction": {
                    "backend": "regex",
                    "schema_path": str(schema),
                },
                "concept_linker": {
                    "enabled": False,
                    "db_path": "concepts",
                },
                "model": {
                    "base_model": "m",
                    "quantization": {},
                    "cot_adapter_path": "models/cot",
                    "reranker_model_path": "r",
                    "reranker_adapter_path": "models/reranker",
                },
                "tokenizer": {},
                "global": {"device": "cpu"},
                "embedder": {},
                "cot": {},
                "LLM_reranker": {},
                "search": {},
                "rag": {},
                "vllm": {},
            }
        )
    )
    search_env = tmp_path / "search-env"
    monkeypatch.setenv("TRIALMATCHAI_SEARCH_DB_PATH", str(search_env))

    cfg = load_config(config_path)

    assert cfg["search_backend"]["db_path"] == str(search_env.resolve())


def test_indexer_loads_prepared_criteria_docs(tmp_path):
    processed = tmp_path / "processed_criteria"
    trial_dir = processed / "N1"
    trial_dir.mkdir(parents=True)
    (trial_dir / "C1.json").write_text(
        json.dumps({"criteria_id": "C1", "nct_id": "N1", "criterion": "cancer"})
    )
    docs = _load_nested_json_folder(processed)
    backend = InMemorySearchBackend()
    count = backend.replace_criteria_for_trials(["N1"], docs)

    assert docs[0]["criteria_id"] == "C1"
    assert count == 1
    assert backend.criteria[0]["nct_id"] == "N1"
