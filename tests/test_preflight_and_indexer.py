from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from Matcher.config.config_loader import load_config
from Matcher.search import InMemorySearchBackend
from Matcher.services import preflight
from Matcher.services.preflight import run_preflight_checks


def _base_config(tmp_path):
    patients = tmp_path / "patients"
    patients.mkdir()
    trials = tmp_path / "trials"
    trials.mkdir()
    search_db = tmp_path / "search"
    search_db.mkdir()
    return {
        "paths": {
            "patients_dir": str(patients),
            "trials_json_folder": str(trials),
            "output_dir": str(tmp_path / "results"),
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
        "cot_backend": "default",
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
    cfg["cot_backend"] = "vllm"
    Path(cfg["model"]["cot_adapter_path"]).mkdir(parents=True)
    Path(cfg["model"]["reranker_adapter_path"]).mkdir(parents=True)
    monkeypatch.setattr(preflight.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(preflight.torch.cuda, "is_available", lambda: True)

    issues = run_preflight_checks(cfg, require_models=True)

    assert issues == [
        "cot_backend=vllm requires the GPU extra "
        "(`uv sync --extra gpu`) or the Docker worker image."
    ]


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
                    "patients_dir": "patients",
                    "output_dir": "results",
                    "trials_json_folder": "trials",
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
    indexer_path = Path(__file__).resolve().parents[1] / "utils/Indexer/index_criteria.py"
    spec = importlib.util.spec_from_file_location("index_criteria", indexer_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    processed = tmp_path / "processed_criteria"
    trial_dir = processed / "N1"
    trial_dir.mkdir(parents=True)
    (trial_dir / "C1.json").write_text(
        json.dumps({"criteria_id": "C1", "nct_id": "N1", "criterion": "cancer"})
    )
    backend = InMemorySearchBackend()
    criteria_indexer = module.CriteriaIndexer(
        backend=backend,
        processed_file=tmp_path / "processed_ids.txt",
    )

    docs, completed = criteria_indexer.load_docs(processed, recreate=True)

    assert docs[0]["criteria_id"] == "C1"
    assert completed == {"N1"}
