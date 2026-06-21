from __future__ import annotations

import json
import importlib.util
from pathlib import Path

from Matcher.services import preflight
from Matcher.services.preflight import run_preflight_checks

ES_CONFIG_PATH = Path(__file__).resolve().parents[1] / "utils/Indexer/es_config.py"
spec = importlib.util.spec_from_file_location("indexer_es_config", ES_CONFIG_PATH)
assert spec and spec.loader
indexer_es_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(indexer_es_config)
load_config = indexer_es_config.load_config


class HealthyIndices:
    def __init__(self, existing: set[str]):
        self.existing = existing

    def exists(self, index: str) -> bool:
        return index in self.existing


class FakeES:
    def __init__(self, *, healthy: bool = True, existing: set[str] | None = None):
        self.healthy = healthy
        self.indices = HealthyIndices(existing or set())

    def ping(self) -> bool:
        return self.healthy


def _base_config(tmp_path):
    cert = tmp_path / "ca.crt"
    cert.write_text("cert")
    patients = tmp_path / "patients"
    patients.mkdir()
    trials = tmp_path / "trials"
    trials.mkdir()
    return {
        "paths": {
            "patients_dir": str(patients),
            "trials_json_folder": str(trials),
            "output_dir": str(tmp_path / "results"),
            "docker_certs": str(cert),
        },
        "elasticsearch": {
            "host": "https://localhost:9200",
            "index_trials": "clinical_trials",
            "index_trials_eligibility": "trials_eligibility",
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


def test_preflight_passes_for_required_paths_and_indices(tmp_path):
    cfg = _base_config(tmp_path)
    issues = run_preflight_checks(
        cfg,
        es_client=FakeES(
            existing={"clinical_trials", "trials_eligibility"},
        ),
        require_patient_inputs=True,
        require_trials_json=True,
        require_indices=True,
    )

    assert issues == []


def test_preflight_reports_missing_indices(tmp_path):
    cfg = _base_config(tmp_path)
    issues = run_preflight_checks(
        cfg,
        es_client=FakeES(existing={"clinical_trials"}),
        require_indices=True,
    )

    assert issues == ["Missing Elasticsearch indices: trials_eligibility"]


def test_preflight_reports_unreachable_elasticsearch(tmp_path):
    cfg = _base_config(tmp_path)
    issues = run_preflight_checks(cfg, es_client=FakeES(healthy=False))

    assert issues == ["Elasticsearch is not reachable at https://localhost:9200."]


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


def test_indexer_config_uses_env_overrides_and_resolves_certs(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    cert = tmp_path / "certs" / "ca.crt"
    cert.parent.mkdir()
    cert.write_text("cert")
    config_path.write_text(
        json.dumps(
            {
                "elasticsearch": {
                    "hosts": ["https://localhost:9200"],
                    "ca_certs": "certs/ca.crt",
                    "username": "elastic",
                    "password": "CHANGE_ME",
                }
            }
        )
    )
    monkeypatch.setenv("TRIALMATCHAI_ES_HOST", "https://es.example.test:9200")
    monkeypatch.setenv("TRIALMATCHAI_ES_PASSWORD", "from-env")

    cfg = load_config(config_path)

    assert cfg["elasticsearch"]["hosts"] == ["https://es.example.test:9200"]
    assert cfg["elasticsearch"]["password"] == "from-env"
    assert cfg["elasticsearch"]["ca_certs"] == str(cert.resolve())
