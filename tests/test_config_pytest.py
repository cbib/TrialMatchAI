import json
from pathlib import Path

import pytest

from trialmatchai.config.config_loader import load_config
from trialmatchai.config.settings import EntityExtractionSettings
from trialmatchai.entities.schemas import default_schema_path

REPO_CONFIG = Path(__file__).resolve().parents[1] / "src/trialmatchai/config/config.json"


def test_load_config_from_repo():
    cfg = load_config(str(REPO_CONFIG))
    assert cfg["search_backend"]["backend"] == "lancedb"
    assert "embedder" in cfg
    assert "paths" in cfg


def _write(tmp_path, cfg) -> str:
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    return str(p)


def test_load_config_is_non_lossy_superset(tmp_path):
    """Every key in the raw JSON survives to the consumer dict, including undeclared knobs
    (RC1 fix: the loader overlays the validated dump onto raw instead of replacing it)."""
    raw = json.loads(REPO_CONFIG.read_text(encoding="utf-8"))
    raw.setdefault("vllm", {})["swap_space"] = 4  # undeclared knob a consumer reads via .get()
    raw["embedder"]["future_knob"] = "keepme"
    cfg = load_config(_write(tmp_path, raw))
    assert cfg["vllm"]["swap_space"] == 4
    assert cfg["embedder"]["future_knob"] == "keepme"

    def all_keys_present(src, out, path=""):
        for k, v in src.items():
            assert k in out, f"dropped key: {path}{k}"
            if isinstance(v, dict) and isinstance(out[k], dict):
                all_keys_present(v, out[k], f"{path}{k}.")

    all_keys_present(raw, cfg)


def test_no_think_reaches_consumers(tmp_path):
    """rag.no_think / query_expansion.no_think are declared fields and survive the load
    (previously dropped by model_dump -> a dead knob for reasoning models)."""
    raw = json.loads(REPO_CONFIG.read_text(encoding="utf-8"))
    raw.setdefault("rag", {})["no_think"] = True
    raw.setdefault("query_expansion", {})["no_think"] = True
    cfg = load_config(_write(tmp_path, raw))
    assert cfg["rag"]["no_think"] is True
    assert cfg["query_expansion"]["no_think"] is True


def test_validated_value_wins_over_raw(tmp_path):
    """For a declared field, the validated/coerced value wins (dump overlays raw)."""
    raw = json.loads(REPO_CONFIG.read_text(encoding="utf-8"))
    raw.setdefault("vllm", {})["gpu_memory_utilization"] = "0.55"  # str -> coerced to float
    cfg = load_config(_write(tmp_path, raw))
    assert cfg["vllm"]["gpu_memory_utilization"] == 0.55
    assert isinstance(cfg["vllm"]["gpu_memory_utilization"], float)


def test_packaged_schema_path_resolves_outside_repo(tmp_path, monkeypatch):
    source_config = (
        Path(__file__).resolve().parents[1] / "src/trialmatchai/config/config.json"
    )
    installed_config = tmp_path / "site-packages/trialmatchai/config/config.json"
    installed_config.parent.mkdir(parents=True)
    installed_config.write_text(source_config.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    cfg = load_config(installed_config)

    assert cfg["entity_extraction"]["schema_path"] == str(default_schema_path().resolve())
    assert Path(cfg["entity_extraction"]["schema_path"]).exists()
    assert cfg["paths"]["output_dir"] == str((tmp_path / "results").resolve())


def test_legacy_gliner_backend_and_fallback_key_are_rejected():
    with pytest.raises(ValueError):
        EntityExtractionSettings.model_validate({"backend": "gliner"})

    with pytest.raises(ValueError):
        EntityExtractionSettings.model_validate({"fallback_model_name": "old-model"})
