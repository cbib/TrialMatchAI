from __future__ import annotations

from trialmatchai.config.config_loader import load_config, resolve_config_path
from trialmatchai.matching.eligibility_reasoning import BatchTrialProcessor
from trialmatchai.utils.json_utils import extract_json_object


def test_default_config_resolution_from_repo_root():
    path = resolve_config_path()
    assert path.name == "config.json"
    assert path.as_posix().endswith("src/trialmatchai/config/config.json")


def test_config_env_overrides_and_search_tables(monkeypatch):
    monkeypatch.setenv("TRIALMATCHAI_SEARCH_DB_PATH", "data/search-test")
    monkeypatch.setenv("TRIALMATCHAI_SEARCH_TRIALS_TABLE", "trials-test")
    monkeypatch.setenv("TRIALMATCHAI_SEARCH_CRITERIA_TABLE", "criteria-test")
    monkeypatch.setenv("TRIALMATCHAI_SEARCH_MODE", "bm25")
    monkeypatch.setenv("TRIALMATCHAI_ENTITY_BACKEND", "regex")
    monkeypatch.setenv("TRIALMATCHAI_CONCEPT_DB_PATH", "data/concepts-test")
    monkeypatch.setenv("TRIALMATCHAI_LINK_ACCEPT", "0.9")
    monkeypatch.setenv("TRIALMATCHAI_REGISTRY_SINCE_DAYS", "14")
    monkeypatch.setenv("TRIALMATCHAI_REGISTRY_RAW_DIR", "data/registry/raw-test")

    cfg = load_config()

    assert cfg["search_backend"]["backend"] == "lancedb"
    assert cfg["search_backend"]["db_path"].endswith("data/search-test")
    assert cfg["search_backend"]["trials_table"] == "trials-test"
    assert cfg["search_backend"]["criteria_table"] == "criteria-test"
    assert cfg["search"]["mode"] == "bm25"
    assert cfg["entity_extraction"]["backend"] == "regex"
    assert cfg["concept_linker"]["db_path"].endswith("data/concepts-test")
    assert cfg["concept_linker"]["accept_threshold"] == 0.9
    assert cfg["registry"]["since_days"] == 14
    assert cfg["registry"]["raw_dir"].endswith("data/registry/raw-test")


def test_cot_prompt_does_not_inject_consent():
    processor = BatchTrialProcessor.__new__(BatchTrialProcessor)
    processor.use_cot = True
    processor.tokenizer = object()

    prompt = processor._format_prompt("Age >= 18", "Patient has lung cancer.")

    assert "Written informed consent has been obtained" not in prompt


def test_json_extraction_uses_balanced_object():
    output = 'prefix {"outer": {"inner": "value"}, "items": [1, 2]} suffix {"bad": true}'

    parsed = extract_json_object(output)

    assert parsed == {"outer": {"inner": "value"}, "items": [1, 2]}


def test_json_extraction_rejects_malformed_output():
    try:
        extract_json_object('prefix {"outer": {"inner": "value"}')
    except ValueError as exc:
        assert "Unbalanced JSON object" in str(exc)
    else:
        raise AssertionError("Malformed output should fail JSON extraction")
