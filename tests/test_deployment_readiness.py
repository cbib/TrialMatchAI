from __future__ import annotations

from Matcher.config.config_loader import load_config, resolve_config_path
from Matcher.pipeline.cot_reasoning import BatchTrialProcessor
from Matcher.utils.json_utils import extract_json_object


def test_default_config_resolution_from_repo_root():
    path = resolve_config_path()
    assert path.name == "config.json"
    assert path.as_posix().endswith("source/Matcher/config/config.json")


def test_config_env_overrides_and_standard_index_names(monkeypatch):
    monkeypatch.setenv("TRIALMATCHAI_ES_HOST", "https://es.example.test:9200")
    monkeypatch.setenv("TRIALMATCHAI_ES_PASSWORD", "secret-from-env")
    monkeypatch.setenv("TRIALMATCHAI_INDEX_TRIALS_ELIGIBILITY", "trials_eligibility")
    monkeypatch.setenv("TRIALMATCHAI_BIOMEDNER_AUTO_START", "true")

    cfg = load_config()

    assert cfg["elasticsearch"]["host"] == "https://es.example.test:9200"
    assert cfg["elasticsearch"]["password"] == "secret-from-env"
    assert cfg["elasticsearch"]["index_trials"] == "clinical_trials"
    assert cfg["elasticsearch"]["index_trials_eligibility"] == "trials_eligibility"
    assert cfg["services"]["auto_start"] is True


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
