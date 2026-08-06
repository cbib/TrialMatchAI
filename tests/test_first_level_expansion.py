"""First-level LLM query expansion (the llm_expansion search channel).

This channel was dead before: the protocol, parser, schema and config flag all existed,
but nothing ever constructed a backend, so enabling the flag logged "no expander is
configured" and returned no terms. These tests cover the backend and the config gate.
"""

import json

import pytest

from trialmatchai.matching.query_expansion import (
    _FIRST_LEVEL_FIELDS,
    FirstLevelQueryExpander,
    _first_level_patient_text,
    build_first_level_expander,
)
from trialmatchai.matching.retrieval.first_level_planner import parse_llm_query_expansion

SUMMARY = {
    "main_conditions": ["metastatic breast cancer"],
    "other_conditions": ["hypertension", "HER2 positive"],
    "patient_narrative": ["A 54 year old woman with metastatic breast cancer."],
    "age": 54,
    "gender": "female",
}


class _FakeExpander(FirstLevelQueryExpander):
    """Bypasses __init__ so no model or GPU is touched; _generate returns a canned reply."""

    def __init__(self, reply):
        self._reply = reply
        self.settings = {"guided_json": True, "max_new_tokens": 512}
        self.config = {}
        self.backend = "vllm"

    def _generate(self, narrative):
        if isinstance(self._reply, Exception):
            raise self._reply
        return self._reply


def test_expands_into_the_six_planner_fields():
    payload = {
        "primary_queries": ["metastatic breast cancer"],
        "disease_aliases": ["breast carcinoma", "mammary carcinoma"],
        "broader_queries": ["solid tumor"],
        "biomarker_queries": ["HER2 positive"],
        "treatment_queries": ["trastuzumab"],
        "discarded_or_uncertain": ["hypertension"],
    }
    result = _FakeExpander(json.dumps(payload)).expand_first_level_queries(
        profile=None, matching_summary=SUMMARY
    )
    assert result == payload
    assert set(result) == set(_FIRST_LEVEL_FIELDS)


def test_output_is_consumable_by_the_planner_parser():
    """The backend's contract is the planner's parser, not just valid JSON."""
    payload = {
        "primary_queries": ["metastatic breast cancer"],
        "disease_aliases": ["breast carcinoma"],
        "broader_queries": ["solid tumor"],
        "biomarker_queries": ["HER2 positive"],
        "treatment_queries": ["trastuzumab"],
        "discarded_or_uncertain": [],
    }
    raw = _FakeExpander(json.dumps(payload)).expand_first_level_queries(
        profile=None, matching_summary=SUMMARY
    )
    parsed = parse_llm_query_expansion(raw, max_terms=12)
    assert parsed.primary_queries == ["metastatic breast cancer"]
    assert parsed.biomarker_queries == ["HER2 positive"]


def test_max_terms_is_a_shared_budget_spent_primary_first():
    """llm_max_terms caps the TOTAL across the five query fields, not each one, and is spent
    in field order. A model that fills primary_queries can starve the later channels, so the
    prompt must keep primary_queries to the actual disease rather than padding it."""
    payload = {field: [] for field in _FIRST_LEVEL_FIELDS}
    payload["primary_queries"] = [f"q{i}" for i in range(5)]
    payload["disease_aliases"] = ["alias1", "alias2"]
    payload["biomarker_queries"] = ["EGFR"]

    parsed = parse_llm_query_expansion(payload, max_terms=6)

    assert parsed.primary_queries == [f"q{i}" for i in range(5)]
    assert parsed.disease_aliases == ["alias1"]  # only one slot left
    assert parsed.biomarker_queries == []  # budget exhausted before this field


def test_reasoning_tags_are_stripped_before_json_extraction():
    """Reasoning models emit <think> containing an echo of the schema; extracting from that
    would return the schema instead of the answer."""
    payload = {field: [] for field in _FIRST_LEVEL_FIELDS}
    payload["primary_queries"] = ["glioblastoma"]
    reply = (
        "<think>The schema wants primary_queries, disease_aliases, ...</think>"
        + json.dumps(payload)
    )
    result = _FakeExpander(reply).expand_first_level_queries(
        profile=None, matching_summary=SUMMARY
    )
    assert result["primary_queries"] == ["glioblastoma"]


@pytest.mark.parametrize(
    "reply",
    ["not json at all", json.dumps(["a", "list"]), RuntimeError("engine died")],
)
def test_failures_degrade_to_empty_not_raise(reply):
    """Retrieval must survive a failed expansion: this is 1 of 8 channels, weight 0.5."""
    result = _FakeExpander(reply).expand_first_level_queries(
        profile=None, matching_summary=SUMMARY
    )
    assert result == {field: [] for field in _FIRST_LEVEL_FIELDS}


def test_a_bare_string_field_is_not_shredded_into_characters():
    payload = {field: [] for field in _FIRST_LEVEL_FIELDS}
    payload["primary_queries"] = "glioblastoma"
    result = _FakeExpander(json.dumps(payload)).expand_first_level_queries(
        profile=None, matching_summary=SUMMARY
    )
    assert result["primary_queries"] == ["glioblastoma"]


def test_empty_summary_skips_the_model_entirely():
    expander = _FakeExpander(RuntimeError("must not be called"))
    assert expander.expand_first_level_queries(profile=None, matching_summary={}) == {
        field: [] for field in _FIRST_LEVEL_FIELDS
    }


def test_patient_text_includes_conditions_and_demographics():
    text = _first_level_patient_text(None, SUMMARY)
    assert "metastatic breast cancer" in text
    assert "HER2 positive" in text
    assert "54" in text and "female" in text


def test_patient_text_omits_placeholder_demographics():
    text = _first_level_patient_text(
        None, {"main_conditions": ["asthma"], "age": "all", "gender": "all"}
    )
    assert "asthma" in text
    assert "Age:" not in text and "Sex:" not in text


def test_builder_returns_none_unless_the_flag_is_set():
    assert build_first_level_expander({}) is None
    assert build_first_level_expander({"search": {"first_level": {}}}) is None
    assert (
        build_first_level_expander(
            {"search": {"first_level": {"llm_expansion_enabled": False}}}
        )
        is None
    )


def test_builder_degrades_to_none_when_construction_fails():
    """A misconfigured expander must not abort the run; the channel just stays empty."""
    config = {
        "search": {"first_level": {"llm_expansion_enabled": True}},
        "model": {},  # no base_model -> QueryExpander raises
        "query_expansion": {},
    }
    assert build_first_level_expander(config) is None


def test_schema_caps_primary_queries_tightly_to_protect_the_shared_budget():
    """Guards the interaction pinned above: primary_queries is spent first out of
    llm_max_terms, so its schema cap must leave room for the later channels."""
    from trialmatchai.matching.query_expansion import _FIRST_LEVEL_JSON_SCHEMA

    props = _FIRST_LEVEL_JSON_SCHEMA["properties"]
    primary = props["primary_queries"]["maxItems"]
    assert primary <= 3
    for field in ("biomarker_queries", "treatment_queries", "disease_aliases"):
        assert props[field]["maxItems"] > primary
