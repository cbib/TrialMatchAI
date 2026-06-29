"""Reproduction tests for the medium-severity bugs found in the codebase audit.

Each test fails on the pre-fix code and passes on the fix.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def test_tie_aware_ndcg_is_input_order_invariant():
    # The tie-aware nDCG groups equal-score items; it must not depend on whether
    # the caller handed in a score-sorted list. Ties (b, c) are interleaved below.
    from trialmatchai.trec.metrics import condensed_ndcg, ndcg_at_k

    score = {"a": 0.9, "b": 0.5, "c": 0.5, "d": 0.1}
    gain = {"a": 2, "b": 0, "c": 3, "d": 1}
    by_score = ["a", "b", "c", "d"]
    interleaved = ["c", "a", "d", "b"]  # same items, tie group not contiguous

    assert ndcg_at_k(by_score, score, gain, 4) == pytest.approx(
        ndcg_at_k(interleaved, score, gain, 4)
    )
    assert condensed_ndcg(by_score, score, gain, [4]) == condensed_ndcg(
        interleaved, score, gain, [4]
    )


def test_omop_measurement_keeps_zero_value():
    # A measurement of exactly 0 (e.g. a count, a delta) must be retained, not
    # silently dropped by an `x or ''` falsiness check.
    from trialmatchai.interop.importers.omop import _add_measurement_rows
    from trialmatchai.interop.models import PatientProfile

    profile = PatientProfile(patient_id="P1")
    rows = [
        {
            "measurement_concept_id": "",
            "value_as_number": 0,
            "unit_concept_id": "",
            "value_as_concept_id": "",
            "measurement_date": "2020-01-01",
            "measurement_source_value": "glucose delta",
        }
    ]
    _add_measurement_rows(profile, rows, {}, Path("."))
    assert profile.observations
    assert "0" in (profile.observations[0].description or "")


def test_topic_sex_uses_first_mention():
    # The subject is named before any relative/partner, so the first sex mention
    # wins — previously Female always won regardless of position.
    from trialmatchai.trec.topics import extract_demographics

    _, sex = extract_demographics("A 45-year-old man with a female partner seeks care.")
    assert sex == "Male"
    _, sex2 = extract_demographics("A 30-year-old woman whose husband is male.")
    assert sex2 == "Female"


def test_parse_age_input_rejects_out_of_range():
    from trialmatchai.matching.retrieval.trial_retrieval import ClinicalTrialSearch

    search = ClinicalTrialSearch.__new__(ClinicalTrialSearch)
    assert search.parse_age_input("45") == 45
    assert search.parse_age_input("45 years") == 45
    assert search.parse_age_input("-5") is None  # was: -5
    assert search.parse_age_input(-5) is None  # int branch guarded too
    assert search.parse_age_input(200) is None
