"""Tests for the eligibility-criteria chunker (registry/criteria_chunking.py)."""

from __future__ import annotations

from trialmatchai.registry.criteria_chunking import (
    detect_header,
    split_eligibility_criteria,
)


def _criteria(text):
    return [(c["type"], c["criterion"]) for c in split_eligibility_criteria(text)]


def test_simple_dashed_inclusion_exclusion():
    text = "\n".join(
        [
            "Inclusion Criteria:",
            "- Age 18 years or older",
            "- Histologically confirmed lung cancer",
            "Exclusion Criteria:",
            "- Prior investigational therapy",
        ]
    )
    assert _criteria(text) == [
        ("inclusion", "Age 18 years or older"),
        ("inclusion", "Histologically confirmed lung cancer"),
        ("exclusion", "Prior investigational therapy"),
    ]


def test_unknown_fallback_for_unheadered_single_line():
    assert split_eligibility_criteria("Able to consent.") == [
        {"type": "unknown", "criterion": "Able to consent."}
    ]


def test_varied_headers_are_detected():
    assert detect_header("Key Inclusion Criteria:") == "inclusion"
    assert detect_header("EXCLUSION CRITERIA") == "exclusion"
    assert detect_header("Inclusion criteria for Cohort A:") == "inclusion"
    assert detect_header("Patients must meet the following exclusion criteria:") == "exclusion"
    # Not headers:
    assert detect_header("- Age 18 years or older") is None
    assert detect_header("No prior chemotherapy (inclusion in another study allowed)") is None


def test_numbered_and_multilevel_markers():
    text = "\n".join(
        [
            "Inclusion Criteria",
            "1. Adults aged 18 or older",
            "2. ECOG performance status 0-1",
            "2.1 No prior systemic therapy",
            "Exclusion Criteria",
            "a) Pregnancy",
            "(b) Active infection",
        ]
    )
    assert _criteria(text) == [
        ("inclusion", "Adults aged 18 or older"),
        ("inclusion", "ECOG performance status 0-1"),
        ("inclusion", "No prior systemic therapy"),
        ("exclusion", "Pregnancy"),
        ("exclusion", "Active infection"),
    ]


def test_continuation_lines_are_joined():
    text = "\n".join(
        [
            "Inclusion Criteria:",
            "- Measurable disease per RECIST 1.1 with at least one",
            "  lesion not previously irradiated",
        ]
    )
    assert _criteria(text) == [
        (
            "inclusion",
            "Measurable disease per RECIST 1.1 with at least one lesion not previously irradiated",
        )
    ]


def test_decimal_values_do_not_trigger_false_splits():
    text = "Inclusion Criteria:\n- Hemoglobin 9.5 g/dL or higher at screening"
    assert _criteria(text) == [
        ("inclusion", "Hemoglobin 9.5 g/dL or higher at screening")
    ]


def test_parenthetical_markers_are_not_split_points():
    text = "Inclusion Criteria:\n- Adequate organ function (see section 2.3 for details)"
    assert _criteria(text) == [
        ("inclusion", "Adequate organ function (see section 2.3 for details)")
    ]


def test_multiple_criteria_packed_on_one_line():
    text = "Exclusion Criteria: 1. Pregnancy 2. Active infection 3. Prior therapy"
    # The header is on the same line as the first marker; types resolve to exclusion.
    result = _criteria(text)
    assert ("exclusion", "Pregnancy") in result
    assert ("exclusion", "Active infection") in result
    assert ("exclusion", "Prior therapy") in result


def test_genus_abbreviation_at_line_start_not_mangled():
    assert _criteria("Exclusion Criteria:\nHistory of S. aureus bacteremia") == [
        ("exclusion", "History of S. aureus bacteremia")
    ]


def test_genus_abbreviation_standalone_marker_item():
    assert _criteria("Exclusion Criteria:\n- Active E. coli infection") == [
        ("exclusion", "Active E. coli infection")
    ]


def test_genus_then_real_sentence_boundary_splits():
    text = "Inclusion Criteria:\n- Prior E. coli infection resolved. No active fever."
    assert _criteria(text) == [
        ("inclusion", "Prior E. coli infection resolved."),
        ("inclusion", "No active fever."),
    ]


def test_abbreviation_periods_are_not_boundaries():
    assert _criteria(
        "Inclusion Criteria:\n- Adequate renal function, approx. 60 mL/min or higher"
    ) == [("inclusion", "Adequate renal function, approx. 60 mL/min or higher")]
    assert _criteria("Inclusion Criteria:\n- Meets protocol No. 5 requirements") == [
        ("inclusion", "Meets protocol No. 5 requirements")
    ]


def test_multi_sentence_criterion_is_split():
    text = "Inclusion Criteria:\n- Age 18 or older. Written informed consent obtained."
    assert _criteria(text) == [
        ("inclusion", "Age 18 or older."),
        ("inclusion", "Written informed consent obtained."),
    ]


def test_title_abbreviation_protected_but_real_boundary_splits():
    text = "Inclusion Criteria:\n- Approved by Dr. Smith. Patient is willing to comply."
    assert _criteria(text) == [
        ("inclusion", "Approved by Dr. Smith."),
        ("inclusion", "Patient is willing to comply."),
    ]


def test_ranges_and_ratios_do_not_split():
    assert _criteria(
        "Inclusion Criteria:\n- Creatinine no more than 1.5 mg/dL and ECOG 0-1"
    ) == [("inclusion", "Creatinine no more than 1.5 mg/dL and ECOG 0-1")]
