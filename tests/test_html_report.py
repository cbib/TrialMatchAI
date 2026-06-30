"""Tests for the self-contained HTML results report exporter.

All fixtures are synthetic — no real patient data is used.
"""

from __future__ import annotations

import json

from trialmatchai.interop.exporters.html_report import (
    build_report_model,
    profile_to_html_report,
    render_html_report,
)

_SUMMARY = {
    "patient_id": "P1",
    "age": 63,
    "gender": "Female",
    "main_conditions": ["NSCLC", "EGFR mutation"],
    "other_conditions": ["hypertension"],
    "patient_narrative": ["63-year-old female with metastatic NSCLC."],
}
_RANKED = {
    "RankedTrials": [
        {"TrialID": "NCT002", "Score": 1.0, "RerankerScore": 0.90, "FirstLevelScore": 0.5},
        {"TrialID": "NCT001", "Score": 1.0, "RerankerScore": 0.40, "FirstLevelScore": 0.3},
        {"TrialID": "NL99", "Score": 1.0, "RerankerScore": 0.20, "FirstLevelScore": 0.1},
        {"TrialID": "NCT003", "Score": 0.0, "RerankerScore": 0.10, "FirstLevelScore": 0.0},
    ]
}
_ELIG = {
    "NCT002": {
        "Final Decision": "Eligible",
        "Recap": "Meets disease and mutation.",
        "Inclusion_Criteria_Evaluation": [
            {"Criterion": "EGFR mutation", "Classification": "Met", "Justification": "Confirmed."}
        ],
        "Exclusion_Criteria_Evaluation": [
            {"Criterion": "Second malignancy", "Classification": "Not Violated", "Justification": "None."}
        ],
    },
    "NCT003": {"error": "invalid_json_response", "raw_output": "garbled"},
}
_META = {
    "NCT002": {
        "brief_title": "Targeted Therapy in EGFR-mutant NSCLC",
        "brief_summary": "A study of targeted therapy.",
        "overall_status": "Recruiting",
        "phase": "Phase 2",
        "condition": "NSCLC",
        "intervention": [{"name": "Drug X", "type": "Drug", "description": ""}],
        "brief_title_vector": [0.1, 0.2],  # must be dropped
    }
}


def _model():
    return build_report_model(
        patient_summary=_SUMMARY,
        ranked=_RANKED,
        eligibility_by_id=_ELIG,
        meta_by_id=_META,
        generated_at="2026-06-30 11:00",
    )


def test_build_report_model_joins_ranks_and_scores():
    m = _model()
    assert m["patient"]["id"] == "P1" and m["patient"]["sex"] == "Female"
    assert [t["trial_id"] for t in m["trials"]] == ["NCT002", "NCT001", "NL99", "NCT003"]
    assert [t["rank"] for t in m["trials"]] == [1, 2, 3, 4]
    top = m["trials"][0]
    assert top["reranker_score"] == 0.90  # discriminative score surfaced (Option A)
    assert top["final_decision"] == "Eligible"
    assert top["meta"]["brief_title"] == "Targeted Therapy in EGFR-mutant NSCLC"
    assert "brief_title_vector" not in top["meta"]  # embedding fields dropped
    assert top["meta"]["interventions"] == ["Drug X"]
    assert top["inclusion"][0]["classification"] == "Met"


def test_error_sentinel_and_missing_reasoning_degrade():
    m = _model()
    sentinel = next(t for t in m["trials"] if t["trial_id"] == "NCT003")
    assert sentinel["reasoning_available"] is False
    assert sentinel["inclusion"] == [] and sentinel["final_decision"] is None
    nl = next(t for t in m["trials"] if t["trial_id"] == "NL99")
    assert nl["metadata_available"] is False  # NL registry trial, no metadata
    assert nl["reasoning_available"] is False


def test_accepts_bare_list_ranked_trials():
    m = build_report_model(
        patient_summary={"patient_id": "P9"},
        ranked=[{"TrialID": "NCT1", "Score": 1.0}],  # legacy bare array
        eligibility_by_id={},
        meta_by_id={},
        generated_at="x",
    )
    assert [t["trial_id"] for t in m["trials"]] == ["NCT1"]


def test_render_escapes_script_injection():
    model = build_report_model(
        patient_summary={"patient_id": "P1"},
        ranked={"RankedTrials": [{"TrialID": "NCT1", "Score": 1.0}]},
        eligibility_by_id={
            "NCT1": {
                "Final Decision": "Eligible",
                "Inclusion_Criteria_Evaluation": [
                    {"Criterion": "c", "Classification": "Met",
                     "Justification": "</script><script>alert(1)</script>"}
                ],
            }
        },
        meta_by_id={},
        generated_at="x",
    )
    html = render_html_report(model)
    # the injected closing tag must be neutralized, not left to break out of the island
    assert "<script>alert(1)" not in html
    assert "\\u003cscript\\u003ealert(1)" in html
    # the data still round-trips for the client
    island = html.split('id="report-data">', 1)[1].split("</script>", 1)[0]
    assert json.loads(island)["trials"][0]["trial_id"] == "NCT1"


def test_profile_to_html_report_end_to_end(tmp_path):
    out = tmp_path / "results" / "P1"
    out.mkdir(parents=True)
    (out / "ranked_trials.json").write_text(json.dumps(_RANKED), encoding="utf-8")
    (out / "NCT002.json").write_text(json.dumps(_ELIG["NCT002"]), encoding="utf-8")
    (out / "NCT003.json").write_text(json.dumps(_ELIG["NCT003"]), encoding="utf-8")
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    (meta_dir / "NCT002.json").write_text(json.dumps(_META["NCT002"]), encoding="utf-8")
    summ_dir = tmp_path / "summaries"
    summ_dir.mkdir()
    (summ_dir / "P1.json").write_text(json.dumps(_SUMMARY), encoding="utf-8")

    html = profile_to_html_report(
        out, summary_dir=summ_dir, trial_meta_folders=[meta_dir], generated_at="2026-06-30 11:00"
    )
    assert html.lstrip().startswith("<!doctype html>")
    assert "Targeted Therapy in EGFR-mutant NSCLC" in html  # joined metadata
    assert "NL99" in html  # graceful degrade for the no-metadata trial
    assert "Patient" in html


def test_render_index_html_links_patients():
    from trialmatchai.interop.exporters.html_report import render_index_html

    html = render_index_html(
        [
            {"patient_id": "P1", "n_trials": 5, "href": "P1/report.html"},
            {"patient_id": "P2", "n_trials": 3, "href": "P2/report.html"},
        ],
        "2026-06-30 12:00",
    )
    assert html.lstrip().startswith("<!doctype html>")
    assert 'href="P1/report.html"' in html and "Patient P1" in html
    assert "2 patients" in html


def test_reporting_default_on_and_trec_disables():
    from types import SimpleNamespace

    from trialmatchai.config.settings import ReportingSettings
    from trialmatchai.trec.runner import _track_config

    assert ReportingSettings().emit_html is True  # interactive runs emit by default
    spec = SimpleNamespace(db_path="d", profile_dir="p", summary_dir="s", output_dir="o")
    cfg = _track_config({}, spec)
    assert cfg["reporting"]["emit_html"] is False  # TREC sweep does not


def test_maybe_write_report_respects_gate(tmp_path):
    from trialmatchai.main import _maybe_write_report

    pdir = tmp_path / "P1"
    pdir.mkdir()
    (pdir / "ranked_trials.json").write_text(
        json.dumps({"RankedTrials": [{"TrialID": "NCT1", "Score": 1.0}]}), encoding="utf-8"
    )
    _maybe_write_report(pdir, {"reporting": {"emit_html": False}})
    assert not (pdir / "report.html").exists()  # gated off
    # default-on; never raises even with no metadata folder present
    _maybe_write_report(pdir, {"paths": {"trials_json_folder": str(tmp_path / "meta")}, "patient_inputs": {}})
    assert (pdir / "report.html").exists()


def test_read_cot_extracts_thinking(tmp_path):
    from trialmatchai.interop.exporters.html_report import _read_cot

    p = tmp_path / "NCT1.txt"
    p.write_text("<think>step one\nstep two</think>\n{\"Final Decision\": \"Eligible\"}", encoding="utf-8")
    assert _read_cot(p) == "step one\nstep two"
    p.write_text('{"no": "thinking here"}', encoding="utf-8")
    assert _read_cot(p) is None  # no <think> block -> no panel
    assert _read_cot(tmp_path / "missing.txt") is None


def test_build_report_model_includes_cot():
    from trialmatchai.interop.exporters.html_report import build_report_model

    m = build_report_model(
        patient_summary={"patient_id": "P1"},
        ranked={"RankedTrials": [{"TrialID": "NCT1", "Score": 1.0}]},
        eligibility_by_id={"NCT1": {"Final Decision": "Eligible", "Inclusion_Criteria_Evaluation": []}},
        meta_by_id={},
        cot_by_id={"NCT1": "the model's reasoning"},
        generated_at="x",
    )
    assert m["trials"][0]["cot"] == "the model's reasoning"
