"""Tests for the optional country-level, site-aware trial location filter."""

from __future__ import annotations

import json

from trialmatchai.interop import import_patient_path
from trialmatchai.interop.models import (
    ClinicalFact,
    Location,
    PatientProfile,
    Provenance,
)
from trialmatchai.main import run_first_level_search
from trialmatchai.matching.retrieval.location import (
    filter_trials_by_country,
    patient_country,
    trial_in_country,
)
from trialmatchai.search import InMemorySearchBackend


def _trial(nct_id, countries):
    return {
        "nct_id": nct_id,
        "condition": "lung cancer",
        "brief_title": "Lung cancer trial",
        "eligibility_criteria": "Adults with lung cancer.",
        "overall_status": "Recruiting",
        "gender": "All",
        "location": [{"country": c} for c in countries],
    }


def test_filter_is_recall_safe():
    us = _trial("N_US", ["United States", "Germany"])
    fr = _trial("N_FR", ["France"])
    unknown = {"nct_id": "N_UNK"}  # no location data
    trials = [us, fr, unknown]
    scores = [3.0, 2.0, 1.0]

    kept, kept_scores = filter_trials_by_country(trials, scores, "United States")
    ids = {t["nct_id"] for t in kept}
    assert ids == {"N_US", "N_UNK"}  # multi-site US kept; unknown kept; France dropped
    assert kept_scores == [3.0, 1.0]

    # No patient country -> no filtering.
    assert filter_trials_by_country(trials, scores, None)[0] == trials
    assert trial_in_country(unknown, "France") is True  # unknown location never dropped


def test_patient_country_extraction():
    profile = PatientProfile(patient_id="P", location=Location(country="  France "))
    assert patient_country(profile) == "France"
    assert patient_country(PatientProfile(patient_id="P")) is None


def _fl_config(*, hard_filters):
    return {
        "search": {
            "mode": "bm25",
            "vector_score_threshold": 0.5,
            "max_trials_first_level": 1000,
            "first_level": {
                "enabled": True,
                "max_trials": 1000,
                "per_channel_size": 300,
                "rrf_k": 60,
                "vector_score_threshold": 0.0,
                "llm_expansion_enabled": False,
                "write_reports": False,
                "hard_filters": hard_filters,
            },
        }
    }


def _profile_us():
    return PatientProfile(
        patient_id="P1",
        location=Location(country="United States"),
        conditions=[
            ClinicalFact(
                fact_id="c1",
                category="condition",
                label="lung cancer",
                provenance=Provenance(source_format="test"),
            )
        ],
    )


def test_location_hard_filter_drops_out_of_country_trials(tmp_path):
    backend = InMemorySearchBackend(
        trials=[_trial("N_US", ["United States"]), _trial("N_FR", ["France"])]
    )
    result = run_first_level_search(
        {"main_conditions": ["lung cancer"], "other_conditions": [], "patient_narrative": []},
        str(tmp_path),
        {"age": "all", "gender": "ALL"},
        None,
        None,
        _fl_config(hard_filters=["age", "sex", "overall_status", "location"]),
        backend,
        patient_profile=_profile_us(),
    )
    nct_ids, *_ = result
    assert "N_US" in nct_ids
    assert "N_FR" not in nct_ids


def test_location_filter_off_by_default_keeps_all_countries(tmp_path):
    backend = InMemorySearchBackend(
        trials=[_trial("N_US", ["United States"]), _trial("N_FR", ["France"])]
    )
    result = run_first_level_search(
        {"main_conditions": ["lung cancer"], "other_conditions": [], "patient_narrative": []},
        str(tmp_path),
        {"age": "all", "gender": "ALL"},
        None,
        None,
        _fl_config(hard_filters=["age", "sex", "overall_status"]),
        backend,
        patient_profile=_profile_us(),
    )
    nct_ids, *_ = result
    assert {"N_US", "N_FR"} <= set(nct_ids)


def test_omop_importer_extracts_patient_location(tmp_path):
    import pandas as pd

    omop = tmp_path / "omop"
    omop.mkdir()
    pd.DataFrame(
        [{"person_id": 1, "gender_source_value": "M", "year_of_birth": 1975, "location_id": 10}]
    ).to_csv(omop / "PERSON.csv", index=False)
    pd.DataFrame(
        [{"location_id": 10, "city": "Lyon", "country_source_value": "France"}]
    ).to_csv(omop / "LOCATION.csv", index=False)

    profiles = import_patient_path(omop)
    assert len(profiles) == 1
    assert profiles[0].location is not None
    assert profiles[0].location.country == "France"
    assert profiles[0].location.city == "Lyon"


def test_fhir_importer_extracts_patient_location(tmp_path):
    path = tmp_path / "patient.json"
    path.write_text(
        json.dumps(
            {
                "resourceType": "Patient",
                "id": "pat-1",
                "gender": "female",
                "birthDate": "1980-01-01",
                "address": [{"city": "Boston", "state": "MA", "country": "United States"}],
            }
        )
    )
    profiles = import_patient_path(path)
    assert profiles[0].location is not None
    assert profiles[0].location.country == "United States"
    assert profiles[0].location.city == "Boston"
