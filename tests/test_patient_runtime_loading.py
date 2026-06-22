from __future__ import annotations

import json

from trialmatchai.main import _load_patient_inputs


def test_runtime_loads_profiles_from_configured_directory(tmp_path):
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    summaries = tmp_path / "summaries"
    summaries.mkdir()
    (profiles / "p1.json").write_text(
        json.dumps({"patient_id": "p1", "demographics": {}}),
        encoding="utf-8",
    )
    (summaries / "p1.json").write_text(
        json.dumps(
            {
                "patient_id": "p1",
                "main_conditions": ["melanoma"],
                "other_conditions": [],
                "patient_narrative": ["Patient has melanoma."],
                "age": "all",
                "gender": "all",
            }
        ),
        encoding="utf-8",
    )

    loaded = _load_patient_inputs(
        {
            "patient_inputs": {
                "profile_dir": str(profiles),
                "summary_dir": str(summaries),
            },
        }
    )

    assert loaded[0][0].patient_id == "p1"
    assert loaded[0][1]["main_conditions"] == ["melanoma"]


def test_runtime_requires_canonical_profiles(tmp_path):
    profiles = tmp_path / "profiles"
    summaries = tmp_path / "summaries"

    loaded = _load_patient_inputs(
        {
            "patient_inputs": {
                "profile_dir": str(profiles),
                "summary_dir": str(summaries),
            },
        }
    )

    assert loaded == []
