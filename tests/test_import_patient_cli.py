from __future__ import annotations

import json

from trialmatchai.cli.import_patient import main


def test_import_patient_cli_writes_profile_and_summary(tmp_path, monkeypatch):
    note = tmp_path / "patient.txt"
    note.write_text("Patient has breast cancer.", encoding="utf-8")
    profile_dir = tmp_path / "profiles"
    summary_dir = tmp_path / "summaries"

    monkeypatch.setattr(
        "sys.argv",
        [
            "trialmatchai-import-patient",
            "--input",
            str(note),
            "--format",
            "text",
            "--output-dir",
            str(profile_dir),
            "--summary-dir",
            str(summary_dir),
            "--no-entities",
        ],
    )

    assert main() == 0

    profile = json.loads((profile_dir / "patient.json").read_text(encoding="utf-8"))
    summary = json.loads((summary_dir / "patient.json").read_text(encoding="utf-8"))
    assert profile["patient_id"] == "patient"
    assert summary["patient_id"] == "patient"
    assert summary["main_conditions"] == ["Patient has breast cancer."]
    assert summary["expanded_sentences"]
