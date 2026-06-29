"""P3: realistic patient-input fixtures — unicode, multiline, odd filenames —
that exercise the importers beyond the happy-path ASCII cases."""

from __future__ import annotations

from trialmatchai.interop.importers.text import import_text_note


def test_text_note_preserves_unicode_and_multiline(tmp_path):
    text = "Café patient — 50yo\nDiagnosis: naïve to treatment\nBRAF V600E"
    note = tmp_path / "café-patient_001.txt"
    note.write_text(text, encoding="utf-8")

    profile = import_text_note(note)
    assert profile.patient_id  # safely derived from a unicode filename
    assert profile.notes[0].text == text  # verbatim: unicode + newlines intact
