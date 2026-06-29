"""P1 coverage for the crash-safe-resume + atomic-write invariants the project
headlines. These mechanisms were implemented but had zero failure-path coverage
(test/codebase audit): a regression to the atomic writers or the resume gates
would previously pass the whole suite green while breaking durability.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from trialmatchai.utils.file_utils import (
    is_valid_json_file,
    write_json_file,
    write_text_file,
)


def _boom(*_args, **_kwargs):
    raise OSError("simulated crash")


def _no_tmp_leftover(directory: Path) -> bool:
    return not any(p.name.startswith(".tmp-") for p in directory.iterdir())


# --------------------------------------------------------------------------- #
# is_valid_json_file — the gate every resume path keys on.
# --------------------------------------------------------------------------- #
def test_is_valid_json_file_truth_table(tmp_path):
    (tmp_path / "good.json").write_text('{"a": 1}', encoding="utf-8")
    (tmp_path / "arr.json").write_text("[]", encoding="utf-8")
    (tmp_path / "truncated.json").write_text('{"a": 1', encoding="utf-8")
    (tmp_path / "empty.json").write_text("", encoding="utf-8")
    (tmp_path / "binary.json").write_bytes(b"\x00\x01\x02")
    (tmp_path / "adir").mkdir()

    assert is_valid_json_file(str(tmp_path / "good.json")) is True
    assert is_valid_json_file(str(tmp_path / "arr.json")) is True  # valid JSON list
    assert is_valid_json_file(str(tmp_path / "truncated.json")) is False  # partial write
    assert is_valid_json_file(str(tmp_path / "empty.json")) is False
    assert is_valid_json_file(str(tmp_path / "binary.json")) is False
    assert is_valid_json_file(str(tmp_path / "missing.json")) is False
    assert is_valid_json_file(str(tmp_path / "adir")) is False


# --------------------------------------------------------------------------- #
# Atomic writes — a crash mid-write must never leave a partial file or orphan.
# --------------------------------------------------------------------------- #
def test_write_json_file_failure_leaves_original_and_no_tmp(tmp_path, monkeypatch):
    target = tmp_path / "out.json"
    target.write_text('{"original": true}', encoding="utf-8")
    monkeypatch.setattr(os, "replace", _boom)

    with pytest.raises(ValueError):
        write_json_file({"new": 1}, str(target))

    assert json.loads(target.read_text()) == {"original": True}  # never clobbered
    assert _no_tmp_leftover(tmp_path)  # tmp cleaned up


def test_write_json_file_round_trips_in_nested_dir(tmp_path):
    target = tmp_path / "sub" / "out.json"
    write_json_file({"k": "v"}, str(target))
    assert json.loads(target.read_text()) == {"k": "v"}
    assert _no_tmp_leftover(target.parent)


def test_write_text_file_is_atomic(tmp_path, monkeypatch):
    target = tmp_path / "note.txt"
    target.write_text("ORIGINAL", encoding="utf-8")
    monkeypatch.setattr(os, "replace", _boom)

    with pytest.raises(ValueError):
        write_text_file(["new line"], str(target))
    assert target.read_text() == "ORIGINAL"  # never partially overwritten
    assert _no_tmp_leftover(tmp_path)

    monkeypatch.undo()
    write_text_file(["a", "b"], str(target))
    assert target.read_text() == "a\nb"


def test_preparation_atomic_write_text_cleans_up_on_failure(tmp_path, monkeypatch):
    from trialmatchai.registry.preparation import _atomic_write_text

    target = tmp_path / "c.json"
    monkeypatch.setattr(os, "replace", _boom)
    with pytest.raises(OSError):
        _atomic_write_text(target, "payload")
    assert not target.exists()
    assert _no_tmp_leftover(tmp_path)


def test_linking_atomic_write_json_cleans_up_on_failure(tmp_path, monkeypatch):
    from trialmatchai.linking import _atomic_write_json

    target = tmp_path / "x.json"
    monkeypatch.setattr(os, "replace", _boom)
    with pytest.raises(OSError):
        _atomic_write_json(target, {"a": 1})
    assert not target.exists()
    assert _no_tmp_leftover(tmp_path)


# --------------------------------------------------------------------------- #
# Eligibility resume gates — a transiently-failed trial must be retried, not
# locked into the ranking.
# --------------------------------------------------------------------------- #
def test_is_error_output_truth_table(tmp_path):
    from trialmatchai.matching.eligibility_base import _is_error_output

    (tmp_path / "ok.json").write_text('{"Inclusion_Criteria_Evaluation": []}', encoding="utf-8")
    (tmp_path / "err.json").write_text('{"error": "invalid_json_response"}', encoding="utf-8")
    (tmp_path / "corrupt.json").write_text("{not json", encoding="utf-8")

    assert _is_error_output(str(tmp_path / "ok.json")) is False  # valid result -> done
    assert _is_error_output(str(tmp_path / "err.json")) is True  # recorded failure -> retry
    assert _is_error_output(str(tmp_path / "corrupt.json")) is True  # unparseable -> retry
    assert _is_error_output(str(tmp_path / "missing.json")) is True  # absent -> (re)process


def test_strip_thinking_tags():
    from trialmatchai.matching.eligibility_base import BaseTrialProcessor

    strip = BaseTrialProcessor._strip_thinking_tags
    assert strip("before<think>reasoning</think>after") == "beforeafter"
    assert strip("answer <think>cut off mid thought") == "answer"
    assert strip("no tags here") == "no tags here"


# --------------------------------------------------------------------------- #
# prepare_corpus — the resume engine: skip valid markers, re-prepare corrupt/
# missing ones, isolate per-trial failures, and never load the model when there
# is nothing to do.
# --------------------------------------------------------------------------- #
def _make_trials(tmp_path, ncts):
    src = tmp_path / "trials_jsons"
    src.mkdir()
    for nct in ncts:
        (src / f"{nct}.json").write_text(json.dumps({"nct_id": nct}), encoding="utf-8")
    return src


def _stub_prepare(monkeypatch, processed, *, crit=None, embedder=None):
    import trialmatchai.entities as ent_mod
    import trialmatchai.models.embedding as emb_mod
    import trialmatchai.registry.preparation as prep_mod

    monkeypatch.setattr(emb_mod, "build_embedder", embedder or (lambda config: "EMB"))
    monkeypatch.setattr(ent_mod, "build_entity_annotator", lambda config, embedder=None: None)
    monkeypatch.setattr(prep_mod, "prepare_trial_document", lambda doc, emb: dict(doc))
    monkeypatch.setattr(
        prep_mod,
        "prepare_criteria_documents",
        crit or (lambda doc, emb, entity_annotator=None: (processed.append(doc["nct_id"]) or [])),
    )
    monkeypatch.setattr(prep_mod, "write_prepared_trial", lambda row, folder: None)
    monkeypatch.setattr(prep_mod, "write_prepared_criteria", lambda rows, folder: 0)


def test_prepare_corpus_skips_done_and_reprepares_corrupt(tmp_path, monkeypatch):
    from trialmatchai.orchestration import prepare_corpus

    src = _make_trials(tmp_path, ["NCT1", "NCT2", "NCT3"])
    pt = tmp_path / "processed_trials"
    pt.mkdir()
    (pt / "NCT1.json").write_text('{"nct_id": "NCT1"}', encoding="utf-8")  # valid -> skip
    (pt / "NCT2.json").write_text('{"nct_id": "NCT2"', encoding="utf-8")  # corrupt -> redo

    processed = []
    _stub_prepare(monkeypatch, processed)
    stats = prepare_corpus(
        {}, trials_json_folder=src, processed_trials_folder=pt, processed_criteria_folder=tmp_path / "pc"
    )
    assert processed == ["NCT2", "NCT3"]  # NCT1 skipped; corrupt + missing reprocessed
    assert stats == {"total": 3, "prepared": 2, "skipped": 1, "failed": 0}


def test_prepare_corpus_isolates_per_trial_failure(tmp_path, monkeypatch):
    from trialmatchai.orchestration import prepare_corpus

    src = _make_trials(tmp_path, ["NCT1", "NCT2", "NCT3"])

    def crit(doc, emb, entity_annotator=None):
        if doc["nct_id"] == "NCT2":
            raise ValueError("malformed trial document")
        return []

    _stub_prepare(monkeypatch, [], crit=crit)
    stats = prepare_corpus(
        {}, trials_json_folder=src, processed_trials_folder=tmp_path / "pt",
        processed_criteria_folder=tmp_path / "pc",
    )
    assert stats == {"total": 3, "prepared": 2, "skipped": 0, "failed": 1}


def test_prepare_corpus_does_not_load_model_when_nothing_pending(tmp_path, monkeypatch):
    from trialmatchai.orchestration import prepare_corpus

    src = _make_trials(tmp_path, ["NCT1", "NCT2"])
    pt = tmp_path / "processed_trials"
    pt.mkdir()
    for nct in ("NCT1", "NCT2"):
        (pt / f"{nct}.json").write_text(json.dumps({"nct_id": nct}), encoding="utf-8")

    _stub_prepare(monkeypatch, [], embedder=_boom)  # build_embedder must never be called
    stats = prepare_corpus(
        {}, trials_json_folder=src, processed_trials_folder=pt, processed_criteria_folder=tmp_path / "pc"
    )
    assert stats == {"total": 2, "prepared": 0, "skipped": 2, "failed": 0}
