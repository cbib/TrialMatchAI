"""P2: the REAL pipeline stage wrappers (not fake stages) — exit-code handling
and skip/forward behavior (test/codebase audit). Kept in a dedicated file so it
never append-collides with other branches editing test_pipeline.py."""

from __future__ import annotations

import pytest


def test_run_match_raises_on_nonzero_exit(monkeypatch):
    from trialmatchai import orchestration as orch
    from trialmatchai.pipeline import StageContext, _run_match

    monkeypatch.setattr(orch, "run_matching", lambda config, **k: 5)
    with pytest.raises(RuntimeError):
        _run_match(StageContext(config={}))


def test_run_eval_noops_without_qrels():
    from trialmatchai.pipeline import StageContext, _run_eval

    _run_eval(StageContext(config={}))  # qrels None -> returns cleanly, no trec import


def test_run_ingest_noops_without_inputs():
    from trialmatchai.pipeline import StageContext, _run_ingest

    _run_ingest(StageContext(config={}))  # no inputs -> returns cleanly


def test_run_prepare_forwards_force_and_folders(monkeypatch, tmp_path):
    from pathlib import Path

    from trialmatchai import orchestration as orch
    from trialmatchai.pipeline import StageContext, _run_prepare

    captured = {}
    monkeypatch.setattr(orch, "prepare_corpus", lambda config, **k: captured.update(k) or {})
    ctx = StageContext(
        config={"paths": {"trials_json_folder": "TJ"}},
        processed_trials_folder=tmp_path / "pt",
        processed_criteria_folder=tmp_path / "pc",
        force={"prepare"},
    )
    _run_prepare(ctx)
    assert captured["force"] is True
    assert captured["trials_json_folder"] == "TJ"
    assert Path(captured["processed_trials_folder"]) == tmp_path / "pt"
