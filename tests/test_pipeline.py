"""The unified pipeline driver: stage selection, force semantics, teardown."""

import pytest

from trialmatchai import pipeline
from trialmatchai.pipeline import StageContext, select_stages


def _names(stages):
    return [s.name for s in stages]


def test_default_selection_is_every_stage_in_order():
    assert _names(select_stages()) == list(pipeline.STAGE_NAMES)


def test_only_selects_exactly_those_in_canonical_order():
    assert _names(select_stages(only=["match", "prepare"])) == ["prepare", "match"]


def test_from_to_slice():
    assert _names(select_stages(from_stage="index", to_stage="match")) == [
        "index",
        "ingest",
        "expand",
        "match",
    ]


def test_to_alone_runs_the_build_half():
    assert _names(select_stages(to_stage="index")) == ["prepare", "concepts", "link", "index"]


def test_from_alone_runs_the_run_half():
    assert _names(select_stages(from_stage="ingest")) == ["ingest", "expand", "match", "eval"]


def test_skip_removes_a_stage():
    assert "expand" not in _names(select_stages(skip=["expand"]))


def test_unknown_stage_raises():
    with pytest.raises(ValueError):
        select_stages(only=["nonsense"])
    with pytest.raises(ValueError):
        select_stages(skip=["nope"])
    with pytest.raises(ValueError):
        select_stages(from_stage="bogus")


def test_from_after_to_raises():
    with pytest.raises(ValueError):
        select_stages(from_stage="match", to_stage="prepare")


def test_forced_semantics():
    ctx = StageContext(config={}, force={"match"})
    assert ctx.forced("match") is True
    assert ctx.forced("prepare") is False
    assert StageContext(config={}, force={"all"}).forced("anything") is True


def _fake_stages(record):
    return tuple(
        pipeline.Stage(n, (lambda name: lambda ctx: record.append(name))(n), "")
        for n in ("a", "b", "c")
    )


def test_run_pipeline_runs_selected_in_order_and_frees_models(monkeypatch):
    ran, freed = [], []
    fakes = _fake_stages(ran)
    monkeypatch.setattr(pipeline, "STAGES", fakes)
    monkeypatch.setattr(pipeline, "STAGE_NAMES", tuple(s.name for s in fakes))
    monkeypatch.setattr("trialmatchai.orchestration.free_models", lambda: freed.append(1))

    rc = pipeline.run_pipeline(StageContext(config={}), only=["a", "c"])
    assert rc == 0
    assert ran == ["a", "c"]
    assert freed == [1]  # teardown ran once


def test_run_pipeline_frees_models_even_on_stage_error(monkeypatch):
    freed = []

    def boom(ctx):
        raise RuntimeError("stage failed")

    fakes = (pipeline.Stage("a", boom, ""),)
    monkeypatch.setattr(pipeline, "STAGES", fakes)
    monkeypatch.setattr(pipeline, "STAGE_NAMES", ("a",))
    monkeypatch.setattr("trialmatchai.orchestration.free_models", lambda: freed.append(1))

    with pytest.raises(RuntimeError):
        pipeline.run_pipeline(StageContext(config={}))
    assert freed == [1]  # GPU freed despite the failure


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
