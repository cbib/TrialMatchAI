from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.e2e
def test_public_cli_synthetic_workflow(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/installed_smoke.py"
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    result = subprocess.run([sys.executable, str(script), "--workspace", str(tmp_path)],
                            env=env, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=180)
    assert result.returncode == 0, result.stdout


def test_demo_refuses_to_overwrite_existing_data(tmp_path):
    from trialmatchai.cli.demo import run_demo

    original = tmp_path / "patient.txt"
    original.write_text("do not change")
    with pytest.raises(ValueError, match="not empty"):
        run_demo(tmp_path)
    assert original.read_text() == "do not change"


def test_demo_resume_requires_its_own_workspace(tmp_path):
    from trialmatchai.cli.demo import run_demo

    with pytest.raises(ValueError, match="existing TrialMatchAI demo"):
        run_demo(tmp_path, resume=True)


@pytest.mark.parametrize("phase", ["config", "manifest"])
@pytest.mark.parametrize("failure", [OSError, KeyboardInterrupt])
def test_demo_can_restart_after_interrupted_initialization(tmp_path, monkeypatch, phase, failure):
    from trialmatchai.cli import demo
    from trialmatchai.utils.integrity import verify_manifest

    workspace = tmp_path / "demo"
    hook = "write_json_file" if phase == "config" else "write_manifest"
    original = getattr(demo, hook)

    def interrupt_after_write(*args, **kwargs):
        original(*args, **kwargs)
        raise failure("simulated initialization failure")

    with monkeypatch.context() as interrupted:
        interrupted.setattr(demo, hook, interrupt_after_write)
        with pytest.raises(failure, match="simulated initialization failure"):
            demo.run_demo(workspace)

    assert list(workspace.iterdir()) == []

    # Stop at the real pipeline boundary; this test exercises startup recovery,
    # while the installed CLI e2e covers execution after successful startup.
    class PipelineStarted(Exception):
        pass

    def pipeline_started(ctx, **kwargs):
        assert ctx.config["paths"]["output_dir"] == str(workspace / "results")
        assert "config.json" in verify_manifest(workspace / "SHA256SUMS", require_exact=True)
        raise PipelineStarted

    monkeypatch.setattr("trialmatchai.pipeline.run_pipeline", pipeline_started)
    with pytest.raises(PipelineStarted):
        demo.run_demo(workspace)


def test_demo_publish_preserves_files_added_during_initialization(tmp_path, monkeypatch):
    from trialmatchai.cli import demo

    workspace = tmp_path / "demo"
    original = demo.write_manifest

    def add_user_file_before_publish(*args, **kwargs):
        result = original(*args, **kwargs)
        (workspace / "patient.txt").write_text("preserve this file")
        return result

    monkeypatch.setattr(demo, "write_manifest", add_user_file_before_publish)
    with pytest.raises(OSError):
        demo.run_demo(workspace)
    assert (workspace / "patient.txt").read_text() == "preserve this file"
    assert not (workspace / "SHA256SUMS").exists()


@pytest.mark.parametrize("phase", ["initialization", "pipeline"])
def test_demo_interrupt_prints_the_correct_retry(tmp_path, monkeypatch, capsys, phase):
    from trialmatchai.cli import demo

    workspace = tmp_path / "demo with spaces"

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    if phase == "initialization":
        monkeypatch.setattr(demo, "write_manifest", interrupt)
    else:
        monkeypatch.setattr("trialmatchai.pipeline.run_pipeline", interrupt)
    assert demo.main(["--workdir", str(workspace)]) == 130
    error = capsys.readouterr().err
    command = error.split("To retry: ", 1)[1].strip()
    expected = ["trialmatchai", "demo", "--workdir", str(workspace)]
    if phase == "pipeline":
        expected.append("--resume")
    assert shlex.split(command) == expected
