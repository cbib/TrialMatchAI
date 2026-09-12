from __future__ import annotations

import os
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
