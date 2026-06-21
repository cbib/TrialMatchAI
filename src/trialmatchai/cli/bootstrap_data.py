from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download and prepare TrialMatchAI data and model artifacts"
    )
    parser.add_argument(
        "--script",
        default="scripts/bootstrap_data.sh",
        help="Bootstrap script path relative to the repository root",
    )
    args = parser.parse_args()

    root = _repo_root()
    script = (root / args.script).resolve()
    if not script.exists():
        raise FileNotFoundError(f"Bootstrap script not found: {script}")
    return subprocess.run(["bash", str(script)], cwd=str(root), check=False).returncode


def _repo_root() -> Path:
    start = Path(__file__).resolve()
    for parent in start.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd().resolve()


if __name__ == "__main__":
    sys.exit(main())
