from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the TrialMatchAI batch pipeline")
    parser.add_argument("--config", default=None, help="Path to config.json")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-match every patient, ignoring existing results (default: resume/skip done).",
    )
    args = parser.parse_args()
    from trialmatchai.main import main_pipeline
    from trialmatchai.orchestration import free_models

    try:
        # Idempotent by default: skip patients with valid results. --force redoes all.
        return main_pipeline(args.config, resume=not args.force)
    finally:
        free_models()


if __name__ == "__main__":
    sys.exit(main())
