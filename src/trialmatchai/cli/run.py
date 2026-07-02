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
    from trialmatchai.config.config_loader import load_config
    from trialmatchai.orchestration import free_models, run_matching

    config = load_config(args.config)
    try:
        # Via run_matching (not main_pipeline) so corpus-change resume invalidation
        # applies — otherwise a re-index would serve stale ranked_trials.json.
        return run_matching(config, resume=not args.force)
    finally:
        free_models()


if __name__ == "__main__":
    sys.exit(main())
