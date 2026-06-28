from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the TrialMatchAI batch pipeline")
    parser.add_argument("--config", default=None, help="Path to config.json")
    args = parser.parse_args()
    from trialmatchai.main import main_pipeline
    from trialmatchai.orchestration import free_models

    try:
        return main_pipeline(args.config)
    finally:
        free_models()


if __name__ == "__main__":
    sys.exit(main())
