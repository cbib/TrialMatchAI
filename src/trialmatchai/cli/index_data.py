"""``trialmatchai index`` — prepare + build the LanceDB search tables.

A thin entry point over the same idempotent stages `build` uses (`prepare_corpus`
+ `build_index`), so there is one resumable/skip-if-done implementation of each.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from trialmatchai.config.config_loader import load_config
from trialmatchai.orchestration import build_index, prepare_corpus
from trialmatchai.utils.logging_config import setup_logging

logger = setup_logging(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare + build TrialMatchAI LanceDB search tables (idempotent)."
    )
    parser.add_argument("--config", default=None, help="Path to TrialMatchAI config JSON")
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Prepare embeddings/entities from normalized trial JSONs before indexing.",
    )
    parser.add_argument(
        "--trials-json-folder",
        default=None,
        help="Normalized trial JSONs. Defaults to config paths.trials_json_folder.",
    )
    parser.add_argument("--processed-trials-folder", default="data/processed_trials")
    parser.add_argument("--processed-criteria-folder", default="data/processed_criteria")
    parser.add_argument(
        "--force-prepare",
        action="store_true",
        help="Re-prepare every trial even if already prepared.",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Rebuild the search tables even if they already exist.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    root = _repo_root()
    processed_trials = _resolve_path(args.processed_trials_folder, root)
    processed_criteria = _resolve_path(args.processed_criteria_folder, root)

    if args.prepare:
        prepare_corpus(
            config,
            trials_json_folder=_resolve_path(
                args.trials_json_folder or config["paths"]["trials_json_folder"], root
            ),
            processed_trials_folder=processed_trials,
            processed_criteria_folder=processed_criteria,
            force=args.force_prepare,
        )

    build_index(
        config,
        processed_trials_folder=processed_trials,
        processed_criteria_folder=processed_criteria,
        force=args.reindex,
    )
    logger.info("Search tables ready.")
    return 0


def _resolve_path(value: str, root: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (root / path).resolve()


def _repo_root() -> Path:
    start = Path(__file__).resolve()
    for parent in start.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd().resolve()


if __name__ == "__main__":
    sys.exit(main())
