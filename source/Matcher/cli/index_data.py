from __future__ import annotations

import argparse
import sys
from pathlib import Path

from Matcher.config.config_loader import load_config
from Matcher.search import LanceDBSearchBackend
from Matcher.utils.logging_config import setup_logging

logger = setup_logging(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build TrialMatchAI LanceDB search tables from prepared JSON data."
    )
    parser.add_argument("--config", default=None, help="Path to TrialMatchAI config JSON")
    parser.add_argument(
        "--processed-trials-folder",
        default="data/processed_trials",
        help="Folder containing prepared trial JSON files.",
    )
    parser.add_argument(
        "--processed-criteria-folder",
        default="data/processed_criteria",
        help="Folder containing prepared criteria subfolders.",
    )
    parser.add_argument(
        "--skip-trials",
        action="store_true",
        help="Do not build the trial table.",
    )
    parser.add_argument(
        "--skip-criteria",
        action="store_true",
        help="Do not build the criteria table.",
    )
    parser.add_argument(
        "--recreate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overwrite target tables before writing.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    backend = LanceDBSearchBackend.from_config(config)
    root = _repo_root()
    failures = 0

    if not args.skip_trials:
        trials_folder = _resolve_path(args.processed_trials_folder, root)
        trial_docs = _load_flat_json_folder(trials_folder)
        if not trial_docs:
            logger.error("No prepared trial JSON files found in %s.", trials_folder)
            failures += 1
        else:
            count = backend.index_trials(trial_docs, recreate=args.recreate)
            logger.info("Indexed %s trial documents.", count)

    if not args.skip_criteria:
        criteria_folder = _resolve_path(args.processed_criteria_folder, root)
        criteria_docs = _load_nested_json_folder(criteria_folder)
        if not criteria_docs:
            logger.error("No prepared criteria JSON files found in %s.", criteria_folder)
            failures += 1
        else:
            count = backend.index_criteria(criteria_docs, recreate=args.recreate)
            logger.info("Indexed %s criteria documents.", count)

    if failures:
        return 1
    logger.info("Search tables ready at %s.", backend.db_path)
    return 0


def _load_flat_json_folder(folder: Path) -> list[dict]:
    if not folder.exists():
        return []
    return [
        _read_json(path)
        for path in sorted(folder.glob("*.json"))
        if path.is_file()
    ]


def _load_nested_json_folder(folder: Path) -> list[dict]:
    if not folder.exists():
        return []
    return [
        _read_json(path)
        for path in sorted(folder.glob("*/*.json"))
        if path.is_file()
    ]


def _read_json(path: Path) -> dict:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


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
