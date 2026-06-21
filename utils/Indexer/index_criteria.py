#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "source"
if str(SOURCE) not in sys.path:
    sys.path.append(str(SOURCE))

from Matcher.config.config_loader import load_config  # noqa: E402
from Matcher.search import LanceDBSearchBackend  # noqa: E402


logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class CriteriaIndexer:
    def __init__(
        self,
        backend: LanceDBSearchBackend,
        *,
        processed_file: Path,
    ) -> None:
        self.backend = backend
        self.processed_file = processed_file
        self.processed_file.parent.mkdir(parents=True, exist_ok=True)
        self.processed_ids = self._load_processed_ids()

    def _load_processed_ids(self) -> set[str]:
        if self.processed_file.exists():
            return set(self.processed_file.read_text(encoding="utf-8").splitlines())
        return set()

    def _save_processed_ids(self) -> None:
        self.processed_file.write_text(
            "\n".join(sorted(self.processed_ids)) + "\n",
            encoding="utf-8",
        )

    def load_docs(
        self,
        processed_folder: Path,
        *,
        recreate: bool,
    ) -> tuple[list[dict], set[str]]:
        docs: list[dict] = []
        completed: set[str] = set()
        trial_dirs = sorted(path for path in processed_folder.iterdir() if path.is_dir())
        for trial_dir in trial_dirs:
            nct_id = trial_dir.name
            if not recreate and nct_id in self.processed_ids:
                logger.info("Skipping %s: already indexed", nct_id)
                continue
            trial_docs = []
            for path in sorted(trial_dir.glob("*.json")):
                try:
                    trial_docs.append(json.loads(path.read_text(encoding="utf-8")))
                except Exception as exc:
                    logger.warning("%s: failed to load %s: %s", nct_id, path.name, exc)
            if trial_docs:
                docs.extend(trial_docs)
            completed.add(nct_id)
        return docs, completed

    def index_all(
        self,
        processed_folder: Path,
        *,
        recreate: bool = True,
    ) -> int:
        if not processed_folder.exists():
            raise FileNotFoundError(f"Criteria folder not found: {processed_folder}")
        docs, completed = self.load_docs(processed_folder, recreate=recreate)
        if not docs:
            logger.info("No prepared criteria JSON files found.")
            return 0
        count = self.backend.index_criteria(docs, recreate=recreate)
        self.processed_ids.update(completed)
        self._save_processed_ids()
        logger.info(
            "Indexed %s criteria documents into %s/%s.",
            count,
            self.backend.db_path,
            self.backend.criteria_table,
        )
        return count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create or update the LanceDB eligibility criteria search table."
    )
    parser.add_argument("--config", default=None, help="Path to TrialMatchAI config JSON")
    parser.add_argument(
        "--processed-folder",
        required=True,
        help="Root folder containing one prepared criteria subfolder per trial",
    )
    parser.add_argument("--db-path", default=None, help="Override search DB path")
    parser.add_argument("--table", default=None, help="Override criteria table name")
    parser.add_argument(
        "--processed-file",
        default="utils/Indexer/processed_ids.txt",
        help="File used to track already appended trial IDs",
    )
    parser.add_argument(
        "--recreate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overwrite the target table before writing.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    search_cfg = config["search_backend"]
    if args.db_path:
        search_cfg["db_path"] = str(Path(args.db_path).expanduser().resolve())
    if args.table:
        search_cfg["criteria_table"] = args.table

    backend = LanceDBSearchBackend.from_config(config)
    indexer = CriteriaIndexer(
        backend=backend,
        processed_file=(ROOT / args.processed_file).resolve(),
    )
    count = indexer.index_all(
        Path(args.processed_folder),
        recreate=args.recreate,
    )
    return 0 if count else 1


if __name__ == "__main__":
    raise SystemExit(main())
