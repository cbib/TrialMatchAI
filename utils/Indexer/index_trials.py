#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "source"
if str(SOURCE) not in sys.path:
    sys.path.append(str(SOURCE))

from Matcher.config.config_loader import load_config  # noqa: E402
from Matcher.search import LanceDBSearchBackend  # noqa: E402


def load_processed(folder: Path) -> list[dict]:
    docs: list[dict] = []
    for path in sorted(folder.glob("*.json")):
        docs.append(json.loads(path.read_text(encoding="utf-8")))
    return docs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create or update the LanceDB trial search table."
    )
    parser.add_argument("--config", default=None, help="Path to TrialMatchAI config JSON")
    parser.add_argument(
        "--processed-folder",
        required=True,
        help="Folder of prepared trial JSON files",
    )
    parser.add_argument("--db-path", default=None, help="Override search DB path")
    parser.add_argument("--table", default=None, help="Override trials table name")
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
        search_cfg["trials_table"] = args.table

    processed_path = Path(args.processed_folder)
    docs = load_processed(processed_path)
    if not docs:
        print(f"No prepared trial JSON files found in {processed_path}.")
        return 1

    backend = LanceDBSearchBackend.from_config(config)
    count = backend.index_trials(docs, recreate=args.recreate)
    print(
        f"Indexed {count} trial documents into "
        f"{backend.db_path}/{backend.trials_table}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
