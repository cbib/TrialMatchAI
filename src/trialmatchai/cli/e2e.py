"""``trialmatchai e2e`` — match a patient end-to-end in one command.

A preset over the unified pipeline (the slice index -> ingest -> expand -> match):
imports patient inputs (any supported format, auto-detected), ensures the search
index, and matches. Every stage is idempotent: re-running skips an existing index,
already-imported patients, and already-matched patients.
"""

from __future__ import annotations

import argparse
import sys

from trialmatchai.config.config_loader import load_config
from trialmatchai.orchestration import run_e2e
from trialmatchai.utils.logging_config import setup_logging

logger = setup_logging(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Match a patient end-to-end (pipeline slice: index -> ingest -> expand -> match)."
    )
    parser.add_argument("--config", default=None, help="Path to TrialMatchAI config JSON")
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        help="Patient input file or directory (repeatable). Format is auto-detected. "
        "Omit if profiles are already staged.",
    )
    parser.add_argument(
        "--format",
        default="auto",
        choices=["auto", "text", "phenopacket", "fhir", "fhir-ndjson", "omop"],
        help="Input format for --input. Defaults to auto-detection.",
    )
    parser.add_argument(
        "--processed-trials-folder",
        default="data/processed_trials",
        help="Folder of prepared trial JSON files used to build the index.",
    )
    parser.add_argument(
        "--processed-criteria-folder",
        default="data/processed_criteria",
        help="Folder of prepared criteria subfolders used to build the index.",
    )
    parser.add_argument("--no-entities", action="store_true", help="Skip entity annotation on ingest.")
    parser.add_argument("--reingest", action="store_true", help="Re-import patients even if profiles exist.")
    parser.add_argument("--reindex", action="store_true", help="Rebuild the search index even if present.")
    parser.add_argument("--rematch", action="store_true", help="Re-match patients even if results exist.")
    args = parser.parse_args()

    config = load_config(args.config)
    return run_e2e(
        config,
        args.input,
        input_format=args.format,
        with_entities=not args.no_entities,
        processed_trials_folder=args.processed_trials_folder,
        processed_criteria_folder=args.processed_criteria_folder,
        force_reingest=args.reingest,
        force_reindex=args.reindex,
        force_rematch=args.rematch,
    )


if __name__ == "__main__":
    sys.exit(main())
