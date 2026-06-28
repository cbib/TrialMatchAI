"""``trialmatchai pipeline`` — the one end-to-end pipeline, or any slice of it.

Default (no selection) runs every stage; each is idempotent, so finished work is
skipped. Select with --only/--from/--to, omit with --skip (handy for ablation),
and redo with --force.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from trialmatchai.config.config_loader import load_config
from trialmatchai.pipeline import STAGE_NAMES, StageContext, run_pipeline
from trialmatchai.utils.logging_config import setup_logging

logger = setup_logging(__name__)


def _split(value: str | None) -> list[str]:
    return [v.strip() for v in (value or "").split(",") if v.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="trialmatchai pipeline",
        description=(
            "Run the unified TrialMatchAI pipeline or any slice. Stages in order: "
            + ", ".join(STAGE_NAMES)
            + ". Every stage is idempotent — finished work is skipped automatically."
        ),
    )
    parser.add_argument("--config", default=None, help="Path to config.json")
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        help="Patient input file/dir to ingest (repeatable).",
    )
    parser.add_argument("--format", default="auto", help="Patient input format (default: auto).")
    parser.add_argument(
        "--no-entities", action="store_true", help="Ingest patients without entity annotation."
    )
    parser.add_argument(
        "--trials-json-folder",
        default=None,
        help="Normalized trial JSONs for prepare. Defaults to config paths.trials_json_folder.",
    )
    parser.add_argument("--processed-trials-folder", default="data/processed_trials")
    parser.add_argument("--processed-criteria-folder", default="data/processed_criteria")
    parser.add_argument(
        "--concepts",
        action="store_true",
        help="In the concepts stage, build the open-vocabulary concept store.",
    )
    parser.add_argument("--concepts-csv", default=None, help="OMOP CONCEPT.csv for the concepts stage.")
    parser.add_argument("--synonym-csv", default=None, help="OMOP CONCEPT_SYNONYM.csv.")

    sel = parser.add_argument_group("stage selection")
    sel.add_argument("--only", default=None, metavar="STAGES", help="Run only these stages.")
    sel.add_argument(
        "--skip",
        default="",
        metavar="STAGES",
        help="Skip these stages (e.g. ablation: --skip expand).",
    )
    sel.add_argument("--from", dest="from_stage", default=None, metavar="STAGE", help="First stage.")
    sel.add_argument("--to", dest="to_stage", default=None, metavar="STAGE", help="Last stage.")
    sel.add_argument(
        "--force",
        default="",
        metavar="STAGES",
        help="Re-run these stages even if done ('all' forces everything).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    ctx = StageContext(
        config=config,
        trials_json_folder=Path(args.trials_json_folder) if args.trials_json_folder else None,
        processed_trials_folder=Path(args.processed_trials_folder),
        processed_criteria_folder=Path(args.processed_criteria_folder),
        inputs=list(args.input),
        input_format=args.format,
        with_entities=not args.no_entities,
        concepts="open" if args.concepts else None,
        concept_csv=args.concepts_csv,
        synonym_csv=args.synonym_csv,
        force=set(_split(args.force)),
    )
    try:
        return run_pipeline(
            ctx,
            only=_split(args.only) or None,
            skip=_split(args.skip),
            from_stage=args.from_stage,
            to_stage=args.to_stage,
        )
    except ValueError as exc:
        parser.error(str(exc))
        return 2  # pragma: no cover - parser.error exits


if __name__ == "__main__":
    sys.exit(main())
