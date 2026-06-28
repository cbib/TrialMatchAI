"""``trialmatchai build`` — prepare the system (the setup half), once.

Builds the heavy, reusable artifacts a deployment needs before it can match:
embeds + annotates the trial corpus (``processed_*``) and builds the LanceDB
search index. Idempotent and resumable — a disrupted build re-run continues from
the last completed work — and records a manifest so you can see what is done.

  trialmatchai build                 # prepare (resumable) + index
  trialmatchai build --status        # report what is already built, then exit
  trialmatchai build --concepts      # also build the open concept-linking DB
  trialmatchai build --concepts --concepts-csv data/omop/CONCEPT.csv  # + OMOP vocab
"""

from __future__ import annotations

import argparse
import json
import sys

from trialmatchai.config.config_loader import load_config
from trialmatchai.orchestration import build_state, build_system
from trialmatchai.services.preflight import run_build_preflight
from trialmatchai.utils.logging_config import setup_logging

logger = setup_logging(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the TrialMatchAI system (prepare corpus + search index)."
    )
    parser.add_argument("--config", default=None, help="Path to TrialMatchAI config JSON")
    parser.add_argument(
        "--trials-json-folder",
        default=None,
        help="Normalized trial JSONs to prepare from. Defaults to paths.trials_json_folder.",
    )
    parser.add_argument("--processed-trials-folder", default="data/processed_trials")
    parser.add_argument("--processed-criteria-folder", default="data/processed_criteria")
    parser.add_argument(
        "--concepts",
        action="store_true",
        help="Also build the concept-linking DB from open vocabularies "
        "(genes, diseases, chemicals, cell lines, cell types, phenotypes; auto-downloaded).",
    )
    parser.add_argument(
        "--concepts-csv",
        default=None,
        help="OMOP CONCEPT.csv to add SNOMED/LOINC/RxNorm to the concept DB (optional).",
    )
    parser.add_argument("--synonym-csv", default=None, help="OMOP CONCEPT_SYNONYM.csv (optional).")
    parser.add_argument("--force-prepare", action="store_true", help="Re-prepare all trials.")
    parser.add_argument("--reindex", action="store_true", help="Rebuild the index even if present.")
    parser.add_argument(
        "--status", action="store_true", help="Print what is already built and exit."
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.status:
        state = build_state(
            config,
            processed_trials_folder=args.processed_trials_folder,
            processed_criteria_folder=args.processed_criteria_folder,
        )
        print(json.dumps(state, indent=2))
        ready = state["ready_to_match"]
        logger.info("System %s to match.", "READY" if ready else "NOT ready")
        return 0 if ready else 1

    # Fail fast on missing GPU / extras / HF access before any heavy work.
    preflight_issues = run_build_preflight(config)
    if preflight_issues:
        logger.error("Build aborted: resolve the %s issue(s) above.", len(preflight_issues))
        return 1

    build_system(
        config,
        trials_json_folder=args.trials_json_folder,
        processed_trials_folder=args.processed_trials_folder,
        processed_criteria_folder=args.processed_criteria_folder,
        force_prepare=args.force_prepare,
        force_reindex=args.reindex,
    )

    # Optional: chain the concept-DB build. --concepts pulls the open vocabularies
    # (auto-downloaded); --concepts-csv adds the licensed OMOP vocab on top.
    if args.concepts or args.concepts_csv:
        logger.info("=== build: concepts stage ===")
        from trialmatchai.cli.build_concepts import run_build_concepts

        run_build_concepts(
            config,
            sources="open" if args.concepts else None,
            concept_csv=args.concepts_csv,
            synonym_csv=args.synonym_csv,
        )
    else:
        logger.warning(
            "Concept DB not built: entity->concept linking will degrade gracefully. "
            "Run `trialmatchai build --concepts` to enable it (open vocabularies, "
            "auto-downloaded); add --concepts-csv for OMOP SNOMED/LOINC/RxNorm."
        )

    state = build_state(
        config,
        processed_trials_folder=args.processed_trials_folder,
        processed_criteria_folder=args.processed_criteria_folder,
    )
    logger.info("Build done. ready_to_match=%s", state["ready_to_match"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
