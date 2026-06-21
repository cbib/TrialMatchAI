from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from trialmatchai.config.config_loader import load_config
from trialmatchai.interop.exporters import profile_to_matching_summary
from trialmatchai.interop.importers import import_patient_path
from trialmatchai.utils.logging_config import setup_logging

logger = setup_logging(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import patient data into canonical TrialMatchAI profiles."
    )
    parser.add_argument("--config", default=None, help="Path to TrialMatchAI config JSON")
    parser.add_argument(
        "--input",
        required=True,
        help="Patient input file or OMOP extract directory.",
    )
    parser.add_argument(
        "--format",
        default="auto",
        choices=["auto", "text", "phenopacket", "fhir", "fhir-ndjson", "omop"],
        help="Input format. Defaults to auto-detection.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Profile output directory. Defaults to config patient_inputs.profile_dir.",
    )
    parser.add_argument(
        "--summary-dir",
        default=None,
        help="Matching summary output directory. Defaults to config patient_inputs.summary_dir.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on invalid or unsupported source records instead of degrading.",
    )
    parser.add_argument(
        "--no-entities",
        action="store_true",
        help="Skip model-backed entity annotation for free-text inputs.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    patient_cfg = config.get("patient_inputs", {})
    output_dir = Path(args.output_dir or patient_cfg.get("profile_dir", "data/patients/profiles"))
    summary_dir = Path(args.summary_dir or patient_cfg.get("summary_dir", "data/patients/summaries"))
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    entity_annotator = None if args.no_entities else _try_build_entity_annotator(config)
    profiles = import_patient_path(
        args.input,
        input_format=args.format,
        entity_annotator=entity_annotator,
        strict=args.strict or bool(patient_cfg.get("strict_validation", False)),
    )
    if not profiles:
        logger.error("No patient profiles were imported from %s.", args.input)
        return 1

    for profile in profiles:
        profile_path = output_dir / f"{profile.patient_id}.json"
        summary_path = summary_dir / f"{profile.patient_id}.json"
        _write_json(profile.model_dump(mode="json", exclude_none=True), profile_path)
        _write_json(profile_to_matching_summary(profile), summary_path)
        logger.info("Imported patient profile %s -> %s", profile.patient_id, profile_path)
    return 0


def _try_build_entity_annotator(config: dict[str, Any]):
    try:
        from trialmatchai.entities import build_entity_annotator

        return build_entity_annotator(config)
    except Exception as exc:
        logger.warning("Entity annotation unavailable; importing without entities: %s", exc)
        return None


def _write_json(payload: dict, path: Path) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    sys.exit(main())
