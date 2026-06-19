from __future__ import annotations

import argparse
import sys

from Matcher.config.config_loader import load_config
from Matcher.services.elasticsearch_service import (
    build_elasticsearch_client,
    ensure_elasticsearch,
)
from Matcher.services.preflight import run_preflight_checks
from Matcher.utils.logging_config import setup_logging

logger = setup_logging(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="TrialMatchAI healthcheck")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config.json",
    )
    parser.add_argument(
        "--start-es",
        action="store_true",
        help="Attempt to start Elasticsearch if unreachable",
    )
    parser.add_argument(
        "--require-indices",
        action="store_true",
        help="Fail if configured Elasticsearch indices are missing",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    issues = 0

    preflight_issues = run_preflight_checks(
        config,
        require_patient_inputs=False,
        require_trials_json=False,
        require_models=False,
    )
    issues += len(preflight_issues)

    es_cfg = config["elasticsearch"]
    es_client = build_elasticsearch_client(config)

    if args.start_es and es_cfg.get("auto_start") is False:
        es_cfg["auto_start"] = True

    if not ensure_elasticsearch(es_client, config):
        logger.error("Elasticsearch healthcheck failed.")
        issues += 1
    else:
        logger.info("Elasticsearch reachable.")

    if args.require_indices:
        issues += len(
            run_preflight_checks(
                config,
                es_client=es_client,
                require_indices=True,
            )
        )

    return 1 if issues else 0
if __name__ == "__main__":
    sys.exit(main())
