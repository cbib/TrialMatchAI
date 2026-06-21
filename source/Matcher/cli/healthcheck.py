from __future__ import annotations

import argparse
import sys

from Matcher.config.config_loader import load_config
from Matcher.search import LanceDBSearchBackend
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
        "--require-tables",
        action="store_true",
        help="Fail if configured LanceDB search tables are missing",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    issues = 0
    search_backend = LanceDBSearchBackend.from_config(config)

    preflight_issues = run_preflight_checks(
        config,
        require_patient_inputs=False,
        require_trials_json=False,
        require_models=False,
    )
    issues += len(preflight_issues)

    backend_issues = search_backend.health(require_tables=args.require_tables)
    if backend_issues:
        for issue in backend_issues:
            logger.error("Search backend healthcheck failed: %s", issue)
        issues += len(backend_issues)
    else:
        logger.info("LanceDB search backend reachable at %s.", search_backend.db_path)

    return 1 if issues else 0
if __name__ == "__main__":
    sys.exit(main())
