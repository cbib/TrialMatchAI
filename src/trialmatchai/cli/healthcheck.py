from __future__ import annotations

import argparse
import sys

from trialmatchai.config.config_loader import load_config
from trialmatchai.search import build_search_backend
from trialmatchai.services.preflight import run_preflight_checks
from trialmatchai.utils.logging_config import setup_logging

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
    parser.add_argument(
        "--registry",
        action="store_true",
        help="Check registry updater paths and manifest readability.",
    )
    parser.add_argument(
        "--models",
        action="store_true",
        help="Require configured model artifacts and optional model dependencies.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    issues = 0
    search_backend = build_search_backend(config)

    preflight_issues = run_preflight_checks(
        config,
        require_patient_inputs=False,
        require_trials_json=False,
        require_models=args.models,
    )
    issues += len(preflight_issues)

    backend_issues = search_backend.health(require_tables=args.require_tables)
    if backend_issues:
        for issue in backend_issues:
            logger.error("Search backend healthcheck failed: %s", issue)
        issues += len(backend_issues)
    else:
        logger.info("LanceDB search backend reachable at %s.", search_backend.db_path)

    if args.registry:
        registry_issues = _check_registry(config)
        for issue in registry_issues:
            logger.error("Registry healthcheck failed: %s", issue)
        issues += len(registry_issues)

    return 1 if issues else 0


def _check_registry(config: dict) -> list[str]:
    from pathlib import Path

    issues: list[str] = []
    registry_cfg = config.get("registry", {})
    paths_cfg = config.get("paths", {})
    for key in ("raw_dir", "reports_dir"):
        value = registry_cfg.get(key)
        if not value:
            issues.append(f"registry.{key} is not configured.")
            continue
        try:
            Path(value).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            issues.append(f"registry.{key} is not writable: {value} ({exc})")

    trials_folder = paths_cfg.get("trials_json_folder")
    if not trials_folder:
        issues.append("paths.trials_json_folder is not configured.")
    else:
        try:
            Path(trials_folder).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            issues.append(f"paths.trials_json_folder is not writable: {trials_folder} ({exc})")

    manifest_path = registry_cfg.get("manifest_path")
    if not manifest_path:
        issues.append("registry.manifest_path is not configured.")
    else:
        manifest = Path(manifest_path)
        try:
            manifest.parent.mkdir(parents=True, exist_ok=True)
            if manifest.exists():
                with manifest.open("r", encoding="utf-8"):
                    pass
        except OSError as exc:
            issues.append(f"registry.manifest_path is not readable: {manifest} ({exc})")
    return issues
if __name__ == "__main__":
    sys.exit(main())
