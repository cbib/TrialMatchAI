from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from trialmatchai.config.config_loader import load_config
from trialmatchai.registry.clinicaltrials_gov import ClinicalTrialsGovClient
from trialmatchai.registry.defaults import DEFAULT_REGISTRY_STATUSES
from trialmatchai.registry.updater import (
    RegistryUpdateConfig,
    RegistryUpdater,
    normalize_keywords,
)
from trialmatchai.search import InMemorySearchBackend, LanceDBSearchBackend
from trialmatchai.utils.logging_config import setup_logging

logger = setup_logging(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch changed ClinicalTrials.gov studies and upsert LanceDB tables."
    )
    parser.add_argument("--config", default=None, help="Path to TrialMatchAI config JSON")
    parser.add_argument(
        "--keyword",
        action="append",
        default=[],
        help="ClinicalTrials.gov keyword query. Repeat for multiple queries.",
    )
    parser.add_argument(
        "--keywords-file",
        default=None,
        help="Text file with one ClinicalTrials.gov keyword query per line.",
    )
    parser.add_argument("--since", default=None, help="Only process updates since YYYY-MM-DD")
    parser.add_argument("--max-studies", type=int, default=None, help="Maximum studies to process")
    parser.add_argument(
        "--status",
        action="append",
        default=[],
        help="ClinicalTrials.gov overall status filter. Repeat for multiple statuses.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Fetch and plan without writing")
    parser.add_argument(
        "--reindex-all-changed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recompute embeddings/entities and upsert changed studies.",
    )
    parser.add_argument(
        "--report-path",
        default=None,
        help="Optional exact path for a copy of the run report JSON.",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Run continuously (server mode): update every --interval seconds.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=86400.0,
        help="Seconds between updates in --watch mode (default 24h).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    registry_cfg = config.get("registry", {})
    paths_cfg = config.get("paths", {})

    since_days = int(registry_cfg.get("since_days", 7))
    since = _resolve_since(args.since, since_days)
    keywords = normalize_keywords(
        [
            *args.keyword,
            *_read_keywords_file(
                args.keywords_file or registry_cfg.get("keywords_file")
            ),
        ]
    )
    statuses = tuple(args.status or DEFAULT_REGISTRY_STATUSES)
    # Explicit --max-studies 0 is a valid cap; only fall back to config when truly unset.
    max_studies = (
        args.max_studies if args.max_studies is not None else registry_cfg.get("max_studies")
    )
    if max_studies is not None:
        max_studies = int(max_studies)

    update_config = RegistryUpdateConfig(
        raw_dir=Path(registry_cfg.get("raw_dir", "data/registry/raw")),
        normalized_trials_dir=Path(paths_cfg.get("trials_json_folder", "data/trials_jsons")),
        manifest_path=Path(
            registry_cfg.get("manifest_path", "data/registry/manifest.jsonl")
        ),
        reports_dir=Path(registry_cfg.get("reports_dir", "data/registry/runs")),
        keywords=keywords,
        statuses=statuses,
        since=since,
        max_studies=max_studies,
        dry_run=args.dry_run,
        reindex_all_changed=args.reindex_all_changed,
        failure_threshold=float(registry_cfg.get("failure_threshold", 0.25)),
    )

    client = ClinicalTrialsGovClient(
        base_url=registry_cfg.get(
            "api_base_url",
            "https://clinicaltrials.gov/api/v2/studies",
        ),
        timeout=float(registry_cfg.get("request_timeout", 30.0)),
        rate_limit_per_second=float(registry_cfg.get("rate_limit_per_second", 2.0)),
    )
    backend = InMemorySearchBackend() if args.dry_run else LanceDBSearchBackend.from_config(config)
    embedder = _NullEmbedder() if args.dry_run else _build_embedder(config)
    entity_annotator = None if args.dry_run else _build_entity_annotator(config, embedder)

    updater = RegistryUpdater(
        client=client,
        backend=backend,
        embedder=embedder,
        entity_annotator=entity_annotator,
    )
    if not args.watch:
        return _run_and_report(updater, update_config, args.report_path)

    logger.info(
        "Registry watch mode: updating every %.0fs (Ctrl-C to stop).", args.interval
    )
    while True:
        # Slide the lookback window forward each cycle (unless --since is pinned);
        # manifest dedup makes the window overlap cheap.
        cycle_config = replace(update_config, since=_resolve_since(args.since, since_days))
        try:
            _run_and_report(updater, cycle_config, args.report_path)
        except Exception:
            logger.exception("Registry update cycle failed; retrying next interval")
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            logger.info("Registry watch stopped.")
            return 0


def _run_and_report(updater: RegistryUpdater, update_config, report_path) -> int:
    report = updater.run(update_config)
    if report_path:
        _write_report_copy(Path(report_path), report.to_dict())
    logger.info("Registry update report: %s", json.dumps(report.to_dict(), sort_keys=True))
    return 1 if report.failure_rate > update_config.failure_threshold else 0


class _NullEmbedder:
    def embed_texts(self, texts):
        raise RuntimeError("Dry-run updater must not request embeddings.")


def _build_embedder(config: dict[str, Any]):
    from trialmatchai.models.embedding import build_embedder

    return build_embedder(config)


def _build_entity_annotator(config: dict[str, Any], embedder):
    from trialmatchai.entities import build_entity_annotator

    return build_entity_annotator(config, embedder=embedder)


def _read_keywords_file(path: str | None) -> list[str]:
    if not path:
        return []
    keyword_path = Path(path)
    if not keyword_path.exists():
        raise FileNotFoundError(f"Registry keywords file does not exist: {keyword_path}")
    return [
        line.strip()
        for line in keyword_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _resolve_since(value: str | None, since_days: int) -> date | None:
    if value:
        return date.fromisoformat(value)
    if since_days <= 0:
        return None
    return date.today() - timedelta(days=since_days)


def _write_report_copy(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
