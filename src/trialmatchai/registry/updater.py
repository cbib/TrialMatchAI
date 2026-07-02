from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from trialmatchai.registry.clinicaltrials_gov import ClinicalTrialsGovClient
from trialmatchai.registry.defaults import DEFAULT_REGISTRY_KEYWORDS, DEFAULT_REGISTRY_STATUSES
from trialmatchai.registry.manifest import (
    ManifestRecord,
    RegistryManifest,
    source_hash,
    utc_now_iso,
)
from trialmatchai.registry.normalization import normalize_study
from trialmatchai.registry.preparation import (
    EntityAnnotationBackend,
    TextEmbeddingBackend,
    prepare_criteria_documents,
    prepare_trial_document,
)
from trialmatchai.search import LanceDBSearchBackend
from trialmatchai.utils.logging_config import setup_logging

logger = setup_logging(__name__)


@dataclass(frozen=True)
class RegistryUpdateConfig:
    raw_dir: Path
    normalized_trials_dir: Path
    manifest_path: Path
    reports_dir: Path
    keywords: tuple[str, ...] = DEFAULT_REGISTRY_KEYWORDS
    statuses: tuple[str, ...] = DEFAULT_REGISTRY_STATUSES
    since: date | None = None
    max_studies: int | None = None
    dry_run: bool = False
    reindex_all_changed: bool = True
    failure_threshold: float = 0.25


@dataclass
class RegistryStudyFailure:
    nct_id: str | None
    error: str


@dataclass
class RegistryUpdateReport:
    fetched: int = 0
    new: int = 0
    changed: int = 0
    unchanged: int = 0
    failed: int = 0
    duplicate: int = 0
    indexed: int = 0
    criteria_indexed: int = 0
    dry_run: bool = False
    failures: list[RegistryStudyFailure] = field(default_factory=list)

    @property
    def failure_rate(self) -> float:
        denominator = max(1, self.fetched)
        return self.failed / denominator

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["failure_rate"] = self.failure_rate
        return data


class RegistryUpdater:
    def __init__(
        self,
        *,
        client: ClinicalTrialsGovClient,
        backend: LanceDBSearchBackend,
        embedder: TextEmbeddingBackend,
        entity_annotator: EntityAnnotationBackend | None = None,
    ) -> None:
        self.client = client
        self.backend = backend
        self.embedder = embedder
        self.entity_annotator = entity_annotator

    def run(self, config: RegistryUpdateConfig) -> RegistryUpdateReport:
        report = RegistryUpdateReport(dry_run=config.dry_run)
        manifest = RegistryManifest(config.manifest_path)
        latest = manifest.load_latest()
        seen: set[str] = set()

        for keyword in config.keywords:
            remaining = _remaining(config.max_studies, report.fetched)
            if remaining == 0:
                break
            logger.info("Fetching registry studies for keyword: %s", keyword)
            try:
                for study in self.client.iter_studies(
                    keyword=keyword,
                    statuses=config.statuses,
                    since=config.since,
                    max_studies=remaining,
                ):
                    if (
                        config.max_studies is not None
                        and report.fetched >= config.max_studies
                    ):
                        break
                    self._process_study(
                        study,
                        config=config,
                        manifest=manifest,
                        latest=latest,
                        seen=seen,
                        report=report,
                    )
            except Exception as exc:
                logger.exception("Registry source fetch failed for keyword: %s", keyword)
                report.failed += 1
                report.failures.append(
                    RegistryStudyFailure(
                        nct_id=None,
                        error=f"{keyword}: {exc}",
                    )
                )

        if not config.dry_run:
            self.write_run_report(config.reports_dir, report)
        return report

    def _process_study(
        self,
        study: dict[str, Any],
        *,
        config: RegistryUpdateConfig,
        manifest: RegistryManifest,
        latest: dict[str, ManifestRecord],
        seen: set[str],
        report: RegistryUpdateReport,
    ) -> None:
        nct_id: str | None = None
        try:
            normalized = normalize_study(study)
            nct_id = str(normalized["nct_id"])
            if nct_id in seen:
                report.duplicate += 1
                return
            seen.add(nct_id)
            report.fetched += 1

            digest = source_hash(study)
            previous = latest.get(nct_id)
            # Skip only if unchanged and previously succeeded; "failed" records must be retried.
            if (
                previous
                and previous.source_hash == digest
                and previous.processing_status in {"indexed", "fetched"}
            ):
                report.unchanged += 1
                return

            is_new = previous is None
            if is_new:
                report.new += 1
            else:
                report.changed += 1

            if config.dry_run:
                return

            self._write_source_and_normalized(study, normalized, config=config)
            if config.reindex_all_changed:
                prepared_trial = prepare_trial_document(normalized, self.embedder)
                prepared_criteria = prepare_criteria_documents(
                    normalized,
                    self.embedder,
                    entity_annotator=self.entity_annotator,
                )
                report.indexed += self.backend.upsert_trials([prepared_trial])
                report.criteria_indexed += self.backend.replace_criteria_for_trials(
                    [nct_id],
                    prepared_criteria,
                )

            record = ManifestRecord(
                nct_id=nct_id,
                source_url=str(normalized.get("source_url", "")),
                source_hash=digest,
                fetched_at=utc_now_iso(),
                last_update_posted=normalized.get("last_update_posted"),
                processing_status="indexed" if config.reindex_all_changed else "fetched",
            )
            manifest.append(record)
            latest[nct_id] = record
        except Exception as exc:
            logger.exception("Registry update failed for study %s", nct_id or "<unknown>")
            report.failed += 1
            report.failures.append(
                RegistryStudyFailure(nct_id=nct_id, error=str(exc))
            )
            if not config.dry_run and nct_id:
                manifest.append(
                    ManifestRecord(
                        nct_id=nct_id,
                        source_url=f"https://clinicaltrials.gov/study/{nct_id}",
                        source_hash=source_hash(study),
                        fetched_at=utc_now_iso(),
                        last_update_posted=None,
                        processing_status="failed",
                        error_summary=str(exc),
                    )
                )

    def _write_source_and_normalized(
        self,
        study: dict[str, Any],
        normalized: dict[str, Any],
        *,
        config: RegistryUpdateConfig,
    ) -> None:
        nct_id = str(normalized["nct_id"])
        _write_json(config.raw_dir / f"{nct_id}.json", study)
        _write_json(config.normalized_trials_dir / f"{nct_id}.json", normalized)

    @staticmethod
    def write_run_report(
        reports_dir: str | Path,
        report: RegistryUpdateReport,
    ) -> Path:
        reports_path = Path(reports_dir)
        reports_path.mkdir(parents=True, exist_ok=True)
        path = reports_path / f"registry-update-{utc_now_iso().replace(':', '')}.json"
        _write_json(path, report.to_dict())
        return path


def _remaining(max_studies: int | None, fetched: int) -> int | None:
    if max_studies is None:
        return None
    return max(0, max_studies - fetched)


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def normalize_keywords(keywords: Sequence[str]) -> tuple[str, ...]:
    values = tuple(dict.fromkeys(keyword.strip() for keyword in keywords if keyword.strip()))
    return values or DEFAULT_REGISTRY_KEYWORDS
