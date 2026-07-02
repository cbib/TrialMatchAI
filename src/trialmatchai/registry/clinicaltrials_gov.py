from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from trialmatchai.utils.logging_config import setup_logging

logger = setup_logging(__name__)


class RegistrySourceError(RuntimeError):
    pass


class TransientRegistrySourceError(RegistrySourceError):
    pass


@dataclass
class ClinicalTrialsGovClient:
    base_url: str = "https://clinicaltrials.gov/api/v2/studies"
    timeout: float = 30.0
    rate_limit_per_second: float = 2.0
    session: requests.Session = field(default_factory=requests.Session)
    page_size: int = 100

    def __post_init__(self) -> None:
        if self.page_size < 1 or self.page_size > 1000:
            raise ValueError("page_size must be between 1 and 1000.")
        if self.rate_limit_per_second <= 0:
            raise ValueError("rate_limit_per_second must be positive.")
        self._last_request_at = 0.0

    def iter_studies(
        self,
        *,
        keyword: str,
        statuses: Sequence[str] = (),
        since: date | None = None,
        max_studies: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield studies for one ClinicalTrials.gov keyword query.

        Date filtering is post-fetch: upstream filter names have changed before,
        so keeping it local is more robust and testable.
        """
        yielded = 0
        page_token: str | None = None
        status_filter = tuple(status for status in statuses if status)
        while True:
            payload = self._fetch_page(
                keyword=keyword,
                statuses=status_filter,
                page_token=page_token,
            )
            studies = payload.get("studies") or []
            if not isinstance(studies, list):
                raise RegistrySourceError("ClinicalTrials.gov returned non-list studies.")
            for study in studies:
                if not isinstance(study, dict):
                    continue
                if since is not None and not _study_updated_since(study, since):
                    continue
                yield study
                yielded += 1
                if max_studies is not None and yielded >= max_studies:
                    return
            page_token = payload.get("nextPageToken")
            if not page_token:
                return

    def _fetch_page(
        self,
        *,
        keyword: str,
        statuses: Sequence[str],
        page_token: str | None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "format": "json",
            "pageSize": self.page_size,
            "query.term": keyword,
        }
        if statuses:
            params["filter.overallStatus"] = ",".join(statuses)
        if page_token:
            params["pageToken"] = page_token
        return self._get_json(params)

    @retry(
        retry=retry_if_exception_type(
            (requests.Timeout, requests.ConnectionError, TransientRegistrySourceError)
        ),
        wait=wait_exponential_jitter(initial=0.5, max=8),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def _get_json(self, params: dict[str, Any]) -> dict[str, Any]:
        self._throttle()
        try:
            response = self.session.get(
                self.base_url,
                params=params,
                timeout=self.timeout,
                headers={"Accept": "application/json"},
            )
        except (requests.Timeout, requests.ConnectionError):
            raise
        except requests.RequestException as exc:
            raise RegistrySourceError(f"ClinicalTrials.gov request failed: {exc}") from exc

        if response.status_code in {408, 425, 429, 500, 502, 503, 504}:
            raise TransientRegistrySourceError(
                f"ClinicalTrials.gov transient HTTP {response.status_code}"
            )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise RegistrySourceError(
                f"ClinicalTrials.gov HTTP {response.status_code}: {response.text[:300]}"
            ) from exc
        payload = response.json()
        if not isinstance(payload, dict):
            raise RegistrySourceError("ClinicalTrials.gov returned non-object JSON.")
        return payload

    def _throttle(self) -> None:
        minimum_gap = 1.0 / self.rate_limit_per_second
        now = time.monotonic()
        elapsed = now - self._last_request_at
        if elapsed < minimum_gap:
            time.sleep(minimum_gap - elapsed)
        self._last_request_at = time.monotonic()


def _study_updated_since(study: dict[str, Any], since: date) -> bool:
    updated = _last_update_date(study)
    return updated is None or updated >= since


def _last_update_date(study: dict[str, Any]) -> date | None:
    protocol = study.get("protocolSection") if isinstance(study, dict) else None
    if not isinstance(protocol, dict):
        return None
    status = protocol.get("statusModule")
    if not isinstance(status, dict):
        return None
    for key in (
        "lastUpdatePostDateStruct",
        "lastUpdateSubmitDateStruct",
        "studyFirstPostDateStruct",
    ):
        value = status.get(key)
        if isinstance(value, dict):
            parsed = _parse_date(value.get("date"))
            if parsed is not None:
                return parsed
    return None


def _parse_date(value: Any) -> date | None:
    if not value:
        return None
    text = str(value)
    try:
        if len(text) == 4:
            return date(int(text), 1, 1)
        if len(text) == 7:
            year, month = text.split("-", 1)
            return date(int(year), int(month), 1)
        return date.fromisoformat(text[:10])
    except ValueError:
        logger.warning("Could not parse ClinicalTrials.gov date: %s", text)
        return None
