from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from trialmatchai.registry.clinicaltrials_gov import ClinicalTrialsGovClient
from trialmatchai.registry.defaults import DEFAULT_REGISTRY_KEYWORDS
from trialmatchai.registry.manifest import RegistryManifest, source_hash
from trialmatchai.registry.normalization import normalize_study, split_eligibility_criteria
from trialmatchai.registry.updater import RegistryUpdateConfig, RegistryUpdater, normalize_keywords
from trialmatchai.search import InMemorySearchBackend


def test_client_paginates_and_filters_since():
    session = _FakeSession(
        [
            {
                "studies": [
                    _study("NCT00000001", last_update="2026-06-01"),
                    _study("NCT00000002", last_update="2026-01-01"),
                ],
                "nextPageToken": "page-2",
            },
            {"studies": [_study("NCT00000003", last_update="2026-06-02")]},
        ]
    )
    client = ClinicalTrialsGovClient(
        session=session,
        rate_limit_per_second=1000,
        timeout=1,
    )

    studies = list(
        client.iter_studies(
            keyword="cancer",
            statuses=("RECRUITING",),
            since=date(2026, 6, 1),
        )
    )

    assert [study["protocolSection"]["identificationModule"]["nctId"] for study in studies] == [
        "NCT00000001",
        "NCT00000003",
    ]
    assert session.calls[0]["params"]["query.term"] == "cancer"
    assert session.calls[0]["params"]["filter.overallStatus"] == "RECRUITING"
    assert session.calls[1]["params"]["pageToken"] == "page-2"


def test_prepare_trial_document_preserves_detailed_description_and_official_title():
    # These two fields carry BM25 weight in the backend (TRIAL_TEXT_WEIGHTS) but
    # were previously dropped before indexing, silently losing retrieval signal.
    from trialmatchai.registry.preparation import prepare_trial_document

    class _StubEmbedder:
        def embed_texts(self, texts):
            return [[0.0, 0.0] for _ in texts]

    prepared = prepare_trial_document(
        {
            "nct_id": "NCT00000009",
            "brief_title": "Brief",
            "condition": ["Lung cancer"],
            "eligibility_criteria": "Adults",
            "detailed_description": "A longer detailed description of the study.",
            "official_title": "The Official Long Title of the Study",
        },
        _StubEmbedder(),
    )

    assert "description" in prepared["detailed_description"].lower()
    assert "official" in prepared["official_title"].lower()


def test_normalize_study_maps_v2_modules_and_splits_criteria():
    normalized = normalize_study(_study("NCT00000004"))

    assert normalized["nct_id"] == "NCT00000004"
    assert normalized["brief_title"] == "Trial for NCT00000004"
    assert normalized["condition"] == ["Lung cancer"]
    assert normalized["source_url"] == "https://clinicaltrials.gov/study/NCT00000004"
    assert normalized["criteria"] == [
        {"type": "inclusion", "criterion": "Age 18 years or older"},
        {"type": "inclusion", "criterion": "Histologically confirmed lung cancer"},
        {"type": "exclusion", "criterion": "Prior investigational therapy"},
    ]


def test_split_eligibility_criteria_has_unknown_fallback():
    assert split_eligibility_criteria("Able to consent.") == [
        {"type": "unknown", "criterion": "Able to consent."}
    ]


def test_manifest_keeps_latest_record_by_nct_id(tmp_path):
    manifest = RegistryManifest(tmp_path / "manifest.jsonl")
    study = _study("NCT00000005")
    record = _manifest_record("NCT00000005", source_hash(study), status="indexed")
    changed = _manifest_record("NCT00000005", "changed", status="indexed")

    manifest.append(record)
    manifest.append(changed)

    latest = manifest.load_latest()
    assert latest["NCT00000005"].source_hash == "changed"


def test_updater_writes_new_trial_and_upserts_tables(tmp_path):
    study = _study("NCT00000006")
    backend = InMemorySearchBackend(
        criteria=[
            {
                "criteria_id": "old",
                "nct_id": "NCT00000006",
                "criterion": "old criterion",
                "criterion_vector": [0.0, 0.0],
            }
        ]
    )
    updater = RegistryUpdater(
        client=_FakeRegistryClient([study]),
        backend=backend,
        embedder=_FakeEmbedder(),
    )

    report = updater.run(_update_config(tmp_path))

    assert report.fetched == 1
    assert report.new == 1
    assert report.indexed == 1
    assert report.criteria_indexed == 3
    assert (tmp_path / "raw/NCT00000006.json").exists()
    assert (tmp_path / "trials/NCT00000006.json").exists()
    assert len(backend.trials) == 1
    assert all(row["nct_id"] == "NCT00000006" for row in backend.criteria)
    assert {row["criterion"] for row in backend.criteria} == {
        "Age 18 years or older",
        "Histologically confirmed lung cancer",
        "Prior investigational therapy",
    }


def test_updater_skips_unchanged_hashes(tmp_path):
    study = _study("NCT00000007")
    config = _update_config(tmp_path)
    RegistryManifest(config.manifest_path).append(
        _manifest_record("NCT00000007", source_hash(study), status="indexed")
    )
    backend = InMemorySearchBackend()
    updater = RegistryUpdater(
        client=_FakeRegistryClient([study]),
        backend=backend,
        embedder=_FakeEmbedder(),
    )

    report = updater.run(config)

    assert report.unchanged == 1
    assert report.indexed == 0
    assert backend.trials == []


def test_updater_dry_run_does_not_write_files_or_tables(tmp_path):
    backend = InMemorySearchBackend()
    updater = RegistryUpdater(
        client=_FakeRegistryClient([_study("NCT00000008")]),
        backend=backend,
        embedder=_FakeEmbedder(),
    )

    report = updater.run(_update_config(tmp_path, dry_run=True))

    assert report.new == 1
    assert not (tmp_path / "raw").exists()
    assert not (tmp_path / "trials").exists()
    assert not (tmp_path / "manifest.jsonl").exists()
    assert backend.trials == []


def test_normalize_keywords_uses_defaults_when_empty():
    assert normalize_keywords([]) == DEFAULT_REGISTRY_KEYWORDS
    assert normalize_keywords([" cancer ", "cancer", "diabetes"]) == (
        "cancer",
        "diabetes",
    )


class _FakeResponse:
    def __init__(self, payload: dict[str, Any], status_code: int = 200):
        self.payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def json(self):
        return self.payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.text)


class _FakeSession:
    def __init__(self, payloads: list[dict[str, Any]]):
        self.payloads = payloads
        self.calls: list[dict[str, Any]] = []

    def get(self, url, *, params, timeout, headers):
        self.calls.append(
            {
                "url": url,
                "params": dict(params),
                "timeout": timeout,
                "headers": dict(headers),
            }
        )
        return _FakeResponse(self.payloads.pop(0))


class _FakeRegistryClient:
    def __init__(self, studies: list[dict[str, Any]]):
        self.studies = studies

    def iter_studies(self, *, keyword, statuses, since, max_studies):
        del keyword, statuses, since
        for study in self.studies[:max_studies]:
            yield study


class _FakeEmbedder:
    def embed_texts(self, texts):
        return [[float(len(text)), 1.0] for text in texts]


def _update_config(tmp_path: Path, *, dry_run: bool = False) -> RegistryUpdateConfig:
    return RegistryUpdateConfig(
        raw_dir=tmp_path / "raw",
        normalized_trials_dir=tmp_path / "trials",
        manifest_path=tmp_path / "manifest.jsonl",
        reports_dir=tmp_path / "runs",
        keywords=("cancer",),
        statuses=("RECRUITING",),
        since=None,
        dry_run=dry_run,
    )


def _manifest_record(nct_id: str, digest: str, *, status: str):
    from trialmatchai.registry.manifest import ManifestRecord, utc_now_iso

    return ManifestRecord(
        nct_id=nct_id,
        source_url=f"https://clinicaltrials.gov/study/{nct_id}",
        source_hash=digest,
        fetched_at=utc_now_iso(),
        last_update_posted="2026-06-01",
        processing_status=status,
    )


def _study(nct_id: str, *, last_update: str = "2026-06-01") -> dict[str, Any]:
    return {
        "protocolSection": {
            "identificationModule": {
                "nctId": nct_id,
                "briefTitle": f"Trial for {nct_id}",
                "officialTitle": f"Official trial for {nct_id}",
            },
            "statusModule": {
                "overallStatus": "RECRUITING",
                "startDateStruct": {"date": "2026-01-01"},
                "completionDateStruct": {"date": "2027-01-01"},
                "lastUpdatePostDateStruct": {"date": last_update},
            },
            "descriptionModule": {
                "briefSummary": "A test trial.",
                "detailedDescription": "A detailed test trial.",
            },
            "conditionsModule": {"conditions": ["Lung cancer"]},
            "designModule": {
                "phases": ["PHASE2"],
                "studyType": "INTERVENTIONAL",
            },
            "eligibilityModule": {
                "sex": "ALL",
                "minimumAge": "18 Years",
                "maximumAge": "80 Years",
                "eligibilityCriteria": "\n".join(
                    [
                        "Inclusion Criteria:",
                        "- Age 18 years or older",
                        "- Histologically confirmed lung cancer",
                        "Exclusion Criteria:",
                        "- Prior investigational therapy",
                    ]
                ),
            },
            "armsInterventionsModule": {
                "interventions": [{"name": "Drug A", "type": "DRUG"}]
            },
            "contactsLocationsModule": {
                "locations": [{"facility": "Site A", "country": "United States"}]
            },
            "referencesModule": {
                "references": [{"pmid": "123", "citation": "Reference"}]
            },
        }
    }
