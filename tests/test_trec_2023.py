"""TREC 2023 track: registration, questionnaire-topic parsing, corpus backfill."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trialmatchai.registry.clinicaltrials_gov import ClinicalTrialsGovClient
from trialmatchai.trec.backfill import backfill_track_corpus, track_corpus_state
from trialmatchai.trec.corpus import TRACK_KEYS, resolve_tracks
from trialmatchai.trec.qrels import QRELS_URLS
from trialmatchai.trec.topics import (
    TOPIC_SOURCES,
    extract_demographics,
    parse_questionnaire_topics,
)

QUESTIONNAIRE_XML = """<topics task="2023 TREC Clinical Trials">
  <topic number="1" template="glaucoma">
    <field name="definitive diagnosis">primary open angle glaucoma</field>
    <field name="intraocular pressure">28 mmHg</field>
    <field name="visual field"></field>
    <field name="prior cataract surgery">no</field>
  </topic>
  <topic number="6" template="anxiety">
    <field name="definitive diagnosis">yes</field>
    <field name="age">39</field>
    <field name="HAM-A">20</field>
  </topic>
</topics>
"""


def test_track_23_registered_everywhere():
    assert "23" in TRACK_KEYS
    assert QRELS_URLS["23"].endswith("qrels2023.txt")
    source = TOPIC_SOURCES["23"]
    assert source.kind == "cds_questionnaire_xml"
    assert source.topics_url.endswith("topics2023.xml")
    spec = resolve_tracks(["23"], data_dir=Path("data"), results_root=Path("."))[0]
    assert spec.id_prefix == "trec-2023"
    assert spec.name == "trec23"


def test_parse_questionnaire_topics_flattens_fields(tmp_path):
    path = tmp_path / "topics_23.xml"
    path.write_text(QUESTIONNAIRE_XML, encoding="utf-8")
    topics = parse_questionnaire_topics(path, "trec-2023")
    assert set(topics) == {"trec-20231", "trec-20236"}

    glaucoma = topics["trec-20231"]
    assert glaucoma.startswith("Patient screening questionnaire for glaucoma.")
    assert "definitive diagnosis: primary open angle glaucoma." in glaucoma
    assert "intraocular pressure: 28 mmHg." in glaucoma
    # Negative answers carry eligibility signal and must survive flattening.
    assert "prior cataract surgery: no." in glaucoma
    # Blank fields mean "not provided" and are dropped entirely.
    assert "visual field" not in glaucoma


def test_parse_questionnaire_topics_empty_file_raises(tmp_path):
    path = tmp_path / "topics_23.xml"
    path.write_text("<topics></topics>", encoding="utf-8")
    with pytest.raises(ValueError):
        parse_questionnaire_topics(path, "trec-2023")


def test_extract_demographics_reads_flattened_age_field(tmp_path):
    path = tmp_path / "topics_23.xml"
    path.write_text(QUESTIONNAIRE_XML, encoding="utf-8")
    anxiety = parse_questionnaire_topics(path, "trec-2023")["trec-20236"]
    age, sex = extract_demographics(anxiety)
    assert age == 39.0
    assert sex is None  # questionnaires carry no sex field; no false positives


class _RecordingClient(ClinicalTrialsGovClient):
    """Serves canned pages and records request params; no network, no throttle."""

    def __init__(self, pages):
        super().__init__(rate_limit_per_second=1000.0)
        self._pages = list(pages)
        self.requests: list[dict] = []

    def _get_json(self, params):
        self.requests.append(dict(params))
        return self._pages.pop(0) if self._pages else {"studies": []}


def _study(nct_id: str) -> dict:
    return {
        "protocolSection": {
            "identificationModule": {"nctId": nct_id, "briefTitle": f"Trial {nct_id}"},
            "eligibilityModule": {
                "sex": "ALL",
                "eligibilityCriteria": "Inclusion Criteria:\n- adults",
            },
        }
    }


def test_iter_studies_by_ids_chunks_and_paginates():
    client = _RecordingClient(
        [
            {"studies": [_study("NCT01")], "nextPageToken": "t"},
            {"studies": [_study("NCT02")]},
            {"studies": [_study("NCT03")]},
        ]
    )
    got = [
        s["protocolSection"]["identificationModule"]["nctId"]
        for s in client.iter_studies_by_ids(["NCT01", "NCT02", "NCT03"], chunk_size=2)
    ]
    assert got == ["NCT01", "NCT02", "NCT03"]
    # Chunk 1 (two ids) pages twice via nextPageToken; chunk 2 (one id) once.
    assert client.requests[0]["filter.ids"] == "NCT01,NCT02"
    assert "pageToken" not in client.requests[0]
    assert client.requests[1]["pageToken"] == "t"
    assert client.requests[2]["filter.ids"] == "NCT03"


def test_iter_studies_by_ids_drops_studies_nobody_asked_for():
    """For unknown ids the live API returns arbitrary unrelated studies rather than an
    empty set, so the client must filter the response back to the requested ids."""
    client = _RecordingClient([{"studies": [_study("NCT99"), _study("NCT01")]}])
    got = [
        s["protocolSection"]["identificationModule"]["nctId"]
        for s in client.iter_studies_by_ids(["NCT01", "NCT02"])
    ]
    assert got == ["NCT01"]


def _write_qrels(data_dir: Path, lines: list[str]) -> None:
    qrels_dir = data_dir / "trec" / "qrels"
    qrels_dir.mkdir(parents=True)
    (qrels_dir / "qrels_23.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_backfill_writes_missing_and_reports_unavailable(tmp_path):
    _write_qrels(tmp_path, ["1 0 NCT01 2", "1 0 NCT02 0", "2 0 NCT03 1"])
    trials = tmp_path / "trials_jsons"
    trials.mkdir()
    (trials / "NCT01.json").write_text("{}", encoding="utf-8")

    pool, missing = track_corpus_state(
        "23", data_dir=tmp_path, trials_json_folder=trials
    )
    assert pool == {"NCT01", "NCT02", "NCT03"}
    assert missing == ["NCT02", "NCT03"]

    # NCT03 is judged but gone from the live registry. The API answers that unknown id
    # with an unrelated study (NCT77), which must not be written; NCT03 -> "unavailable".
    client = _RecordingClient([{"studies": [_study("NCT02"), _study("NCT77")]}])
    stats = backfill_track_corpus(
        "23", data_dir=tmp_path, trials_json_folder=trials, client=client
    )
    assert stats == {
        "pool": 3,
        "present": 1,
        "written": 1,
        "failed": 0,
        "unavailable": 1,
    }
    doc = json.loads((trials / "NCT02.json").read_text(encoding="utf-8"))
    assert doc["nct_id"] == "NCT02"
    assert doc["brief_title"] == "Trial NCT02"
    assert doc["criteria"]  # eligibility text was split into criteria rows
    assert not (trials / "NCT77.json").exists()  # unrequested study never written


def test_backfill_noop_when_corpus_complete(tmp_path):
    _write_qrels(tmp_path, ["1 0 NCT01 2"])
    trials = tmp_path / "trials_jsons"
    trials.mkdir()
    (trials / "NCT01.json").write_text("{}", encoding="utf-8")
    client = _RecordingClient([])
    stats = backfill_track_corpus(
        "23", data_dir=tmp_path, trials_json_folder=trials, client=client
    )
    assert stats["written"] == 0 and stats["unavailable"] == 0
    assert client.requests == []
