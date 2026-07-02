"""Official TREC topic acquisition and import.

Downloads the authoritative patient topics per track (NIST for 2021/2022, the
CSIRO/SIGIR-2016 collection for sigir) and builds canonical
:class:`PatientProfile` objects from the RAW topic text only — no LLM-preprocessed
fields — with demographics extracted deterministically from the narrative. The
runtime CoT query-expansion later turns this raw text into the ``keywords.json``
consumed by first-level retrieval, reproducing the legacy pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from trialmatchai.interop.models import Demographics, PatientNote, PatientProfile, Provenance
from trialmatchai.utils.logging_config import setup_logging

logger = setup_logging(__name__)

SOURCE_FORMAT = "trec_topic"


@dataclass(frozen=True)
class TopicSource:
    """Where a track's official topics live and how to label them.

    kind "nist_xml": download and parse topic XML from ``topics_url``.
    kind "local_raw": read verbatim raw patient text from on-disk ``raw_json``.
    """

    track: str
    id_prefix: str
    kind: str
    topics_url: str | None = None      # nist_xml
    raw_json: str | None = None        # local_raw (filename under <data>/trec/)
    collection_url: str | None = None
    note: str = ""


TOPIC_SOURCES: dict[str, TopicSource] = {
    "21": TopicSource(
        track="21",
        id_prefix="trec-2021",
        kind="nist_xml",
        topics_url="https://trec.nist.gov/data/trials/topics2021.xml",
    ),
    "22": TopicSource(
        track="22",
        id_prefix="trec-2022",
        kind="nist_xml",
        topics_url="https://trec.nist.gov/data/trials/topics2022.xml",
    ),
    # The SIGIR-2016 collection has no stable direct-download URL (CSIRO portal),
    # so sigir topics come from the verbatim on-disk raw admission statements.
    "sigir": TopicSource(
        track="sigir",
        id_prefix="sigir-2014",
        kind="local_raw",
        raw_json="processed_sigir_patients.json",
        collection_url="https://data.csiro.au/collection/csiro:17152",
        note="sigir raw text is read verbatim from data/trec/processed_sigir_patients.json "
        "(raw_description only; no gpt-generated fields).",
    ),
}


@retry(
    retry=retry_if_exception_type(requests.RequestException),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    reraise=True,
)
def _http_get(url: str, timeout: float = 60.0) -> bytes:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.content


def download_topics(track: str, dest_dir: Path) -> Path:
    """Fetch (and cache) the official topic XML for a track.

    Portal-only sources (sigir) must be placed manually; the error says where.
    """
    source = TOPIC_SOURCES[track]
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"topics_{track}.xml"

    if dest.exists() and dest.stat().st_size > 0:
        logger.info("Topics for track %s already present: %s", track, dest)
        return dest

    if not source.topics_url:
        raise FileNotFoundError(
            f"No direct download for track '{track}'. {source.note} "
            f"Source: {source.collection_url}"
        )

    logger.info("Downloading official topics for track %s from %s", track, source.topics_url)
    tmp = dest.with_name(dest.name + ".part")
    tmp.write_bytes(_http_get(source.topics_url))
    tmp.replace(dest)  # atomic: a crash mid-download must not leave a truncated cache
    return dest


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def parse_topics(path: Path, id_prefix: str) -> dict[str, str]:
    """Parse a TREC/TREC-CDS topic XML into {patient_id: raw_text}.

    Handles both the flat ``<topic number="N">text</topic>`` form (2021/2022) and
    the TREC-CDS form wrapping ``<summary>``/``<description>`` children (sigir),
    preferring summary, then description, then the element's own text.
    """
    root = ET.parse(path).getroot()
    topics: dict[str, str] = {}
    for topic in root.iter("topic"):
        number = topic.get("number") or topic.get("id")
        if number is None:
            continue
        text = _topic_text(topic)
        if not text:
            continue
        topics[f"{id_prefix}{number.strip()}"] = text
    if not topics:
        raise ValueError(f"No <topic> elements parsed from {path}")
    return topics


def _topic_text(topic: ET.Element) -> str:
    for child_tag in ("summary", "description"):
        child = topic.find(child_tag)
        if child is not None and (child.text or "").strip():
            return _clean(child.text)
    return _clean("".join(topic.itertext()))


def _clean(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


# --------------------------------------------------------------------------- #
# Deterministic demographics (no LLM, no gpt-preprocessed fields)
# --------------------------------------------------------------------------- #
_AGE_PATTERNS = [
    re.compile(r"(\d{1,3})\s*[- ]?\s*year[\s-]*old", re.I),
    re.compile(r"(\d{1,3})\s*[- ]?\s*(?:yo|y/o|yr)\b", re.I),
    re.compile(r"\b(\d{1,3})\s*[- ]?\s*(?:M|F|male|female|man|woman)\b"),
]
_FEMALE = re.compile(r"\b(?:female|woman|girl|lady|\d{1,3}\s*F)\b", re.I)
_MALE = re.compile(r"\b(?:male|man|boy|gentleman|\d{1,3}\s*M)\b", re.I)


def extract_demographics(text: str) -> tuple[float | None, str | None]:
    """Best-effort (age_years, sex) from the topic narrative; None if unknown."""
    age: float | None = None
    for pattern in _AGE_PATTERNS:
        m = pattern.search(text)
        if m:
            value = int(m.group(1))
            if 0 < value <= 120:
                age = float(value)
                break
    # First mention wins: the subject precedes any relatives, avoiding
    # "a man with a female partner" -> Female.
    sex: str | None = None
    fm = _FEMALE.search(text)
    mm = _MALE.search(text)
    if fm and mm:
        sex = "Female" if fm.start() < mm.start() else "Male"
    elif fm:
        sex = "Female"
    elif mm:
        sex = "Male"
    return age, sex


def build_profile_from_topic(patient_id: str, raw_text: str) -> PatientProfile:
    """Build a canonical profile from raw topic text only (no derived terms)."""
    provenance = Provenance(source_format=SOURCE_FORMAT, source_id=patient_id)
    age, sex = extract_demographics(raw_text)
    return PatientProfile(
        patient_id=patient_id,
        demographics=Demographics(sex=sex, age_years=age),
        notes=[
            PatientNote(
                note_id=f"{patient_id}-note",
                text=raw_text,
                note_type="trec-topic",
                provenance=provenance,
            )
        ],
        provenance=[provenance],
    )


def _load_local_raw(path: Path) -> dict[str, str]:
    """Read {patient_id: raw text} from an on-disk topic JSON.

    Uses only ``raw_description`` (the official admission statement); every
    gpt-generated field is ignored.
    """
    import json

    records = json.loads(Path(path).read_text(encoding="utf-8"))
    topics = {
        pid: _clean(rec.get("raw_description"))
        for pid, rec in records.items()
        if _clean(rec.get("raw_description"))
    }
    if not topics:
        raise ValueError(f"No raw_description text found in {path}")
    return topics


def load_track_topics(track: str, trec_dir: Path) -> dict[str, str]:
    """Resolve a track's topics to {patient_id: raw_text} per its source kind."""
    source = TOPIC_SOURCES[track]
    if source.kind == "nist_xml":
        path = download_topics(track, Path(trec_dir) / "raw_topics")
        return parse_topics(path, source.id_prefix)
    if source.kind == "local_raw":
        return _load_local_raw(Path(trec_dir) / source.raw_json)
    raise ValueError(f"Unknown topic source kind: {source.kind}")


def import_topics(
    track: str,
    *,
    trec_dir: Path,
    profile_dir: Path,
    summary_dir: Path,
    force: bool = False,
) -> int:
    """Acquire a track's official topics and write canonical inputs.

    Writes one PatientProfile per topic (raw text + demographics) plus a matching
    summary. Idempotent; returns the patient count.
    """
    import json

    from trialmatchai.interop.exporters import profile_to_matching_summary

    profile_dir = Path(profile_dir)
    summary_dir = Path(summary_dir)

    topics = load_track_topics(track, Path(trec_dir))

    # Skip only when every topic has both files: a partial import leaves some
    # missing, and skipping on "any exists" would silently drop the rest.
    if not force and all(
        (profile_dir / f"{patient_id}.json").exists()
        and (summary_dir / f"{patient_id}.json").exists()
        for patient_id in topics
    ):
        logger.info("Topic import skipped for track %s: all %s profiles present", track, len(topics))
        return len(topics)

    profile_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    for patient_id, raw_text in topics.items():
        profile = build_profile_from_topic(patient_id, raw_text)
        (profile_dir / f"{patient_id}.json").write_text(
            profile.model_dump_json(indent=2, exclude_none=True), encoding="utf-8"
        )
        (summary_dir / f"{patient_id}.json").write_text(
            json.dumps(profile_to_matching_summary(profile), indent=2), encoding="utf-8"
        )
    logger.info("Imported %s official topics for track %s", len(topics), track)
    return len(topics)
