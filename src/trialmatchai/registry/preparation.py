from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import dateutil.parser

from trialmatchai.constraints import extract_constraint_set
from trialmatchai.registry.criteria_chunking import split_eligibility_criteria
from trialmatchai.utils.text import flatten_text


class TextEmbeddingBackend(Protocol):
    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        ...


class EntityAnnotationBackend(Protocol):
    def annotate_texts_in_parallel(
        self,
        texts: Sequence[str],
        max_workers: int = 1,
        retries: int = 1,
        delay: float = 0,
    ) -> list[list[dict[str, Any]]]:
        ...


TRIAL_TEXT_FIELDS: tuple[tuple[str, str], ...] = (
    ("brief_title", "brief_title_vector"),
    ("brief_summary", "brief_summary_vector"),
    ("condition", "condition_vector"),
    ("eligibility_criteria", "eligibility_criteria_vector"),
)


def prepare_trial_document(
    doc: dict[str, Any],
    embedder: TextEmbeddingBackend,
) -> dict[str, Any]:
    out: dict[str, Any] = {"nct_id": doc["nct_id"]}
    texts = [_preprocess_text(flatten_text(doc.get(field))) for field, _ in TRIAL_TEXT_FIELDS]
    vectors = _embed_texts(embedder, texts)

    for (field, vector_field), text, vector in zip(TRIAL_TEXT_FIELDS, texts, vectors):
        out[field] = text
        out[vector_field] = vector

    # Text-only fields the backend scores on (TRIAL_TEXT_WEIGHTS) but does not embed.
    for text_field in ("detailed_description", "official_title"):
        value = _preprocess_text(flatten_text(doc.get(text_field)))
        out[text_field] = value

    for simple in (
        "overall_status",
        "phase",
        "study_type",
        "gender",
        "source",
        "source_url",
        "last_update_posted",
    ):
        out[simple] = str(doc.get(simple) or "")

    out["start_date"] = _to_iso_date(doc.get("start_date")) or ""
    out["completion_date"] = _to_iso_date(doc.get("completion_date")) or ""

    out["minimum_age"] = _age_to_years(doc.get("minimum_age"))
    if out["minimum_age"] is None:
        out["minimum_age"] = 0.0
    out["maximum_age"] = _age_to_years(doc.get("maximum_age"))
    if out["maximum_age"] is None:
        out["maximum_age"] = 999.0

    out["intervention"] = _stable_dict_list(
        doc.get("intervention"),
        keys=("name", "type", "description"),
    )
    out["location"] = _stable_dict_list(
        doc.get("location"),
        keys=("facility", "city", "state", "country", "status"),
    )
    out["reference"] = _stable_dict_list(
        doc.get("reference"),
        keys=("pmid", "citation", "type"),
    )

    return out


def prepare_criteria_documents(
    doc: dict[str, Any],
    embedder: TextEmbeddingBackend,
    *,
    entity_annotator: EntityAnnotationBackend | None = None,
) -> list[dict[str, Any]]:
    nct_id = str(doc["nct_id"])
    entries: list[dict[str, Any]] = []
    texts: list[str] = []
    criteria = doc.get("criteria")
    if not criteria:
        # No pre-chunked criteria: chunk the raw text here, unflattened to keep line structure.
        raw = doc.get("eligibility_criteria")
        criteria = split_eligibility_criteria(raw) if isinstance(raw, str) else []
    for criterion in criteria or []:
        if not isinstance(criterion, dict):
            continue
        text = _preprocess_text(
            flatten_text(criterion.get("criterion") or criterion.get("sentence"))
        )
        if not text:
            continue
        entries.append(
            {
                "nct_id": nct_id,
                "criterion": text,
                "entities": criterion.get("entities") or [],
                "eligibility_type": criterion.get("type") or "unknown",
            }
        )
        texts.append(text)

    if not entries:
        return []

    _annotate_missing_entities(entries, texts, entity_annotator)
    vectors = _embed_texts(embedder, texts)
    rows: list[dict[str, Any]] = []
    for entry, vector in zip(entries, vectors):
        criteria_id = compute_criteria_id(entry["nct_id"], entry["criterion"])
        constraint_set = extract_constraint_set(
            nct_id=entry["nct_id"],
            criteria_id=criteria_id,
            criterion=entry["criterion"],
            eligibility_type=entry["eligibility_type"],
            entities=entry.get("entities"),
        )
        rows.append(
            {
                "criteria_id": criteria_id,
                "nct_id": entry["nct_id"],
                "criterion": entry["criterion"],
                "entities": _entities_for_index(entry.get("entities")),
                "eligibility_type": entry["eligibility_type"],
                "criterion_vector": vector,
                "constraints": json.dumps(
                    constraint_set.model_dump(mode="json"),
                    sort_keys=True,
                ),
            }
        )
    return rows


def _atomic_write_text(path: Path, text: str) -> None:
    """Write text via a temp file + os.replace so readers never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=path.suffix)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write_prepared_trial(row: dict[str, Any], folder: str | Path) -> Path:
    path = Path(folder) / f"{row['nct_id']}.json"
    _atomic_write_text(path, json.dumps(row, indent=2, sort_keys=True))
    return path


def write_prepared_criteria(rows: Sequence[dict[str, Any]], folder: str | Path) -> int:
    if not rows:
        return 0
    trial_folder = Path(folder) / str(rows[0]["nct_id"])
    # Clear stale criteria first: criteria_id = sha256(nct:text), so changed text
    # orphans old files the index would otherwise ingest as duplicates.
    if trial_folder.exists():
        shutil.rmtree(trial_folder, ignore_errors=True)
    for row in rows:
        _atomic_write_text(
            trial_folder / f"{row['criteria_id']}.json",
            json.dumps(row, indent=2, sort_keys=True),
        )
    return len(rows)


def compute_criteria_id(nct_id: str, criterion: str) -> str:
    return hashlib.sha256(f"{nct_id}:{criterion}".encode("utf-8")).hexdigest()


def _embed_texts(embedder: TextEmbeddingBackend, texts: Sequence[str]) -> list[list[float]]:
    vectors_by_nonempty_text = iter(
        embedder.embed_texts([text for text in texts if text.strip()])
    )
    vectors: list[list[float]] = []
    for text in texts:
        if text.strip():
            vectors.append(list(next(vectors_by_nonempty_text)))
        else:
            vectors.append([])
    return vectors


def _annotate_missing_entities(
    entries: list[dict[str, Any]],
    texts: list[str],
    entity_annotator: EntityAnnotationBackend | None,
) -> None:
    if entity_annotator is None:
        return
    missing_indices = [
        index for index, entry in enumerate(entries) if not entry.get("entities")
    ]
    if not missing_indices:
        return
    missing_texts = [texts[index] for index in missing_indices]
    annotations = entity_annotator.annotate_texts_in_parallel(
        missing_texts,
        max_workers=1,
    )
    for index, entities in zip(missing_indices, annotations):
        entries[index]["entities"] = entities


def _entities_for_index(entities: Any) -> list[dict[str, Any]]:
    if not isinstance(entities, list):
        return []
    indexed: list[dict[str, Any]] = []
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        normalized = dict(entity)
        normalized.setdefault("entity", normalized.get("text", ""))
        normalized.setdefault("class", normalized.get("entity_group", ""))
        normalized.setdefault("normalized_id", ["CUI-less"])
        normalized.setdefault("synonyms", [])
        normalized.setdefault("concept_candidates", [])
        normalized.setdefault("linker_status", "not_linked")
        indexed.append(normalized)
    return indexed


def _preprocess_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _stable_dict_list(value: Any, *, keys: Sequence[str]) -> list[dict[str, str]]:
    rows = value if isinstance(value, list) else []
    normalized: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        normalized.append({key: str(row.get(key) or "") for key in keys})
    if normalized:
        return normalized
    return [{key: "" for key in keys}]


def _to_iso_date(value: Any) -> str | None:
    if not value:
        return None
    try:
        # Fixed default so partial dates ("2021-03", "2021") don't inherit today's day/month.
        return (
            dateutil.parser.parse(str(value), default=datetime(1900, 1, 1))
            .date()
            .isoformat()
        )
    except (TypeError, ValueError, dateutil.parser.ParserError):
        return None


def _age_to_years(value: Any) -> float | None:
    if not value:
        return None
    match = re.search(r"([\d.]+)", str(value))
    if not match:
        return None
    amount = float(match.group(1))
    unit = str(value).casefold()
    if "year" in unit:
        years = amount
    elif "month" in unit:
        years = amount / 12
    elif "week" in unit:
        years = amount / 52
    elif "day" in unit:
        years = amount / 365
    else:
        return None
    return round(years, 2)
