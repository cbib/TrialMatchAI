from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import dateutil.parser


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
    texts = [_preprocess_text(_flatten_text(doc.get(field))) for field, _ in TRIAL_TEXT_FIELDS]
    vectors = _embed_texts(embedder, texts)

    for (field, vector_field), text, vector in zip(TRIAL_TEXT_FIELDS, texts, vectors):
        out[field] = text
        out[vector_field] = vector

    for simple in (
        "overall_status",
        "phase",
        "study_type",
        "gender",
        "source",
        "source_url",
        "last_update_posted",
    ):
        if doc.get(simple) not in (None, ""):
            out[simple] = doc[simple]

    for date_field in ("start_date", "completion_date"):
        iso = _to_iso_date(doc.get(date_field))
        if iso:
            out[date_field] = iso

    for age_field in ("minimum_age", "maximum_age"):
        years = _age_to_years(doc.get(age_field))
        if years is not None:
            out[age_field] = years

    for nested in ("intervention", "location", "reference"):
        if doc.get(nested):
            out[nested] = doc[nested]

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
    for criterion in doc.get("criteria") or []:
        if not isinstance(criterion, dict):
            continue
        text = _preprocess_text(
            _flatten_text(criterion.get("criterion") or criterion.get("sentence"))
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
        rows.append(
            {
                "criteria_id": criteria_id,
                "nct_id": entry["nct_id"],
                "criterion": entry["criterion"],
                "entities": _entities_for_index(entry.get("entities")),
                "eligibility_type": entry["eligibility_type"],
                "criterion_vector": vector,
            }
        )
    return rows


def write_prepared_trial(row: dict[str, Any], folder: str | Path) -> Path:
    path = Path(folder) / f"{row['nct_id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_prepared_criteria(rows: Sequence[dict[str, Any]], folder: str | Path) -> int:
    if not rows:
        return 0
    trial_folder = Path(folder) / str(rows[0]["nct_id"])
    trial_folder.mkdir(parents=True, exist_ok=True)
    for row in rows:
        path = trial_folder / f"{row['criteria_id']}.json"
        path.write_text(json.dumps(row, indent=2, sort_keys=True), encoding="utf-8")
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


def _flatten_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(_flatten_text(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return " ".join(_flatten_text(item) for item in value)
    return str(value)


def _to_iso_date(value: Any) -> str | None:
    if not value:
        return None
    try:
        return dateutil.parser.parse(str(value)).date().isoformat()
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
