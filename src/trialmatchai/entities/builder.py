from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Sequence

from trialmatchai.entities.types import dedupe_strings


DEFAULT_OMOP_VOCABULARIES = (
    # OMOP standard vocabularies (OHDSI Athena). HGNC is listed for when it is
    # included in a download; it is not in every Athena bundle.
    "SNOMED",
    "ICD10",
    "ICD10CM",
    "LOINC",
    "RxNorm",
    "RxNorm Extension",
    "ATC",
    "MeSH",
    "OMOP Extension",
    "HGNC",
    "CIViC",
    "ClinVar",
    "OncoKB",
)


def build_omop_concept_rows(
    concept_csv: str | Path,
    synonym_csv: str | Path | None = None,
    *,
    vocabularies: Sequence[str] = DEFAULT_OMOP_VOCABULARIES,
) -> list[dict[str, Any]]:
    vocab_filter = {v.casefold() for v in vocabularies}
    synonyms_by_concept = _read_omop_synonyms(synonym_csv) if synonym_csv else {}
    rows: list[dict[str, Any]] = []
    with Path(concept_csv).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            vocabulary_id = (row.get("vocabulary_id") or "").strip()
            if vocab_filter and vocabulary_id.casefold() not in vocab_filter:
                continue
            if (row.get("invalid_reason") or "").strip():
                continue  # skip deprecated / non-current concepts
            concept_id = (row.get("concept_id") or "").strip()
            concept_name = (row.get("concept_name") or "").strip()
            concept_code = (row.get("concept_code") or "").strip()
            if not concept_id or not concept_name or not concept_code:
                continue
            synonyms = dedupe_strings(tuple(synonyms_by_concept.get(concept_id, ())))
            rows.append(
                _concept_row(
                    concept_id=concept_id,
                    vocabulary_id=vocabulary_id,
                    concept_code=concept_code,
                    concept_name=concept_name,
                    domain_id=(row.get("domain_id") or "").strip(),
                    concept_class_id=(row.get("concept_class_id") or "").strip(),
                    standard_concept=(row.get("standard_concept") or "").strip(),
                    synonyms=synonyms,
                )
            )
    return rows


def build_dictionary_rows(
    dictionary_path: str | Path,
    *,
    vocabulary_id: str,
    domain_id: str = "",
    concept_class_id: str = "Dictionary",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(dictionary_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or "||" not in stripped:
                continue
            identifiers, names = stripped.split("||", 1)
            synonyms = dedupe_strings(tuple(names.split("|")))
            concept_name = synonyms[0] if synonyms else ""
            if not concept_name:
                continue
            for raw_identifier in identifiers.split(","):
                concept_code = _strip_prefix(raw_identifier.strip(), vocabulary_id)
                if not concept_code:
                    continue
                rows.append(
                    _concept_row(
                        concept_id=f"{vocabulary_id}:{concept_code}",
                        vocabulary_id=vocabulary_id,
                        concept_code=concept_code,
                        concept_name=concept_name,
                        domain_id=domain_id,
                        concept_class_id=concept_class_id,
                        standard_concept="",
                        synonyms=synonyms,
                    )
                )
    return rows


def write_lancedb_table(
    rows: Sequence[dict[str, Any]],
    *,
    db_path: str | Path,
    table_name: str = "concepts",
    embeddings: Sequence[Sequence[float]] | None = None,
    recreate: bool = True,
) -> None:
    try:
        import lancedb  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Building a LanceDB concept table requires `uv sync --extra entity`."
        ) from exc

    payload = [dict(row) for row in rows]
    if embeddings is not None:
        if len(embeddings) != len(payload):
            raise ValueError("Embedding count must match concept row count.")
        for row, embedding in zip(payload, embeddings):
            row["embedding"] = list(embedding)

    db = lancedb.connect(str(db_path))
    if recreate or not _lancedb_table_exists(db, table_name):
        table = db.create_table(table_name, data=payload, mode="overwrite")
    else:
        table = db.open_table(table_name)
        table.add(payload)
    try:
        table.create_fts_index("fts_text", replace=True)
    except Exception:
        # Older LanceDB versions or tiny test builds may not support FTS indexes.
        pass


def concept_texts_for_embedding(rows: Sequence[dict[str, Any]]) -> list[str]:
    texts: list[str] = []
    for row in rows:
        synonyms = row.get("synonyms") or []
        if isinstance(synonyms, str):
            synonyms = synonyms.split("|")
        texts.append(" | ".join([row.get("concept_name", ""), *synonyms]).strip())
    return texts


def _lancedb_table_exists(db: Any, table_name: str) -> bool:
    try:
        return table_name in set(db.table_names())
    except Exception:
        return False


def _read_omop_synonyms(path: str | Path | None) -> dict[str, list[str]]:
    if path is None:
        return {}
    synonyms: dict[str, list[str]] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            concept_id = (row.get("concept_id") or "").strip()
            synonym = (row.get("concept_synonym_name") or "").strip()
            if concept_id and synonym:
                synonyms.setdefault(concept_id, []).append(synonym)
    return synonyms


def _concept_row(
    *,
    concept_id: str,
    vocabulary_id: str,
    concept_code: str,
    concept_name: str,
    domain_id: str,
    concept_class_id: str,
    standard_concept: str,
    synonyms: Iterable[str],
) -> dict[str, Any]:
    synonyms_tuple = dedupe_strings(tuple(synonyms))
    fts_text = " ".join([concept_name, *synonyms_tuple])
    return {
        "concept_id": concept_id,
        "vocabulary_id": vocabulary_id,
        "concept_code": concept_code,
        "concept_name": concept_name,
        "domain_id": domain_id,
        "concept_class_id": concept_class_id,
        "standard_concept": standard_concept,
        "synonyms": list(synonyms_tuple),
        "fts_text": fts_text,
    }


def _strip_prefix(identifier: str, vocabulary_id: str) -> str:
    if not identifier:
        return ""
    if ":" not in identifier:
        return identifier
    prefix, code = identifier.split(":", 1)
    if prefix.casefold() == vocabulary_id.casefold():
        return code
    return identifier
