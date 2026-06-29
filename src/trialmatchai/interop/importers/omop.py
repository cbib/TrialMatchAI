from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from trialmatchai.interop.models import (
    Demographics,
    Location,
    NormalizedCode,
    PatientNote,
    PatientProfile,
    Provenance,
)
from trialmatchai.interop.utils import (
    age_years_from_birth_date,
    clean_text,
    make_fact,
    normalize_gender,
    parse_date,
    safe_patient_id,
    source_path_string,
)
from trialmatchai.utils.logging_config import setup_logging

logger = setup_logging(__name__)


def import_omop_extract(
    path: str | Path,
    *,
    strict: bool = False,
) -> list[PatientProfile]:
    root = Path(path)
    tables = _load_tables(root, strict=strict)
    concepts = _concept_lookup(tables.get("concept"))
    location_index = _location_index(tables.get("location"))
    person = tables.get("person")
    if person is None or person.empty:
        if strict:
            raise ValueError(f"OMOP extract is missing PERSON table: {root}")
        return []

    # Group each child table by normalized person_id ONCE, so each patient is an
    # O(1) lookup instead of a full-table scan (no N+1), and the join is robust
    # to float promotion from null person_id and to id sanitization.
    grouped = {
        name: _group_by_person(tables.get(name))
        for name in (
            "condition_occurrence",
            "measurement",
            "drug_exposure",
            "procedure_occurrence",
            "observation",
            "note",
            "note_nlp",
        )
    }

    profiles: list[PatientProfile] = []
    for row in person.to_dict("records"):
        raw_person_id = row.get("person_id")
        person_key = _normalize_join_id(raw_person_id)
        patient_id = safe_patient_id(raw_person_id, "omop-patient")
        provenance = Provenance(
            source_format="omop",
            source_id=patient_id,
            source_path=source_path_string(root),
            source_table="PERSON",
        )
        birth_date = _person_birth_date(row)
        profile = PatientProfile(
            patient_id=patient_id,
            demographics=Demographics(
                sex=_concept_label(row.get("gender_concept_id"), concepts)
                or normalize_gender(row.get("gender_source_value")),
                gender=normalize_gender(row.get("gender_source_value")),
                birth_date=birth_date,
                age_years=age_years_from_birth_date(birth_date),
            ),
            location=_person_location(row, location_index, concepts),
            provenance=[provenance],
        )
        _add_condition_rows(profile, grouped["condition_occurrence"].get(person_key, []), concepts, root)
        _add_measurement_rows(profile, grouped["measurement"].get(person_key, []), concepts, root)
        _add_drug_rows(profile, grouped["drug_exposure"].get(person_key, []), concepts, root)
        _add_procedure_rows(profile, grouped["procedure_occurrence"].get(person_key, []), concepts, root)
        _add_observation_rows(profile, grouped["observation"].get(person_key, []), concepts, root)
        _add_note_rows(profile, grouped["note"].get(person_key, []), root)
        _add_note_nlp_rows(profile, grouped["note_nlp"].get(person_key, []), concepts, root)
        profiles.append(profile)
    return profiles


def _load_tables(root: Path, *, strict: bool = False) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for file_path in root.iterdir():
        if file_path.suffix.casefold() not in {".csv", ".parquet"}:
            continue
        name = file_path.stem.casefold()
        try:
            if file_path.suffix.casefold() == ".csv":
                table = pd.read_csv(file_path)
            else:
                table = pd.read_parquet(file_path)
        except Exception:
            # A single malformed table must not abort the whole import.
            logger.warning("OMOP: failed to read %s; skipping", file_path, exc_info=True)
            if strict:
                raise
            continue
        table.columns = [str(column).casefold() for column in table.columns]
        tables[name] = table
    return tables


def _concept_lookup(table: pd.DataFrame | None) -> dict[Any, dict[str, Any]]:
    if table is None:
        return {}
    return {
        record.get("concept_id"): record
        for record in table.to_dict("records")
        if record.get("concept_id") is not None
    }


def _person_birth_date(row) -> Any:
    if row.get("birth_datetime"):
        return parse_date(row.get("birth_datetime"))
    year = row.get("year_of_birth")
    if pd.isna(year):
        return None
    month = int(row.get("month_of_birth") or 1)
    day = int(row.get("day_of_birth") or 1)
    return parse_date(f"{int(year):04d}-{month:02d}-{day:02d}")


def _add_condition_rows(
    profile: PatientProfile,
    rows: list[dict[str, Any]],
    concepts: dict[Any, dict[str, Any]],
    root: Path,
) -> None:
    for row in rows:
        code = _omop_code(row.get("condition_concept_id"), concepts)
        label = _concept_label(row.get("condition_concept_id"), concepts) or clean_text(
            row.get("condition_source_value")
        )
        if label:
            profile.conditions.append(
                make_fact(
                    category="condition",
                    label=label,
                    original_code=code,
                    provenance=_row_provenance(root, profile.patient_id, "CONDITION_OCCURRENCE"),
                    temporality=clean_text(row.get("condition_start_date")) or None,
                )
            )


def _add_measurement_rows(
    profile: PatientProfile,
    rows: list[dict[str, Any]],
    concepts: dict[Any, dict[str, Any]],
    root: Path,
) -> None:
    for row in rows:
        code = _omop_code(row.get("measurement_concept_id"), concepts)
        label = _concept_label(row.get("measurement_concept_id"), concepts) or clean_text(
            row.get("measurement_source_value")
        )
        value = row.get("value_as_number")
        if pd.isna(value):
            value = _concept_label(row.get("value_as_concept_id"), concepts)
        unit = _concept_label(row.get("unit_concept_id"), concepts)
        if label:
            profile.observations.append(
                make_fact(
                    category="observation",
                    label=label,
                    original_code=code,
                    provenance=_row_provenance(root, profile.patient_id, "MEASUREMENT"),
                    description=clean_text(
                        f"{value if value is not None else ''} {unit or ''}"
                    )
                    or None,
                    temporality=clean_text(row.get("measurement_date")) or None,
                )
            )


def _add_drug_rows(
    profile: PatientProfile,
    rows: list[dict[str, Any]],
    concepts: dict[Any, dict[str, Any]],
    root: Path,
) -> None:
    for row in rows:
        code = _omop_code(row.get("drug_concept_id"), concepts)
        label = _concept_label(row.get("drug_concept_id"), concepts) or clean_text(
            row.get("drug_source_value")
        )
        if label:
            profile.medications.append(
                make_fact(
                    category="medication",
                    label=label,
                    original_code=code,
                    provenance=_row_provenance(root, profile.patient_id, "DRUG_EXPOSURE"),
                    temporality=clean_text(row.get("drug_exposure_start_date")) or None,
                )
            )


def _add_procedure_rows(
    profile: PatientProfile,
    rows: list[dict[str, Any]],
    concepts: dict[Any, dict[str, Any]],
    root: Path,
) -> None:
    for row in rows:
        code = _omop_code(row.get("procedure_concept_id"), concepts)
        label = _concept_label(row.get("procedure_concept_id"), concepts) or clean_text(
            row.get("procedure_source_value")
        )
        if label:
            profile.procedures.append(
                make_fact(
                    category="procedure",
                    label=label,
                    original_code=code,
                    provenance=_row_provenance(root, profile.patient_id, "PROCEDURE_OCCURRENCE"),
                    temporality=clean_text(row.get("procedure_date")) or None,
                )
            )


def _add_observation_rows(
    profile: PatientProfile,
    rows: list[dict[str, Any]],
    concepts: dict[Any, dict[str, Any]],
    root: Path,
) -> None:
    for row in rows:
        code = _omop_code(row.get("observation_concept_id"), concepts)
        label = _concept_label(row.get("observation_concept_id"), concepts) or clean_text(
            row.get("observation_source_value")
        )
        if label:
            profile.observations.append(
                make_fact(
                    category="observation",
                    label=label,
                    original_code=code,
                    provenance=_row_provenance(root, profile.patient_id, "OBSERVATION"),
                    temporality=clean_text(row.get("observation_date")) or None,
                )
            )


def _add_note_rows(
    profile: PatientProfile,
    rows: list[dict[str, Any]],
    root: Path,
) -> None:
    for row in rows:
        text = clean_text(row.get("note_text"))
        if not text:
            continue
        provenance = _row_provenance(root, profile.patient_id, "NOTE")
        profile.notes.append(
            PatientNote(
                note_id=clean_text(row.get("note_id")) or f"{profile.patient_id}-note",
                text=text,
                note_type=clean_text(row.get("note_type_concept_id")) or "omop-note",
                provenance=provenance,
            )
        )


def _add_note_nlp_rows(
    profile: PatientProfile,
    rows: list[dict[str, Any]],
    concepts: dict[Any, dict[str, Any]],
    root: Path,
) -> None:
    for row in rows:
        code = _omop_code(row.get("note_nlp_concept_id"), concepts)
        label = _concept_label(row.get("note_nlp_concept_id"), concepts) or clean_text(
            row.get("lexical_variant") or row.get("snippet")
        )
        if not label:
            continue
        concept = concepts.get(row.get("note_nlp_concept_id"), {})
        domain = clean_text(concept.get("domain_id")).casefold()
        category = "condition" if domain == "condition" else "observation"
        profile.add_fact(
            make_fact(
                category=category,
                label=label,
                original_code=code,
                provenance=_row_provenance(root, profile.patient_id, "NOTE_NLP"),
                evidence_text=clean_text(row.get("snippet")) or None,
                evidence_start=_int_or_none(row.get("offset")),
                # OMOP note_nlp.term_exists marks whether the term is asserted:
                # "N"/"No"/"0"/"False" mean it is NEGATED (e.g. "no metastasis").
                negated=str(row.get("term_exists")).strip().casefold()
                in {"n", "no", "false", "0", "f"},
            )
        )


def _normalize_join_id(value: Any) -> str:
    """Canonicalize a person_id for joining across tables.

    Robust to float promotion (a null in the column makes person_id 1 -> 1.0)
    and to the raw-vs-string mismatch, so PERSON and child tables join on the
    same key regardless of dtype.
    """
    if value is None:
        return ""
    if isinstance(value, float):
        if pd.isna(value):
            return ""
        if value.is_integer():
            return str(int(value))
        return str(value)
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def _location_index(table: pd.DataFrame | None) -> dict[str, dict[str, Any]]:
    if table is None or table.empty or "location_id" not in table.columns:
        return {}
    index: dict[str, dict[str, Any]] = {}
    for record in table.to_dict("records"):
        key = _normalize_join_id(record.get("location_id"))
        if key:
            index.setdefault(key, record)
    return index


def _person_location(
    row: dict[str, Any],
    location_index: dict[str, dict[str, Any]],
    concepts: dict[Any, dict[str, Any]],
) -> Location | None:
    if not location_index:
        return None
    location_id = _normalize_join_id(row.get("location_id"))
    if not location_id:
        return None
    loc = location_index.get(location_id)
    if not loc:
        return None

    def field(name: str) -> str | None:
        value = loc.get(name)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        return clean_text(value) or None

    country = field("country_source_value") or _concept_label(
        loc.get("country_concept_id"), concepts
    )
    state = field("state")
    city = field("city")
    if not (country or state or city):
        return None
    return Location(country=country, state=state, city=city)


def _group_by_person(table: pd.DataFrame | None) -> dict[str, list[dict[str, Any]]]:
    if table is None or table.empty or "person_id" not in table.columns:
        return {}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in table.to_dict("records"):
        key = _normalize_join_id(record.get("person_id"))
        if not key:
            continue
        grouped.setdefault(key, []).append(record)
    return grouped


def _row_provenance(root: Path, patient_id: str, table_name: str) -> Provenance:
    return Provenance(
        source_format="omop",
        source_id=patient_id,
        source_path=source_path_string(root),
        source_table=table_name,
    )


def _omop_code(concept_id: Any, concepts: dict[Any, dict[str, Any]]) -> NormalizedCode | None:
    if concept_id is None or pd.isna(concept_id):
        return None
    concept = concepts.get(concept_id)
    if not concept:
        return NormalizedCode(
            vocabulary="OMOP",
            code=str(int(concept_id)) if isinstance(concept_id, float) else str(concept_id),
            mapping_status="unmapped",
        )
    return NormalizedCode(
        vocabulary=clean_text(concept.get("vocabulary_id")) or "OMOP",
        code=clean_text(concept.get("concept_code")) or str(concept_id),
        label=clean_text(concept.get("concept_name")) or None,
        mapping_status="exact",
    )


def _concept_label(concept_id: Any, concepts: dict[Any, dict[str, Any]]) -> str | None:
    if concept_id is None or pd.isna(concept_id):
        return None
    concept = concepts.get(concept_id)
    return clean_text((concept or {}).get("concept_name")) or None


def _int_or_none(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
