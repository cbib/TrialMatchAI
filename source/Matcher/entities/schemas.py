from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from Matcher.entities.types import EntitySchema


DEFAULT_SCHEMA_RESOURCE = "entity_schemas/trialmatchai.yaml"


class EntitySchemaError(ValueError):
    pass


def default_schema_path() -> Path:
    return Path(str(resources.files("Matcher").joinpath(DEFAULT_SCHEMA_RESOURCE)))


def load_entity_schemas(path: str | Path | None = None) -> list[EntitySchema]:
    schema_path = Path(path).expanduser() if path else default_schema_path()
    with schema_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return parse_entity_schemas(raw)


def parse_entity_schemas(raw: dict[str, Any]) -> list[EntitySchema]:
    entries = raw.get("entities")
    if not isinstance(entries, list) or not entries:
        raise EntitySchemaError("Entity schema must contain a non-empty 'entities' list.")

    schemas: list[EntitySchema] = []
    seen_ids: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise EntitySchemaError("Each entity schema entry must be an object.")

        schema_id = _required_string(entry, "id")
        if schema_id in seen_ids:
            raise EntitySchemaError(f"Duplicate entity schema id: {schema_id}")
        seen_ids.add(schema_id)

        threshold = float(entry.get("threshold", 0.8))
        if not 0 <= threshold <= 1:
            raise EntitySchemaError(
                f"Entity schema '{schema_id}' threshold must be between 0 and 1."
            )

        target_vocabularies = _string_tuple(entry.get("target_vocabularies", ()))
        if entry.get("linkable_fields", ["text"]) and not target_vocabularies:
            raise EntitySchemaError(
                f"Entity schema '{schema_id}' is linkable but has no target vocabularies."
            )

        schemas.append(
            EntitySchema(
                id=schema_id,
                label=_required_string(entry, "label"),
                entity_group=str(entry.get("entity_group") or schema_id),
                description=_required_string(entry, "description"),
                target_vocabularies=target_vocabularies,
                domain_hints=_string_tuple(entry.get("domain_hints", ())),
                linkable_fields=_string_tuple(entry.get("linkable_fields", ("text",))),
                threshold=threshold,
                query_expansion=bool(entry.get("query_expansion", False)),
                patterns=_string_tuple(entry.get("patterns", ())),
                aliases=_string_tuple(entry.get("aliases", ())),
            )
        )
    return schemas


def schema_by_label(schemas: list[EntitySchema]) -> dict[str, EntitySchema]:
    mapping: dict[str, EntitySchema] = {}
    for schema in schemas:
        for label in schema.recognizer_labels:
            mapping[label.casefold()] = schema
    return mapping


def _required_string(entry: dict[str, Any], key: str) -> str:
    value = entry.get(key)
    if not isinstance(value, str) or not value.strip():
        raise EntitySchemaError(f"Entity schema field '{key}' must be a non-empty string.")
    return value.strip()


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if not isinstance(value, list | tuple):
        raise EntitySchemaError("Schema string-list fields must be strings or lists.")
    cleaned = []
    for item in value:
        if not isinstance(item, str):
            raise EntitySchemaError("Schema string-list fields may only contain strings.")
        if item.strip():
            cleaned.append(item.strip())
    return tuple(cleaned)
