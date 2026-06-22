from __future__ import annotations

import re
from importlib import resources
from typing import Any, Protocol, Sequence

from trialmatchai.entities.schemas import schema_by_label
from trialmatchai.entities.types import EntityAnnotation, EntitySchema, NO_ENTITY_ID
from trialmatchai.utils.logging_config import setup_logging

logger = setup_logging(__name__)

VARIANT_PATTERNS_RESOURCE = ("trialmatchai.entities", "resources/variant_patterns.tsv")


class EntityRecognizer(Protocol):
    def recognize(
        self, texts: Sequence[str], schemas: Sequence[EntitySchema]
    ) -> list[list[EntityAnnotation]]:
        ...


class DisabledRecognizer:
    def recognize(
        self, texts: Sequence[str], schemas: Sequence[EntitySchema]
    ) -> list[list[EntityAnnotation]]:
        return [[] for _ in texts]


class RegexSchemaRecognizer:
    """Small deterministic recognizer used for smoke tests and fixture runs."""

    def recognize(
        self, texts: Sequence[str], schemas: Sequence[EntitySchema]
    ) -> list[list[EntityAnnotation]]:
        compiled = [
            (schema, re.compile(pattern, re.IGNORECASE))
            for schema in schemas
            for pattern in schema.patterns
        ]
        results: list[list[EntityAnnotation]] = []
        for text in texts:
            annotations: list[EntityAnnotation] = []
            for schema, pattern in compiled:
                for match in pattern.finditer(text):
                    mention = match.group(0)
                    annotations.append(
                        EntityAnnotation(
                            entity_group=schema.entity_group,
                            text=mention,
                            start=match.start(),
                            end=match.end(),
                            score=max(schema.threshold, 0.95),
                            normalized_id=(NO_ENTITY_ID,),
                            schema_id=schema.id,
                        )
                    )
            results.append(resolve_overlaps(annotations))
        return results


def _load_variant_patterns() -> list[tuple[str, "re.Pattern[str]"]]:
    """Load curated genetic-variant regexes (label, compiled pattern)."""
    package, name = VARIANT_PATTERNS_RESOURCE
    try:
        raw = resources.files(package).joinpath(name).read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError, OSError):
        logger.warning("Variant pattern resource not found; variant detection off.")
        return []
    patterns: list[tuple[str, "re.Pattern[str]"]] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        label = parts[0].strip() or "variant"
        # The table stores patterns with doubled backslashes; collapse to one.
        pattern_text = parts[-1].replace("\\\\", "\\")
        try:
            patterns.append((label, re.compile(pattern_text)))
        except re.error as exc:
            logger.warning("Skipping invalid variant pattern %r: %s", label, exc)
    return patterns


class RegexVariantRecognizer:
    """Deterministic recognizer for genetic variants (HGVS mutations, fusions,
    chromosome arms, ...) from a curated pattern table.

    Runs alongside the model recognizer to capture precise biomedical strings —
    e.g. ``c.1799T>A``, ``p.V600E``, ``EGFR fusion`` — that a generalist NER
    model often misses or mangles. Emits high-confidence spans tagged with the
    variant type as the entity group.
    """

    def __init__(self, patterns: list[tuple[str, "re.Pattern[str]"]] | None = None):
        self._patterns = patterns if patterns is not None else _load_variant_patterns()

    def recognize(
        self, texts: Sequence[str], schemas: Sequence[EntitySchema]
    ) -> list[list[EntityAnnotation]]:
        results: list[list[EntityAnnotation]] = []
        for text in texts:
            annotations: list[EntityAnnotation] = []
            for label, pattern in self._patterns:
                for match in pattern.finditer(text):
                    start, end = match.start(), match.end()
                    if end <= start or not match.group(0).strip():
                        continue  # skip zero-width / whitespace-only matches
                    annotations.append(
                        EntityAnnotation(
                            entity_group=label,
                            text=match.group(0).strip(),
                            start=start,
                            end=end,
                            score=0.97,
                            normalized_id=(NO_ENTITY_ID,),
                            schema_id=None,
                        )
                    )
            results.append(resolve_overlaps(annotations))
        return results


class CompositeRecognizer:
    """Runs a primary recognizer plus augmenters and merges their annotations.

    Overlaps are resolved by confidence then span length, so a high-precision
    variant match wins over a lower-confidence model span covering the same text.
    """

    def __init__(self, primary: EntityRecognizer, *augmenters: EntityRecognizer):
        self._recognizers = [primary, *augmenters]

    def recognize(
        self, texts: Sequence[str], schemas: Sequence[EntitySchema]
    ) -> list[list[EntityAnnotation]]:
        per_recognizer = [r.recognize(texts, schemas) for r in self._recognizers]
        merged: list[list[EntityAnnotation]] = []
        for i in range(len(texts)):
            combined: list[EntityAnnotation] = []
            for recognizer_results in per_recognizer:
                combined.extend(recognizer_results[i])
            merged.append(resolve_overlaps(combined))
        return merged


class GLiNER2Recognizer:
    def __init__(
        self,
        model_name: str,
        *,
        revision: str | None = None,
        device: str | None = None,
        trust_remote_code: bool = False,
        batch_size: int = 8,
    ):
        try:
            from gliner2 import GLiNER2  # type: ignore
        except Exception as exc:  # pragma: no cover - exercised without optional dep
            raise RuntimeError(
                "entity_extraction.backend=gliner2 requires the entity extra "
                "(`uv sync --extra entity`) and a GLiNER2-compatible model."
            ) from exc

        kwargs: dict[str, Any] = {}
        if revision:
            kwargs["revision"] = revision
        if trust_remote_code:
            kwargs["trust_remote_code"] = trust_remote_code
        self.model = GLiNER2.from_pretrained(model_name, **kwargs)
        if device and device != "auto" and hasattr(self.model, "to"):
            self.model.to(device)
        self.batch_size = batch_size

    def recognize(
        self, texts: Sequence[str], schemas: Sequence[EntitySchema]
    ) -> list[list[EntityAnnotation]]:
        labels = [schema.label for schema in schemas]
        label_map = schema_by_label(list(schemas))
        return [
            resolve_overlaps(
                _parse_model_entities(
                    _call_extractor(self.model, text, labels, schemas),
                    text,
                    label_map,
                )
            )
            for text in texts
        ]


class GLiNERRecognizer:
    def __init__(
        self,
        model_name: str,
        *,
        revision: str | None = None,
        device: str | None = None,
        trust_remote_code: bool = False,
        batch_size: int = 8,
    ):
        try:
            from gliner import GLiNER  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "entity_extraction.backend=gliner requires the GLiNER dependency."
            ) from exc

        kwargs: dict[str, Any] = {}
        if revision:
            kwargs["revision"] = revision
        if trust_remote_code:
            kwargs["trust_remote_code"] = trust_remote_code
        self.model = GLiNER.from_pretrained(model_name, **kwargs)
        if device and device != "auto" and hasattr(self.model, "to"):
            self.model.to(device)
        self.batch_size = batch_size

    def recognize(
        self, texts: Sequence[str], schemas: Sequence[EntitySchema]
    ) -> list[list[EntityAnnotation]]:
        labels = [schema.label for schema in schemas]
        label_map = schema_by_label(list(schemas))
        results: list[list[EntityAnnotation]] = []
        for text in texts:
            raw = self.model.predict_entities(text, labels)
            results.append(resolve_overlaps(_parse_model_entities(raw, text, label_map)))
        return results


def build_recognizer(config: dict[str, Any]) -> EntityRecognizer:
    backend = str(config.get("backend", "gliner2")).lower()
    if backend == "disabled":
        return DisabledRecognizer()
    if backend == "regex":
        return RegexSchemaRecognizer()
    if backend == "gliner":
        recognizer: EntityRecognizer = GLiNERRecognizer(
            model_name=config.get("fallback_model_name")
            or config.get("model_name")
            or "urchade/gliner_base",
            revision=config.get("model_revision"),
            device=config.get("device", "auto"),
            trust_remote_code=bool(config.get("trust_remote_code", False)),
            batch_size=int(config.get("batch_size", 8)),
        )
    elif backend == "gliner2":
        recognizer = GLiNER2Recognizer(
            model_name=config.get("model_name", "fastino/gliner2-base"),
            revision=config.get("model_revision"),
            device=config.get("device", "auto"),
            trust_remote_code=bool(config.get("trust_remote_code", False)),
            batch_size=int(config.get("batch_size", 8)),
        )
    else:
        raise ValueError(
            "entity_extraction.backend must be one of: gliner2, gliner, regex, disabled."
        )

    # Augment model NER with the deterministic variant recognizer (on by default)
    # so precise HGVS/fusion strings are captured alongside model spans.
    if bool(config.get("variant_regex", True)):
        variants = _load_variant_patterns()
        if variants:
            return CompositeRecognizer(recognizer, RegexVariantRecognizer(variants))
    return recognizer


def resolve_overlaps(
    annotations: Sequence[EntityAnnotation],
) -> list[EntityAnnotation]:
    ranked = sorted(
        annotations,
        key=lambda ann: (ann.score, ann.end - ann.start),
        reverse=True,
    )
    accepted: list[EntityAnnotation] = []
    for candidate in ranked:
        if candidate.start < 0 or candidate.end <= candidate.start:
            continue
        if any(_overlaps(candidate, current) for current in accepted):
            continue
        accepted.append(candidate)
    return sorted(accepted, key=lambda ann: (ann.start, ann.end))


def _overlaps(left: EntityAnnotation, right: EntityAnnotation) -> bool:
    return left.start < right.end and right.start < left.end


def _call_extractor(
    model: Any,
    text: str,
    labels: list[str],
    schemas: Sequence[EntitySchema],
) -> list[dict[str, Any]]:
    if hasattr(model, "extract_entities"):
        schema_payload = {
            schema.label: schema.description
            for schema in schemas
        }
        for kwargs in (
            {"schema": schema_payload},
            {"labels": labels},
            {},
        ):
            try:
                return model.extract_entities(text, **kwargs)
            except TypeError:
                continue
    if hasattr(model, "predict_entities"):
        return model.predict_entities(text, labels)
    raise RuntimeError("Selected entity model does not expose an entity extraction API.")


def _parse_model_entities(
    raw_entities: Sequence[dict[str, Any]],
    text: str,
    label_map: dict[str, EntitySchema],
) -> list[EntityAnnotation]:
    parsed: list[EntityAnnotation] = []
    for raw in raw_entities or []:
        label = str(raw.get("label") or raw.get("entity_group") or raw.get("type") or "")
        schema = label_map.get(label.casefold())
        if schema is None:
            continue
        mention = str(raw.get("text") or raw.get("span") or raw.get("mention") or "")
        start = _as_int(raw.get("start"))
        end = _as_int(raw.get("end"))
        if start is None or end is None:
            start, end = _find_span(text, mention)
        if start is None or end is None:
            continue
        mention = text[start:end]
        score = float(raw.get("score") or raw.get("confidence") or 0.0)
        if score < schema.threshold:
            continue
        parsed.append(
            EntityAnnotation(
                entity_group=schema.entity_group,
                text=mention,
                start=start,
                end=end,
                score=score,
                normalized_id=(NO_ENTITY_ID,),
                schema_id=schema.id,
            )
        )
    return parsed


def _find_span(text: str, mention: str) -> tuple[int | None, int | None]:
    if not mention:
        return None, None
    start = text.casefold().find(mention.casefold())
    if start < 0:
        return None, None
    return start, start + len(mention)


def _as_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None
