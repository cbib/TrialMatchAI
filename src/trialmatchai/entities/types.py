from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


NO_ENTITY_ID = "CUI-less"


@dataclass(frozen=True)
class EntitySchema:
    id: str
    label: str
    entity_group: str
    description: str
    target_vocabularies: tuple[str, ...] = ()
    domain_hints: tuple[str, ...] = ()
    linkable_fields: tuple[str, ...] = ("text",)
    threshold: float = 0.8
    query_expansion: bool = False
    patterns: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()

    @property
    def is_linkable(self) -> bool:
        return bool(self.linkable_fields)

    @property
    def recognizer_labels(self) -> tuple[str, ...]:
        labels = [self.label, self.id, self.entity_group, *self.aliases]
        return tuple(dict.fromkeys(label for label in labels if label))


@dataclass(frozen=True)
class ConceptCandidate:
    concept_id: str
    vocabulary_id: str
    concept_code: str
    concept_name: str
    domain_id: str = ""
    concept_class_id: str = ""
    standard_concept: str = ""
    synonyms: tuple[str, ...] = ()
    score: float = 0.0
    source_scores: dict[str, float] = field(default_factory=dict)

    @property
    def normalized_id(self) -> str:
        return normalize_concept_id(self.vocabulary_id, self.concept_code)

    def to_dict(self) -> dict[str, Any]:
        return {
            "concept_id": self.concept_id,
            "vocabulary_id": self.vocabulary_id,
            "concept_code": self.concept_code,
            "concept_name": self.concept_name,
            "domain_id": self.domain_id,
            "concept_class_id": self.concept_class_id,
            "standard_concept": self.standard_concept,
            "synonyms": list(self.synonyms),
            "score": self.score,
            "source_scores": dict(self.source_scores),
            "normalized_id": self.normalized_id,
        }


@dataclass(frozen=True)
class EntityAnnotation:
    entity_group: str
    text: str
    start: int
    end: int
    score: float
    normalized_id: tuple[str, ...] = (NO_ENTITY_ID,)
    synonyms: tuple[str, ...] = ()
    concept_candidates: tuple[ConceptCandidate, ...] = ()
    linker_score: float | None = None
    linker_status: str = "not_linked"
    schema_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_group": self.entity_group,
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "score": self.score,
            "normalized_id": list(self.normalized_id),
            "synonyms": list(self.synonyms),
            "concept_candidates": [
                candidate.to_dict() for candidate in self.concept_candidates
            ],
            "linker_score": self.linker_score,
            "linker_status": self.linker_status,
        }

    def to_index_entity(self) -> dict[str, Any]:
        data = self.to_dict()
        data["entity"] = self.text
        data["class"] = self.entity_group
        return data


def normalize_concept_id(vocabulary_id: str, concept_code: str) -> str:
    vocab = vocabulary_id.strip()
    code = concept_code.strip()
    if not vocab or not code:
        return NO_ENTITY_ID
    if ":" in code:
        return code

    canonical_vocab = {
        "SNOMED": "SNOMED",
        "SNOMEDCT": "SNOMED",
        "ICD10": "ICD10",
        "ICD10CM": "ICD10CM",
        "LOINC": "LOINC",
        "RXNORM": "RxNorm",
        "RXNORM EXTENSION": "RxNorm",
        "ATC": "ATC",
        "ENTREZGENE": "EntrezGene",
        "NCBIGENE": "EntrezGene",
        "CELLOSAURUS": "Cellosaurus",
        "NCBITAXON": "NCBITaxon",
    }.get(vocab.upper(), vocab)
    return f"{canonical_vocab}:{code}"


def dedupe_strings(values: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        cleaned = value.strip()
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(cleaned)
    return tuple(deduped)
