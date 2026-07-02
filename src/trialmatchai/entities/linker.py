from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from trialmatchai.entities.schemas import schema_by_label
from trialmatchai.entities.types import (
    NO_ENTITY_ID,
    ConceptCandidate,
    EntityAnnotation,
    EntitySchema,
    dedupe_strings,
)
from trialmatchai.utils.logging_config import setup_logging

logger = setup_logging(__name__)


class ConceptStore(Protocol):
    def search(
        self,
        query: str,
        *,
        vocabularies: Sequence[str] = (),
        domain_hints: Sequence[str] = (),
        query_vector: Sequence[float] | None = None,
        limit: int = 10,
    ) -> list[ConceptCandidate]:
        ...


class InMemoryConceptStore:
    def __init__(self, concepts: Sequence[ConceptCandidate | dict[str, Any]]):
        self.concepts = [
            item if isinstance(item, ConceptCandidate) else concept_from_mapping(item)
            for item in concepts
        ]

    def search(
        self,
        query: str,
        *,
        vocabularies: Sequence[str] = (),
        domain_hints: Sequence[str] = (),
        query_vector: Sequence[float] | None = None,
        limit: int = 10,
    ) -> list[ConceptCandidate]:
        vocab_filter = {v.casefold() for v in vocabularies}
        domain_filter = {d.casefold() for d in domain_hints}
        scored: list[ConceptCandidate] = []
        for concept in self.concepts:
            if vocab_filter and concept.vocabulary_id.casefold() not in vocab_filter:
                continue
            if domain_filter and concept.domain_id.casefold() not in domain_filter:
                continue
            score = _lexical_score(query, concept)
            if score <= 0:
                continue
            scored.append(
                replace(
                    concept,
                    score=score,
                    source_scores={"lexical": score},
                )
            )
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:limit]


class LanceDBConceptStore:
    def __init__(
        self,
        db_path: str | Path,
        *,
        table_name: str = "concepts",
        embedder: Any | None = None,
    ):
        try:
            import lancedb  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "LanceDB concept linking requires the entity extra "
                "(`uv sync --extra entity`)."
            ) from exc

        self.db_path = str(db_path)
        self.table_name = table_name
        self.embedder = embedder
        self.db = lancedb.connect(self.db_path)
        self.table = self.db.open_table(table_name)

    def search(
        self,
        query: str,
        *,
        vocabularies: Sequence[str] = (),
        domain_hints: Sequence[str] = (),
        query_vector: Sequence[float] | None = None,
        limit: int = 10,
    ) -> list[ConceptCandidate]:
        vector = query_vector or self._embed_query(query)
        lexical_rows = self._search_fts(query, vocabularies, domain_hints, limit)
        vector_rows = (
            self._search_vector(vector, vocabularies, domain_hints, limit)
            if vector
            else []
        )
        return _rrf_merge(lexical_rows, vector_rows, limit)

    def _embed_query(self, query: str) -> list[float] | None:
        if self.embedder is None:
            return None
        # Empty mention would make embed_text() raise; degrade to FTS-only instead.
        if not query.strip():
            return None
        if hasattr(self.embedder, "embed_text"):
            return self.embedder.embed_text(query)
        if hasattr(self.embedder, "embed_texts"):
            vectors = self.embedder.embed_texts([query])
            return vectors[0] if vectors else None
        return None

    def _search_fts(
        self,
        query: str,
        vocabularies: Sequence[str],
        domain_hints: Sequence[str],
        limit: int,
    ) -> list[ConceptCandidate]:
        try:
            search = self.table.search(query, query_type="fts")
            where = _lancedb_filter(vocabularies, domain_hints)
            if where:
                search = search.where(where)
            rows = search.limit(limit).to_list()
        except Exception as exc:
            logger.warning("LanceDB FTS concept search failed: %s", exc)
            rows = []
        return [concept_from_mapping(row) for row in rows]

    def _search_vector(
        self,
        vector: Sequence[float],
        vocabularies: Sequence[str],
        domain_hints: Sequence[str],
        limit: int,
    ) -> list[ConceptCandidate]:
        try:
            search = self.table.search(vector)
            where = _lancedb_filter(vocabularies, domain_hints)
            if where:
                search = search.where(where)
            rows = search.limit(limit).to_list()
        except Exception as exc:
            logger.warning("LanceDB vector concept search failed: %s", exc)
            rows = []
        return [concept_from_mapping(row) for row in rows]


class ConceptLinker:
    def __init__(
        self,
        store: ConceptStore | None,
        schemas: Sequence[EntitySchema],
        *,
        accept_threshold: float = 0.7,
        reject_threshold: float = 0.5,
        margin: float = 0.05,
        reranker: Callable[[str, Sequence[ConceptCandidate]], Sequence[ConceptCandidate]]
        | None = None,
        search_limit: int = 10,
    ):
        if reject_threshold > accept_threshold:
            raise ValueError("reject_threshold must be <= accept_threshold.")
        self.store = store
        self.schemas = {schema.id: schema for schema in schemas}
        self.schemas_by_label = schema_by_label(list(schemas))
        self.accept_threshold = accept_threshold
        self.reject_threshold = reject_threshold
        self.margin = margin
        # Optional reranker: reorders candidates before the accept gate; None keeps RRF order.
        self.reranker = reranker
        self.search_limit = search_limit

    def link_annotations(
        self, annotations: Sequence[EntityAnnotation]
    ) -> list[EntityAnnotation]:
        return [self.link_annotation(annotation) for annotation in annotations]

    def link_annotation(self, annotation: EntityAnnotation) -> EntityAnnotation:
        schema = self._schema_for(annotation)
        if schema is None or not schema.is_linkable:
            return replace(annotation, linker_status="not_linkable")
        if self.store is None:
            return replace(
                annotation,
                normalized_id=(NO_ENTITY_ID,),
                linker_status="concept_store_unavailable",
            )

        candidates = self.store.search(
            annotation.text,
            vocabularies=schema.target_vocabularies,
            domain_hints=schema.domain_hints,
            limit=self.search_limit,
        )
        if not candidates:
            return replace(
                annotation,
                normalized_id=(NO_ENTITY_ID,),
                concept_candidates=(),
                linker_score=0.0,
                linker_status="rejected",
            )

        candidates = list(candidates)
        if self.reranker is not None:
            reranked = list(self.reranker(annotation.text, candidates))
            if reranked:
                candidates = reranked
        top = candidates[0]

        # Store ranking is RRF-normalized (#1 always ~1.0), so gate ACCEPT on an absolute
        # lexical match to the concept name/synonyms and abstain when it's not met.
        quality = _lexical_score(annotation.text, top)
        runner_up = (
            _lexical_score(annotation.text, candidates[1]) if len(candidates) > 1 else 0.0
        )
        status = gate_status(
            quality,
            runner_up=runner_up,
            accept_threshold=self.accept_threshold,
            reject_threshold=self.reject_threshold,
            margin=self.margin,
        )
        if status == "accepted":
            return replace(
                annotation,
                normalized_id=(top.normalized_id,),
                synonyms=_candidate_synonyms(top),
                concept_candidates=tuple(candidates),
                linker_score=quality,
                linker_status=status,
            )
        return replace(
            annotation,
            normalized_id=(NO_ENTITY_ID,),
            synonyms=(),
            concept_candidates=tuple(candidates),
            linker_score=quality,
            linker_status=status,
        )

    def _schema_for(self, annotation: EntityAnnotation) -> EntitySchema | None:
        if annotation.schema_id and annotation.schema_id in self.schemas:
            return self.schemas[annotation.schema_id]
        return self.schemas_by_label.get(annotation.entity_group.casefold())


def concept_from_mapping(row: dict[str, Any]) -> ConceptCandidate:
    synonyms = row.get("synonyms") or ()
    if isinstance(synonyms, str):
        synonyms = [part.strip() for part in synonyms.split("|")]
    return ConceptCandidate(
        concept_id=str(row.get("concept_id") or ""),
        vocabulary_id=str(row.get("vocabulary_id") or ""),
        concept_code=str(row.get("concept_code") or ""),
        concept_name=str(row.get("concept_name") or ""),
        domain_id=str(row.get("domain_id") or ""),
        concept_class_id=str(row.get("concept_class_id") or ""),
        standard_concept=str(row.get("standard_concept") or ""),
        synonyms=dedupe_strings(tuple(synonyms)),
        score=float(row.get("score") or row.get("_score") or 0.0),
        source_scores=dict(row.get("source_scores") or {}),
    )


def _candidate_synonyms(candidate: ConceptCandidate) -> tuple[str, ...]:
    return dedupe_strings((candidate.concept_name, *candidate.synonyms))


def gate_status(
    quality: float,
    *,
    runner_up: float = 0.0,
    accept_threshold: float,
    reject_threshold: float,
    margin: float = 0.0,
) -> str:
    """Accept / reject / ambiguous on an ABSOLUTE match-quality score in [0, 1].

    <reject rejects; [reject, accept) is ambiguous; >=accept accepts unless a runner-up is
    also >=accept and within ``margin`` (near-tie -> ambiguous). Only 'accepted' links.
    """
    if quality < reject_threshold:
        return "rejected"
    if quality < accept_threshold:
        return "ambiguous"
    if runner_up >= accept_threshold and (quality - runner_up) < margin:
        return "ambiguous"
    return "accepted"


def lexical_reranker(
    query: str, candidates: Sequence[ConceptCandidate]
) -> list[ConceptCandidate]:
    """Reorder candidates by absolute lexical match to ``query`` (stable, scispaCy-style),
    so an exact name/synonym match ranked below RRF's #1 is promoted before the accept gate.
    Same (query, candidates) -> candidates shape as a cross-encoder reranker.
    """
    order = sorted(
        range(len(candidates)),
        key=lambda i: (-_lexical_score(query, candidates[i]), i),
    )
    return [candidates[i] for i in order]


# Partial-containment coverage bonus requires the mention be at least this long,
# so tiny mentions don't earn it.
_MIN_CONTAINMENT_CHARS = 3


def _lexical_score(query: str, concept: ConceptCandidate) -> float:
    query_norm = _normalize_text(query)
    if not query_norm:
        return 0.0
    query_tokens = set(query_norm.split())
    names = [concept.concept_name, *concept.synonyms]
    best = 0.0
    for name in names:
        name_norm = _normalize_text(name)
        if not name_norm:
            continue
        if query_norm == name_norm:
            best = max(best, 1.0)
            continue
        # Base signal is token Jaccard; a subset containment earns a length-ratio score
        # (not a flat constant) so a bare "carcinoma" stays below the accept gate.
        score = _token_jaccard(query_norm, name_norm)
        name_tokens = set(name_norm.split())
        if len(query_norm) >= _MIN_CONTAINMENT_CHARS and (
            query_tokens <= name_tokens or name_tokens <= query_tokens
        ):
            ratio = min(len(query_norm), len(name_norm)) / max(
                len(query_norm), len(name_norm)
            )
            score = max(score, ratio)
        best = max(best, score)
    return best


def _normalize_text(value: str) -> str:
    return " ".join(value.casefold().replace("-", " ").split())


def _token_jaccard(left: str, right: str) -> float:
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    if overlap == 0:
        return 0.0
    return 0.2 + 0.6 * (overlap / len(left_tokens | right_tokens))


def _lancedb_filter(
    vocabularies: Sequence[str],
    domain_hints: Sequence[str],
) -> str:
    clauses: list[str] = []
    if vocabularies:
        quoted = ", ".join(f"'{_sql_escape(v)}'" for v in vocabularies)
        clauses.append(f"vocabulary_id IN ({quoted})")
    if domain_hints:
        quoted = ", ".join(f"'{_sql_escape(d)}'" for d in domain_hints)
        clauses.append(f"domain_id IN ({quoted})")
    return " AND ".join(clauses)


def _sql_escape(value: str) -> str:
    return value.replace("'", "''")


def _rrf_merge(
    lexical_rows: Sequence[ConceptCandidate],
    vector_rows: Sequence[ConceptCandidate],
    limit: int,
    *,
    k: int = 60,
) -> list[ConceptCandidate]:
    by_id: dict[str, ConceptCandidate] = {}
    scores: dict[str, float] = {}
    sources: dict[str, dict[str, float]] = {}

    for source_name, rows in (("fts", lexical_rows), ("vector", vector_rows)):
        for rank, row in enumerate(rows, start=1):
            # Dedup on concept_id: normalized_id collapses all CUI-less rows onto one key.
            key = row.concept_id
            by_id.setdefault(key, row)
            score = 1.0 / (k + rank)
            scores[key] = scores.get(key, 0.0) + score
            sources.setdefault(key, {})[source_name] = score

    if not by_id:
        return []

    max_score = max(scores.values()) or 1.0
    merged = [
        replace(
            row,
            score=min(1.0, scores[key] / max_score),
            source_scores=sources.get(key, {}),
        )
        for key, row in by_id.items()
    ]
    merged.sort(key=lambda item: (-item.score, item.concept_name))
    return merged[:limit]
