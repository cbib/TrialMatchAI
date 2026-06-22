from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

from trialmatchai.utils.logging_config import setup_logging

logger = setup_logging(__name__)


TRIAL_TEXT_WEIGHTS: tuple[tuple[str, float], ...] = (
    ("condition", 6.0),
    ("eligibility_criteria", 4.0),
    ("brief_title", 3.0),
    ("brief_summary", 2.0),
    ("detailed_description", 1.5),
    ("official_title", 1.0),
)
TRIAL_VECTOR_WEIGHTS: tuple[tuple[str, float], ...] = (
    ("condition_vector", 0.3),
    ("brief_title_vector", 0.1),
    ("brief_summary_vector", 0.1),
    ("eligibility_criteria_vector", 0.3),
)
CRITERIA_TEXT_WEIGHTS: tuple[tuple[str, float], ...] = (
    ("criterion", 1.0),
    ("entity_synonyms_text", 1.2),
    ("entity_text", 0.8),
)


@dataclass(frozen=True)
class SearchHit:
    source: dict[str, Any]
    score: float

    def to_es_like_hit(self) -> dict[str, Any]:
        return {"_source": self.source, "_score": self.score}


class SearchBackendUnavailable(RuntimeError):
    pass


class TrialSearchBackend(Protocol):
    def search_trials(
        self,
        *,
        primary_terms: Sequence[str],
        other_terms: Sequence[str] = (),
        embeddings: Mapping[str, Sequence[float]] | None = None,
        age: int | None = None,
        sex: str = "ALL",
        overall_status: str | None = None,
        pre_selected_nct_ids: Sequence[str] | None = None,
        size: int = 10,
        vector_score_threshold: float = 0.0,
        search_mode: str = "hybrid",
    ) -> tuple[list[dict[str, Any]], list[float]]:
        ...

    def search_criteria(
        self,
        *,
        query: str,
        nct_ids: Sequence[str],
        query_vector: Sequence[float] | None = None,
        size: int = 250,
        search_mode: str = "hybrid",
        use_entity_synonyms: bool = True,
        vector_score_threshold: float = 0.5,
    ) -> list[dict[str, Any]]:
        ...


class InMemorySearchBackend:
    """Deterministic backend used for tests and small fixture smoke runs."""

    def __init__(
        self,
        *,
        trials: Sequence[Mapping[str, Any]] = (),
        criteria: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        self.trials = [build_trial_record(row) for row in trials]
        self.criteria = [build_criteria_record(row) for row in criteria]

    def health(self, *, require_tables: bool = False) -> list[str]:
        return []

    def upsert_trials(self, docs: Sequence[Mapping[str, Any]]) -> int:
        rows = [build_trial_record(row) for row in docs]
        updated_ids = {str(row.get("nct_id")) for row in rows if row.get("nct_id")}
        self.trials = [
            row for row in self.trials if str(row.get("nct_id")) not in updated_ids
        ]
        self.trials.extend(rows)
        return len(rows)

    def replace_criteria_for_trials(
        self,
        nct_ids: Sequence[str],
        docs: Sequence[Mapping[str, Any]],
    ) -> int:
        updated_ids = {str(nct_id) for nct_id in nct_ids if nct_id}
        self.criteria = [
            row for row in self.criteria if str(row.get("nct_id")) not in updated_ids
        ]
        rows = [build_criteria_record(row) for row in docs]
        self.criteria.extend(rows)
        return len(rows)

    def search_trials(
        self,
        *,
        primary_terms: Sequence[str],
        other_terms: Sequence[str] = (),
        embeddings: Mapping[str, Sequence[float]] | None = None,
        age: int | None = None,
        sex: str = "ALL",
        overall_status: str | None = None,
        pre_selected_nct_ids: Sequence[str] | None = None,
        size: int = 10,
        vector_score_threshold: float = 0.0,
        search_mode: str = "hybrid",
    ) -> tuple[list[dict[str, Any]], list[float]]:
        hits = _rank_trial_rows(
            self.trials,
            primary_terms=primary_terms,
            other_terms=other_terms,
            embeddings=embeddings or {},
            age=age,
            sex=sex,
            overall_status=overall_status,
            pre_selected_nct_ids=pre_selected_nct_ids,
            size=size,
            vector_score_threshold=vector_score_threshold,
            search_mode=search_mode,
        )
        return [hit.source for hit in hits], [hit.score for hit in hits]

    def search_criteria(
        self,
        *,
        query: str,
        nct_ids: Sequence[str],
        query_vector: Sequence[float] | None = None,
        size: int = 250,
        search_mode: str = "hybrid",
        use_entity_synonyms: bool = True,
        vector_score_threshold: float = 0.5,
    ) -> list[dict[str, Any]]:
        hits = _rank_criteria_rows(
            self.criteria,
            query=query,
            nct_ids=nct_ids,
            query_vector=query_vector,
            size=size,
            search_mode=search_mode,
            use_entity_synonyms=use_entity_synonyms,
            vector_score_threshold=vector_score_threshold,
        )
        return [hit.to_es_like_hit() for hit in hits]


class LanceDBSearchBackend:
    """Embedded LanceDB backend for TrialMatchAI trial and criteria retrieval."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        trials_table: str = "trials",
        criteria_table: str = "criteria",
        candidate_limit: int = 1000,
    ) -> None:
        try:
            import lancedb  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency guard
            raise SearchBackendUnavailable(
                "LanceDB search requires the runtime dependency `lancedb`."
            ) from exc

        self.db_path = Path(db_path)
        self.trials_table = trials_table
        self.criteria_table = criteria_table
        self.candidate_limit = candidate_limit
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(self.db_path))

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "LanceDBSearchBackend":
        search_cfg = config.get("search_backend", {})
        backend = search_cfg.get("backend", "lancedb")
        if backend != "lancedb":
            raise ValueError(f"Unsupported search backend: {backend}")
        return cls(
            search_cfg.get("db_path", "data/search"),
            trials_table=search_cfg.get("trials_table", "trials"),
            criteria_table=search_cfg.get("criteria_table", "criteria"),
            candidate_limit=int(search_cfg.get("candidate_limit", 1000)),
        )

    def health(self, *, require_tables: bool = False) -> list[str]:
        issues: list[str] = []
        if not self.db_path.exists():
            issues.append(f"search_backend.db_path does not exist: {self.db_path}")
            return issues
        if require_tables:
            names = set(self._table_names())
            missing = [
                name
                for name in (self.trials_table, self.criteria_table)
                if name not in names
            ]
            if missing:
                issues.append("Missing LanceDB tables: " + ", ".join(missing))
        return issues

    def table_exists(self, table_name: str) -> bool:
        return table_name in set(self._table_names())

    def index_trials(
        self,
        docs: Sequence[Mapping[str, Any]],
        *,
        recreate: bool = True,
    ) -> int:
        rows = [build_trial_record(doc) for doc in docs]
        table = self._write_rows(self.trials_table, rows, recreate=recreate)
        _create_fts_index(table, "search_text")
        _create_vector_index(table, "search_vector")
        return len(rows)

    def index_criteria(
        self,
        docs: Sequence[Mapping[str, Any]],
        *,
        recreate: bool = True,
    ) -> int:
        rows = [build_criteria_record(doc) for doc in docs]
        table = self._write_rows(self.criteria_table, rows, recreate=recreate)
        _create_fts_index(table, "search_text")
        _create_vector_index(table, "criterion_vector")
        return len(rows)

    def upsert_trials(self, docs: Sequence[Mapping[str, Any]]) -> int:
        rows = [build_trial_record(doc) for doc in docs]
        if not rows:
            return 0
        if self.table_exists(self.trials_table):
            nct_ids = [str(row["nct_id"]) for row in rows if row.get("nct_id")]
            if nct_ids:
                self._delete_where(self.trials_table, _nct_where(nct_ids))
        table = self._write_rows(self.trials_table, rows, recreate=False)
        _create_fts_index(table, "search_text")
        _create_vector_index(table, "search_vector")
        return len(rows)

    def replace_criteria_for_trials(
        self,
        nct_ids: Sequence[str],
        docs: Sequence[Mapping[str, Any]],
    ) -> int:
        rows = [build_criteria_record(doc) for doc in docs]
        if self.table_exists(self.criteria_table):
            where = _nct_where(nct_ids)
            if where:
                self._delete_where(self.criteria_table, where)
        if not rows:
            return 0
        table = self._write_rows(self.criteria_table, rows, recreate=False)
        _create_fts_index(table, "search_text")
        _create_vector_index(table, "criterion_vector")
        return len(rows)

    def search_trials(
        self,
        *,
        primary_terms: Sequence[str],
        other_terms: Sequence[str] = (),
        embeddings: Mapping[str, Sequence[float]] | None = None,
        age: int | None = None,
        sex: str = "ALL",
        overall_status: str | None = None,
        pre_selected_nct_ids: Sequence[str] | None = None,
        size: int = 10,
        vector_score_threshold: float = 0.0,
        search_mode: str = "hybrid",
    ) -> tuple[list[dict[str, Any]], list[float]]:
        table = self._open_table(self.trials_table)
        where = _nct_where(pre_selected_nct_ids)
        vector = _mean_vectors(
            [
                embeddings[term]
                for term in primary_terms
                if embeddings and term in embeddings
            ]
        )
        rows = self._candidate_rows(
            table,
            text_query=" ".join([*primary_terms, *other_terms]),
            query_vector=vector,
            vector_column="search_vector",
            where=where,
            mode=search_mode,
            limit=max(size, self.candidate_limit),
        )
        hits = _rank_trial_rows(
            rows,
            primary_terms=primary_terms,
            other_terms=other_terms,
            embeddings=embeddings or {},
            age=age,
            sex=sex,
            overall_status=overall_status,
            pre_selected_nct_ids=pre_selected_nct_ids,
            size=size,
            vector_score_threshold=vector_score_threshold,
            search_mode=search_mode,
        )
        return [hit.source for hit in hits], [hit.score for hit in hits]

    def search_criteria(
        self,
        *,
        query: str,
        nct_ids: Sequence[str],
        query_vector: Sequence[float] | None = None,
        size: int = 250,
        search_mode: str = "hybrid",
        use_entity_synonyms: bool = True,
        vector_score_threshold: float = 0.5,
    ) -> list[dict[str, Any]]:
        table = self._open_table(self.criteria_table)
        where = _nct_where(nct_ids)
        rows = self._candidate_rows(
            table,
            text_query=query,
            query_vector=query_vector,
            vector_column="criterion_vector",
            where=where,
            mode=search_mode,
            limit=max(size, self.candidate_limit),
        )
        hits = _rank_criteria_rows(
            rows,
            query=query,
            nct_ids=nct_ids,
            query_vector=query_vector,
            size=size,
            search_mode=search_mode,
            use_entity_synonyms=use_entity_synonyms,
            vector_score_threshold=vector_score_threshold,
        )
        return [hit.to_es_like_hit() for hit in hits]

    def _write_rows(
        self,
        table_name: str,
        rows: Sequence[dict[str, Any]],
        *,
        recreate: bool,
    ) -> Any:
        if not rows:
            raise ValueError(f"No rows supplied for LanceDB table {table_name}.")
        if recreate or not self.table_exists(table_name):
            return self.db.create_table(table_name, data=list(rows), mode="overwrite")
        table = self._open_table(table_name)
        table.add(list(rows))
        return table

    def _delete_where(self, table_name: str, where: str) -> None:
        if not where:
            return
        try:
            table = self._open_table(table_name)
            table.delete(where)
        except Exception as exc:
            raise SearchBackendUnavailable(
                f"Could not delete existing rows from LanceDB table {table_name}: {exc}"
            ) from exc

    def _candidate_rows(
        self,
        table: Any,
        *,
        text_query: str,
        query_vector: Sequence[float] | None,
        vector_column: str,
        where: str,
        mode: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        mode = (mode or "hybrid").lower()
        rows_by_key: dict[str, dict[str, Any]] = {}

        if mode in {"bm25", "hybrid"} and text_query.strip():
            for row in self._search_fts(table, text_query, where=where, limit=limit):
                rows_by_key[_row_key(row)] = row

        if mode in {"vector", "hybrid"} and query_vector:
            for row in self._search_vector(
                table,
                query_vector,
                vector_column=vector_column,
                where=where,
                limit=limit,
            ):
                rows_by_key[_row_key(row)] = row

        if not rows_by_key:
            for row in self._scan_rows(table, where=where, limit=limit):
                rows_by_key[_row_key(row)] = row

        return list(rows_by_key.values())

    def _search_fts(
        self,
        table: Any,
        query: str,
        *,
        where: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        try:
            search = table.search(query, query_type="fts")
            if where:
                search = search.where(where)
            return list(search.limit(limit).to_list())
        except Exception as exc:
            logger.warning("LanceDB FTS search failed; falling back if possible: %s", exc)
            return []

    def _search_vector(
        self,
        table: Any,
        vector: Sequence[float],
        *,
        vector_column: str,
        where: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        try:
            search = table.search(list(vector), vector_column_name=vector_column)
            if where:
                search = search.where(where)
            return list(search.limit(limit).to_list())
        except Exception as exc:
            logger.warning(
                "LanceDB vector search failed; falling back if possible: %s", exc
            )
            return []

    def _scan_rows(
        self, table: Any, *, where: str, limit: int
    ) -> list[dict[str, Any]]:
        try:
            # Honor the nct_id filter on the fallback path; an unfiltered head
            # slice could return rows that exclude the requested trials entirely.
            if where:
                return list(table.search().where(where).limit(limit).to_list())
            return list(table.to_arrow().to_pylist())[:limit]
        except Exception as exc:
            logger.warning("Could not scan LanceDB table rows: %s", exc)
            return []

    def _open_table(self, table_name: str) -> Any:
        try:
            return self.db.open_table(table_name)
        except Exception as exc:
            raise SearchBackendUnavailable(
                f"LanceDB table is not available: {table_name}"
            ) from exc

    def _table_names(self) -> list[str]:
        try:
            return list(self.db.table_names())
        except Exception as exc:
            logger.warning("Could not list LanceDB tables: %s", exc)
            return []


def build_trial_record(doc: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(doc)
    search_text = _flatten_text(
        [
            row.get("condition"),
            row.get("eligibility_criteria"),
            row.get("brief_title"),
            row.get("brief_summary"),
            row.get("detailed_description"),
            row.get("official_title"),
        ]
    )
    row["search_text"] = search_text
    row["search_vector"] = _mean_vectors(
        _clean_vector(row.get(field)) for field, _ in TRIAL_VECTOR_WEIGHTS
    )
    return row


def build_criteria_record(doc: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(doc)
    entity_text, synonym_text = _flatten_entities(row.get("entities"))
    row["entity_text"] = entity_text
    row["entity_synonyms_text"] = synonym_text
    row["search_text"] = _flatten_text(
        [row.get("criterion"), entity_text, synonym_text]
    )
    return row


def _rank_trial_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    primary_terms: Sequence[str],
    other_terms: Sequence[str],
    embeddings: Mapping[str, Sequence[float]],
    age: int | None,
    sex: str,
    overall_status: str | None,
    pre_selected_nct_ids: Sequence[str] | None,
    size: int,
    vector_score_threshold: float,
    search_mode: str,
) -> list[SearchHit]:
    mode = (search_mode or "hybrid").lower()
    hits: list[SearchHit] = []
    for raw in rows:
        row = build_trial_record(raw)
        if not _trial_passes_filters(
            row,
            age=age,
            sex=sex,
            overall_status=overall_status,
            pre_selected_nct_ids=pre_selected_nct_ids,
        ):
            continue
        text_score = _weighted_text_score(row, primary_terms, TRIAL_TEXT_WEIGHTS)
        if other_terms:
            text_score = max(
                text_score,
                0.75 * _weighted_text_score(row, other_terms, TRIAL_TEXT_WEIGHTS),
            )
        vector_score = _trial_vector_score(row, primary_terms, other_terms, embeddings)
        if mode in {"vector", "hybrid"} and vector_score < vector_score_threshold:
            continue
        score = _combine_scores(mode, text_score, vector_score)
        if score <= 0:
            continue
        hits.append(SearchHit(source=row, score=score))
    hits.sort(key=lambda hit: hit.score, reverse=True)
    return hits[:size]


def _rank_criteria_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    query: str,
    nct_ids: Sequence[str],
    query_vector: Sequence[float] | None,
    size: int,
    search_mode: str,
    use_entity_synonyms: bool,
    vector_score_threshold: float,
) -> list[SearchHit]:
    allowed = {nct_id for nct_id in nct_ids if nct_id}
    mode = (search_mode or "hybrid").lower()
    fields = (
        CRITERIA_TEXT_WEIGHTS
        if use_entity_synonyms
        else tuple(item for item in CRITERIA_TEXT_WEIGHTS if item[0] == "criterion")
    )
    hits: list[SearchHit] = []
    for raw in rows:
        row = build_criteria_record(raw)
        if allowed and row.get("nct_id") not in allowed:
            continue
        text_score = _weighted_text_score(row, [query], fields)
        vector_score = _vector_score(query_vector, _clean_vector(row.get("criterion_vector")))
        if mode in {"vector", "hybrid"} and vector_score < vector_score_threshold:
            continue
        score = _combine_scores(mode, text_score, vector_score)
        if score <= 0:
            continue
        hits.append(SearchHit(source=row, score=score))
    hits.sort(key=lambda hit: hit.score, reverse=True)
    return hits[:size]


def _combine_scores(mode: str, text_score: float, vector_score: float) -> float:
    if mode == "bm25":
        return text_score
    if mode == "vector":
        return vector_score
    return 0.5 * text_score + 0.5 * vector_score


def _weighted_text_score(
    row: Mapping[str, Any],
    terms: Sequence[str],
    fields: Sequence[tuple[str, float]],
) -> float:
    clean_terms = [term for term in terms if term and term.strip()]
    if not clean_terms:
        return 0.0
    max_weight = sum(weight for _, weight in fields) or 1.0
    best = 0.0
    for term in clean_terms:
        field_score = 0.0
        for field, weight in fields:
            field_score += weight * _lexical_score(term, _flatten_text(row.get(field)))
        best = max(best, field_score / max_weight)
    return min(best, 1.0)


def _trial_vector_score(
    row: Mapping[str, Any],
    primary_terms: Sequence[str],
    other_terms: Sequence[str],
    embeddings: Mapping[str, Sequence[float]],
) -> float:
    primary_vectors = [
        embeddings[term] for term in primary_terms if term in embeddings and embeddings[term]
    ]
    other_vectors = [
        embeddings[term] for term in other_terms if term in embeddings and embeddings[term]
    ]
    score = 0.0
    weight_total = 0.0
    for field, weight in TRIAL_VECTOR_WEIGHTS:
        field_vector = _clean_vector(row.get(field))
        if not field_vector:
            continue
        score += weight * _max_vector_score(primary_vectors, field_vector)
        weight_total += weight
    if other_vectors:
        other_field_scores: list[float] = []
        for field, _ in TRIAL_VECTOR_WEIGHTS:
            field_vector = _clean_vector(row.get(field))
            if field_vector:
                other_field_scores.append(_max_vector_score(other_vectors, field_vector))
        if other_field_scores:
            score += 0.2 * (sum(other_field_scores) / len(other_field_scores))
            weight_total += 0.2
    if weight_total == 0:
        return 0.0
    return min(score / weight_total, 1.0)


def _max_vector_score(
    query_vectors: Sequence[Sequence[float]],
    field_vector: Sequence[float],
) -> float:
    if not query_vectors or not field_vector:
        return 0.0
    return max(_vector_score(vector, field_vector) for vector in query_vectors)


def _vector_score(
    left: Sequence[float] | None,
    right: Sequence[float] | None,
) -> float:
    if not left or not right:
        return 0.0
    similarity = _cosine(left, right)
    return max(0.0, min(1.0, (similarity + 1.0) / 2.0))


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    size = min(len(left), len(right))
    if size == 0:
        return 0.0
    dot = sum(float(left[i]) * float(right[i]) for i in range(size))
    left_norm = math.sqrt(sum(float(left[i]) ** 2 for i in range(size)))
    right_norm = math.sqrt(sum(float(right[i]) ** 2 for i in range(size)))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _lexical_score(query: str, text: str) -> float:
    query_norm = _normalize_text(query)
    text_norm = _normalize_text(text)
    if not query_norm or not text_norm:
        return 0.0
    if query_norm == text_norm:
        return 1.0
    if query_norm in text_norm:
        return 0.95
    query_tokens = set(query_norm.split())
    text_tokens = set(text_norm.split())
    if not query_tokens or not text_tokens:
        return 0.0
    overlap = len(query_tokens & text_tokens)
    if overlap == 0:
        return 0.0
    coverage = overlap / len(query_tokens)
    jaccard = overlap / len(query_tokens | text_tokens)
    return min(1.0, 0.75 * coverage + 0.25 * jaccard)


def _trial_passes_filters(
    row: Mapping[str, Any],
    *,
    age: int | None,
    sex: str,
    overall_status: str | None,
    pre_selected_nct_ids: Sequence[str] | None,
) -> bool:
    if pre_selected_nct_ids and row.get("nct_id") not in set(pre_selected_nct_ids):
        return False
    if overall_status and overall_status.casefold() != "all":
        status = str(row.get("overall_status") or "")
        if status.casefold() != overall_status.casefold():
            return False
    if age is not None:
        minimum = _as_float(row.get("minimum_age"))
        maximum = _as_float(row.get("maximum_age"))
        if minimum is not None and minimum > age:
            return False
        if maximum is not None and maximum < age:
            return False
    return _gender_matches(row.get("gender"), sex)


def _gender_matches(value: Any, sex: str) -> bool:
    requested = (sex or "ALL").casefold()
    raw = str(value or "All").casefold()
    if requested in {"all", "both"}:
        return True
    if raw in {"", "all", "both"}:
        return True
    if requested == "male":
        return raw in {"male", "m"}
    if requested == "female":
        return raw in {"female", "f"}
    return True


def _flatten_entities(entities: Any) -> tuple[str, str]:
    if not isinstance(entities, list):
        return "", ""
    texts: list[str] = []
    synonyms: list[str] = []
    for entity in entities:
        if not isinstance(entity, Mapping):
            continue
        texts.append(_flatten_text([entity.get("text"), entity.get("entity")]))
        synonyms.append(_flatten_text(entity.get("synonyms")))
        for candidate in entity.get("concept_candidates") or []:
            if isinstance(candidate, Mapping):
                synonyms.append(_flatten_text(candidate.get("concept_name")))
    return _flatten_text(texts), _flatten_text(synonyms)


def _flatten_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, Mapping):
        return " ".join(_flatten_text(item) for item in value.values()).strip()
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        return " ".join(_flatten_text(item) for item in value).strip()
    return str(value)


def _normalize_text(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _clean_vector(value: Any) -> list[float]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return []


def _mean_vectors(vectors: Iterable[Sequence[float] | None]) -> list[float]:
    cleaned = [_clean_vector(vector) for vector in vectors]
    cleaned = [vector for vector in cleaned if vector]
    if not cleaned:
        return []
    width = min(len(vector) for vector in cleaned)
    if width == 0:
        return []
    return [
        sum(vector[index] for vector in cleaned) / len(cleaned)
        for index in range(width)
    ]


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _nct_where(nct_ids: Sequence[str] | None) -> str:
    values = [str(item) for item in (nct_ids or []) if item]
    if not values:
        return ""
    quoted = ", ".join(f"'{_sql_escape(value)}'" for value in values)
    return f"nct_id IN ({quoted})"


def _sql_escape(value: str) -> str:
    return value.replace("'", "''")


def _row_key(row: Mapping[str, Any]) -> str:
    return str(row.get("criteria_id") or row.get("nct_id") or id(row))


def _create_fts_index(table: Any, column: str) -> None:
    try:
        table.create_fts_index(column, replace=True)
    except TypeError:
        try:
            table.create_fts_index(column)
        except Exception as exc:
            logger.warning("Could not create LanceDB FTS index on %s: %s", column, exc)
    except Exception as exc:
        logger.warning("Could not create LanceDB FTS index on %s: %s", column, exc)


def _create_vector_index(table: Any, column: str) -> None:
    try:
        table.create_index(vector_column_name=column, metric="cosine")
    except TypeError:
        try:
            table.create_index(column)
        except Exception as exc:
            logger.warning("Could not create LanceDB vector index on %s: %s", column, exc)
    except Exception as exc:
        logger.warning("Could not create LanceDB vector index on %s: %s", column, exc)
