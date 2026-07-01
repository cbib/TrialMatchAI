from __future__ import annotations

import json
import math
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional

from trialmatchai.constraints import apply_constraint_score, evaluate_constraint_set
from trialmatchai.constraints.models import ConstraintSet, CriterionConstraintEvaluation
from trialmatchai.matching.retrieval.synonyms import disease_synonyms
from trialmatchai.search.lancedb_backend import TrialSearchBackend
from trialmatchai.utils.file_utils import write_text_file
from trialmatchai.utils.logging_config import setup_logging

if TYPE_CHECKING:
    from trialmatchai.constraints.models import PatientConstraintContext
    from trialmatchai.models.embedding.text_embedder import TextEmbedder
    from trialmatchai.models.llm.llm_reranker import LLMReranker

logger = setup_logging(__name__)


class SecondStageRetriever:
    def __init__(
        self,
        search_backend: TrialSearchBackend,
        llm_reranker: Optional[LLMReranker],  # Make optional
        embedder: Optional[TextEmbedder],
        size: int = 250,
        inclusion_weight: float = 1.0,
        exclusion_weight: float = 0.25,
        entity_annotator=None,
        search_mode: str = "hybrid",
    ):
        self.search_backend = search_backend
        self.llm_reranker = llm_reranker  # Can be None
        self.embedder = embedder
        self.size = size
        self.inclusion_weight = inclusion_weight
        self.exclusion_weight = exclusion_weight
        self.entity_annotator = entity_annotator
        self.search_mode = search_mode.lower() if search_mode else "hybrid"
        self.last_constraint_evaluations: list[CriterionConstraintEvaluation] = []

    def get_synonyms(self, condition: str) -> List[str]:
        return disease_synonyms(self.entity_annotator, condition)

    def retrieve_criteria(
        self, nct_ids: List[str], queries: List[str]
    ) -> Dict[str, List[Dict]]:
        query_to_hits = {}

        def execute_query(query):
            mode = self.search_mode
            query_vector = None
            if mode in {"vector", "hybrid"}:
                if self.embedder is None:
                    logger.warning(
                        "%s mode selected but embedder is None. Falling back to BM25.",
                        mode,
                    )
                    mode = "bm25"
                else:
                    vectors = self.embedder.embed_texts([query])
                    if vectors:
                        query_vector = vectors[0]
                    else:
                        logger.warning("Empty query after preprocessing. Falling back to BM25.")
                        mode = "bm25"

            try:
                hits = self.search_backend.search_criteria(
                    query=query,
                    nct_ids=nct_ids,
                    query_vector=query_vector,
                    size=self.size,
                    search_mode=mode,
                    use_entity_synonyms=self.entity_annotator is not None,
                )
                logger.info(
                    "[%s] Retrieved %s documents for query: '%s'",
                    mode,
                    len(hits),
                    query,
                )
                return query, hits
            except Exception:
                logger.exception("Second-level search failed for query: %s", query)
                return query, []

        if not queries:
            return {}

        with ThreadPoolExecutor(max_workers=min(8, len(queries))) as executor:
            future_to_query = {
                executor.submit(execute_query, query): query for query in queries
            }
            for future in as_completed(future_to_query):
                query, hits = future.result()
                query_to_hits[query] = hits
        return query_to_hits

    def rerank_criteria(self, criteria: List[Dict]) -> List[Dict]:
        if self.llm_reranker is None:
            logger.warning("LLM reranker not available, using search scores only")
            return self.score_criteria_without_llm(criteria)

        pairs = [
            (criterion["query"], criterion["_source"]["criterion"])
            for criterion in criteria
        ]
        llm_scores = self.llm_reranker.rank_pairs(pairs)
        llm_scores = [
            score.get("llm_score", 0.0) if isinstance(score, dict) else float(score)
            for score in llm_scores
        ]
        if len(llm_scores) != len(pairs):
            logger.error("Mismatch between LLM scores and pairs!")
            raise ValueError("Mismatch between LLM scores and pairs!")
        for i, criterion in enumerate(criteria):
            llm_score = llm_scores[i]
            # stored eligibility_type is the short form "inclusion"/"exclusion"/"unknown"
            # (criteria_chunking); "unknown" falls through at weight 1.0, matching the ES-era reranker.
            eligibility_type = criterion["_source"].get("eligibility_type", "").lower()
            if eligibility_type == "inclusion":
                llm_score *= self.inclusion_weight
            elif eligibility_type == "exclusion":
                llm_score *= self.exclusion_weight
            criterion["llm_score"] = llm_score
        return criteria

    def score_criteria_without_llm(self, criteria: List[Dict]) -> List[Dict]:
        if not criteria:
            return criteria
        max_es = max((c.get("_score", 0.0) for c in criteria), default=1.0) or 1.0
        for criterion in criteria:
            base = float(criterion.get("_score", 0.0)) / max_es
            eligibility_type = criterion["_source"].get("eligibility_type", "").lower()
            if eligibility_type == "inclusion":
                base *= self.inclusion_weight
            elif eligibility_type == "exclusion":
                base *= self.exclusion_weight
            criterion["llm_score"] = base
        return criteria

    def apply_constraint_adjustments(
        self,
        criteria: List[Dict],
        *,
        patient_context: Optional["PatientConstraintContext"] = None,
        constraints_config: Mapping[str, Any] | None = None,
    ) -> List[Dict]:
        self.last_constraint_evaluations = []
        config = constraints_config or {}
        if not config.get("enabled", True) or patient_context is None:
            return criteria

        score_weight = _constraint_score_weight(config.get("score_weight", 0.25))
        unknown_is_neutral = bool(config.get("unknown_is_neutral", True))
        seen_criteria: set[str] = set()
        for criterion in criteria:
            source = criterion.get("_source") or {}
            raw_constraints = source.get("constraints")
            if not raw_constraints:
                continue
            try:
                constraint_set = _constraint_set_from_payload(raw_constraints)
            except Exception:
                logger.warning(
                    "Ignoring invalid constraint payload for criterion %s.",
                    source.get("criteria_id", "unknown"),
                )
                continue
            if not constraint_set.constraints:
                continue

            evaluation = evaluate_constraint_set(
                constraint_set, patient_context, unknown_is_neutral=unknown_is_neutral
            )
            if evaluation.criteria_id not in seen_criteria:
                self.last_constraint_evaluations.append(evaluation)
                seen_criteria.add(evaluation.criteria_id)

            base_score = float(
                criterion.get("llm_score", criterion.get("_score", 0.0)) or 0.0
            )
            adjusted_score = apply_constraint_score(
                base_score,
                evaluation.constraint_signal,
                score_weight=score_weight,
            )
            criterion["pre_constraint_score"] = base_score
            criterion["constraint_signal"] = evaluation.constraint_signal
            criterion["constraint_adjusted_score"] = adjusted_score
            criterion["constraint_evaluation"] = evaluation.model_dump(mode="json")
            criterion["llm_score"] = adjusted_score
        return criteria

    def aggregate_to_trials(
        self, criteria: List[Dict], threshold: float = 0.5, method: str = "weighted"
    ) -> List[Dict]:
        trial_scores = defaultdict(list)
        for criterion in criteria:
            nct_id = criterion["_source"]["nct_id"]
            score = criterion["llm_score"]
            # Keep a criterion that was relevant BEFORE constraint penalties even if the
            # penalty pushed it below threshold: constraints penalize (downweight) but do not
            # hard-exclude at retrieval — the CoT stage makes the final eligibility call.
            threshold_basis = criterion.get("pre_constraint_score", score)
            if score >= threshold or threshold_basis >= threshold:
                trial_scores[nct_id].append(score)
        aggregated_scores = {}
        for nct_id, scores in trial_scores.items():
            count = len(scores)
            total = sum(scores)
            if count == 0:
                continue
            if method == "avg":
                agg_score = total / count
            elif method == "sqrt":
                agg_score = total / math.sqrt(count)
            elif method == "log":
                agg_score = total / math.log(count + 1)
            elif method == "weighted":
                agg_score = 0.7 * (total / math.sqrt(count)) + 0.3 * max(scores)
            else:
                raise ValueError(f"Unsupported aggregation method: {method}")
            aggregated_scores[nct_id] = agg_score
        sorted_trials = [
            {"nct_id": nct_id, "score": score}
            for nct_id, score in sorted(
                aggregated_scores.items(), key=lambda x: x[1], reverse=True
            )
        ]
        return sorted_trials

    def retrieve_and_rank(
        self,
        queries: List[str],
        nct_ids: List[str],
        top_n: int,
        use_reranker: bool = True,
        save_path: Optional[str] = None,
        patient_context: Optional["PatientConstraintContext"] = None,
        constraints_config: Mapping[str, Any] | None = None,
    ) -> List[Dict]:
        # Cap queries to prevent memory/performance issues
        max_queries = 150  # Reasonable limit for second-level search
        if len(queries) > max_queries:
            logger.warning(
                f"Capping queries from {len(queries)} to {max_queries} for second-level search"
            )
            queries = queries[:max_queries]

        query_to_hits = self.retrieve_criteria(nct_ids, queries)
        all_criteria = []
        for query, hits in query_to_hits.items():
            for hit in hits:
                hit["query"] = query
                all_criteria.append(hit)

        # Check if reranker is available before trying to use it
        if use_reranker and self.llm_reranker is not None:
            ranked_criteria = self.rerank_criteria(all_criteria)
        else:
            if use_reranker and self.llm_reranker is None:
                logger.info(
                    "Reranking requested but LLM reranker not available; using search scores for aggregation."
                )
            else:
                logger.info(
                    "Second-level reranking disabled; using search scores for aggregation."
                )
            ranked_criteria = self.score_criteria_without_llm(all_criteria)

        ranked_criteria = self.apply_constraint_adjustments(
            ranked_criteria,
            patient_context=patient_context,
            constraints_config=constraints_config,
        )
        sorted_trials = self.aggregate_to_trials(ranked_criteria)
        top_trials = sorted_trials[:top_n]
        if len(sorted_trials) < top_n:
            logger.warning(
                "Second-level aggregation produced only %s trials (fewer than the requested top %s).",
                len(sorted_trials),
                top_n,
            )
        else:
            logger.info("Aggregated %s trials; returning top %s.", len(sorted_trials), top_n)
        if save_path:
            write_text_file([trial["nct_id"] for trial in top_trials], save_path)
            logger.info(f"Top trials saved to {save_path}")
        return top_trials


def _constraint_score_weight(value: Any) -> float:
    try:
        weight = float(value)
    except (TypeError, ValueError):
        return 0.25
    return max(0.0, min(1.0, weight))


def _constraint_set_from_payload(payload: Any) -> ConstraintSet:
    if isinstance(payload, ConstraintSet):
        return payload
    if isinstance(payload, str):
        payload = json.loads(payload)
    return ConstraintSet.model_validate(payload)
