from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

from dateutil import parser as date_parser
from Matcher.models.embedding.text_embedder import TextEmbedder
from Matcher.utils.logging_config import setup_logging
from Matcher.utils.retry import with_retries

from elasticsearch import Elasticsearch

logger = setup_logging(__name__)


class ClinicalTrialSearch:
    def __init__(
        self,
        es_client: Elasticsearch,
        embedder: Optional[TextEmbedder],
        index_name: str,
        bio_med_ner=None,
        entity_annotator=None,
    ):
        self.es_client = es_client
        self.embedder = embedder
        self.index_name = index_name
        self.entity_annotator = entity_annotator or bio_med_ner

    def get_synonyms(self, condition: str) -> List[str]:
        if not self.entity_annotator:
            logger.info("Entity annotator disabled; skipping synonyms extraction.")
            return []
        try:
            raw_result = self.entity_annotator.annotate_texts_in_parallel(
                [condition], max_workers=1
            )
            ner_results = raw_result
            if ner_results and ner_results[0]:
                synonyms = set()
                for entity in ner_results[0]:
                    if entity.get("entity_group", "").lower() == "disease":
                        synonyms.update(entity.get("synonyms", []))
                return list(synonyms)
            logger.warning(f"No annotations found for condition: {condition}")
        except Exception as e:
            logger.error(f"Entity synonym extraction failed for '{condition}': {e}")
        return []

    def parse_age_input(self, age_input: Union[int, str]) -> Optional[int]:
        if isinstance(age_input, int):
            return age_input
        elif isinstance(age_input, str):
            age_input = age_input.strip().lower()
            age_keywords = ["year", "years", "yr", "yrs"]
            try:
                for keyword in age_keywords:
                    if age_input.endswith(keyword):
                        age_str = age_input.replace(keyword, "").strip()
                        return int(age_str)
                return int(age_input)
            except ValueError:
                pass
            try:
                dob = date_parser.parse(age_input, fuzzy=True)
                today = datetime.today()
                age = (
                    today.year
                    - dob.year
                    - ((today.month, today.day) < (dob.month, dob.day))
                )
                if age < 0:
                    raise ValueError("Invalid date of birth.")
                return age
            except (ValueError, OverflowError, date_parser.ParserError):
                pass
        return None

    def get_max_text_score(self, synonyms: List[str]) -> float:
        should_clauses = [
            {
                "multi_match": {
                    "query": syn,
                    "fields": [
                        "condition^6",
                        "eligibility_criteria^4",
                        "brief_title^3",
                        "brief_summary^2",
                        "detailed_description^1.5",
                        "official_title",
                    ],
                    "type": "best_fields",
                    "operator": "and",
                }
            }
            for syn in synonyms
        ]
        try:
            response = with_retries(
                lambda: self.es_client.search(
                    index=self.index_name,
                    body={
                        "size": 1,
                        "query": {
                            "bool": {"should": should_clauses, "minimum_should_match": 0}
                        },
                        "track_total_hits": False,
                        "_source": False,
                    },
                ),
                logger=logger,
                action="ES max_text_score search",
            )
            max_score = response["hits"]["max_score"]
            return max_score if max_score else 1.0
        except Exception:
            logger.exception("Failed to compute max text score; defaulting to 1.0")
            return 1.0

    def create_query(
        self,
        synonyms: List[str],
        embeddings: Dict[str, List[float]],
        age: int,
        sex: str,
        overall_status: Optional[str],
        max_text_score: float,
        vector_score_threshold: float = 0.5,
        pre_selected_nct_ids: Optional[List[str]] = None,
        other_conditions: Optional[List[str]] = None,
        search_mode: str = "hybrid",
    ) -> Dict:
        sex = sex.upper()
        gender_terms = {
            "MALE": ["MALE", "Male", "male", "M", "All", "all", "ALL"],
            "FEMALE": ["FEMALE", "Female", "female", "F", "All", "all", "ALL"],
            "ALL": [
                "All",
                "all",
                "ALL",
                "Both",
                "both",
                "BOTH",
                "FEMALE",
                "Female",
                "female",
                "F",
                "MALE",
                "Male",
                "male",
                "M",
            ],
        }.get(sex, ["All"])
        filters = []
        if age not in [None, "all", "ALL", "All"]:
            filters.append(
                {
                    "bool": {
                        "must": [
                            {
                                "bool": {
                                    "should": [
                                        {"range": {"minimum_age": {"lte": age}}},
                                        {
                                            "bool": {
                                                "must_not": {
                                                    "exists": {"field": "minimum_age"}
                                                }
                                            }
                                        },
                                    ]
                                }
                            },
                            {
                                "bool": {
                                    "should": [
                                        {"range": {"maximum_age": {"gte": age}}},
                                        {
                                            "bool": {
                                                "must_not": {
                                                    "exists": {"field": "maximum_age"}
                                                }
                                            }
                                        },
                                    ]
                                }
                            },
                        ]
                    }
                }
            )
        if overall_status and overall_status.lower() != "all":
            filters.append({"match": {"overall_status": overall_status}})
        if pre_selected_nct_ids:
            filters.append({"terms": {"nct_id": pre_selected_nct_ids}})
        if gender_terms:
            filters.append(
                {
                    "bool": {
                        "should": [
                            {"terms": {"gender": gender_terms}},
                            {"bool": {"must_not": {"exists": {"field": "gender"}}}},
                        ]
                    }
                }
            )

        # Cap conditions to prevent too many ES clauses (each condition creates 2 clauses)
        # ES default maxClauseCount is 1024, leaving room for filters and other clauses
        max_conditions_per_query = 800  # Conservative limit
        all_conditions = synonyms + (other_conditions or [])
        if len(all_conditions) > max_conditions_per_query:
            logger.warning(
                f"Capping search conditions from {len(all_conditions)} to {max_conditions_per_query} to avoid ES clause limit"
            )
            # Prioritize synonyms over other_conditions
            capped_synonyms = synonyms[:max_conditions_per_query]
            remaining_slots = max_conditions_per_query - len(capped_synonyms)
            capped_other = (
                (other_conditions or [])[:remaining_slots]
                if remaining_slots > 0
                else []
            )
            synonyms = capped_synonyms
            other_conditions = capped_other

        should_clauses = []
        for condition in synonyms + (other_conditions or []):
            if condition:
                for match_type in ["best_fields", "phrase"]:
                    multi_match = {
                        "query": condition,
                        "fields": [
                            "condition^6",
                            "eligibility_criteria^4",
                            "brief_title^3",
                            "brief_summary^2",
                            "detailed_description^1.5",
                            "official_title",
                        ],
                        "type": match_type,
                    }
                    if match_type == "best_fields":
                        multi_match["operator"] = "and"
                    should_clauses.append({"multi_match": multi_match})

        logger.info(
            f"Created query with {len(should_clauses)} should clauses for {len(synonyms)} synonyms and {len(other_conditions or [])} other conditions"
        )

        search_mode = (search_mode or "hybrid").lower()
        if search_mode == "bm25":
            return {
                "bool": {
                    "should": should_clauses,
                    "minimum_should_match": 0,
                    "filter": filters,
                }
            }

        # Prepare vectors for vector/hybrid
        query_vectors = [
            embeddings[term] for term in synonyms if term in embeddings and term
        ]
        other_vectors = [
            embeddings[term]
            for term in (other_conditions or [])
            if term in embeddings and term
        ]

        if search_mode == "vector":
            return {
                "script_score": {
                    "query": {
                        "bool": {
                            "filter": filters,
                        }
                    },
                    "script": {
                        "source": """
                            double maxConditionVectorScore = 0.0;
                            double maxTitleVectorScore = 0.0;
                            double maxSummaryVectorScore = 0.0;
                            double maxEligibilityVectorScore = 0.0;
                            double totalOtherConditionScore = 0.0;
                            for (int i = 0; i < params.query_vectors.length; ++i) {
                                maxConditionVectorScore = Math.max(maxConditionVectorScore, cosineSimilarity(params.query_vectors[i], 'condition_vector'));
                                maxTitleVectorScore = Math.max(maxTitleVectorScore, cosineSimilarity(params.query_vectors[i], 'brief_title_vector'));
                                maxSummaryVectorScore = Math.max(maxSummaryVectorScore, cosineSimilarity(params.query_vectors[i], 'brief_summary_vector'));
                                maxEligibilityVectorScore = Math.max(maxEligibilityVectorScore, cosineSimilarity(params.query_vectors[i], 'eligibility_criteria_vector'));
                            }
                            int otherConditionCount = params.other_condition_vectors.length;
                            for (int i = 0; i < otherConditionCount; ++i) {
                                totalOtherConditionScore += cosineSimilarity(params.other_condition_vectors[i], 'condition_vector');
                                totalOtherConditionScore += cosineSimilarity(params.other_condition_vectors[i], 'brief_title_vector');
                                totalOtherConditionScore += cosineSimilarity(params.other_condition_vectors[i], 'eligibility_criteria_vector');
                                totalOtherConditionScore += cosineSimilarity(params.other_condition_vectors[i], 'brief_summary_vector');
                            }
                            if (otherConditionCount > 0) {
                                totalOtherConditionScore /= (otherConditionCount * 4);
                            }
                            double normalizedConditionScore = (maxConditionVectorScore + 1.0) / 2.0;
                            double normalizedTitleScore = (maxTitleVectorScore + 1.0) / 2.0;
                            double normalizedSummaryScore = (maxSummaryVectorScore + 1.0) / 2.0;
                            double normalizedEligibilityScore = (maxEligibilityVectorScore + 1.0) / 2.0;
                            double normalizedOtherConditionScore = (totalOtherConditionScore + 1.0) / 2.0;
                            double combinedVectorScore = (
                                0.3 * normalizedConditionScore +
                                0.1 * normalizedTitleScore +
                                0.1 * normalizedSummaryScore +
                                0.2 * normalizedOtherConditionScore +
                                0.3 * normalizedEligibilityScore
                            );
                            if (combinedVectorScore < params.vector_score_threshold) {
                                return 0;
                            }
                            return combinedVectorScore;
                        """,
                        "params": {
                            "query_vectors": query_vectors,
                            "other_condition_vectors": other_vectors,
                            "vector_score_threshold": vector_score_threshold,
                        },
                    },
                }
            }

        # Hybrid (default)
        return {
            "script_score": {
                "query": {
                    "bool": {
                        "should": should_clauses,
                        "minimum_should_match": 0,
                        "filter": filters,
                    }
                },
                "script": {
                    "source": """
                        double alpha = 0.5;
                        double beta = 0.5;
                        double textScore = _score;
                        double maxTextScore = params.max_text_score;
                        double normalizedTextScore = (maxTextScore == 0) ? 0 : textScore / maxTextScore;
                        double maxConditionVectorScore = 0.0;
                        double maxTitleVectorScore = 0.0;
                        double maxSummaryVectorScore = 0.0;
                        double maxEligibilityVectorScore = 0.0;
                        double totalOtherConditionScore = 0.0;
                        for (int i = 0; i < params.query_vectors.length; ++i) {
                            maxConditionVectorScore = Math.max(maxConditionVectorScore, cosineSimilarity(params.query_vectors[i], 'condition_vector'));
                            maxTitleVectorScore = Math.max(maxTitleVectorScore, cosineSimilarity(params.query_vectors[i], 'brief_title_vector'));
                            maxSummaryVectorScore = Math.max(maxSummaryVectorScore, cosineSimilarity(params.query_vectors[i], 'brief_summary_vector'));
                            maxEligibilityVectorScore = Math.max(maxEligibilityVectorScore, cosineSimilarity(params.query_vectors[i], 'eligibility_criteria_vector'));
                        }
                        int otherConditionCount = params.other_condition_vectors.length;
                        for (int i = 0; i < otherConditionCount; ++i) {
                            totalOtherConditionScore += cosineSimilarity(params.other_condition_vectors[i], 'condition_vector');
                            totalOtherConditionScore += cosineSimilarity(params.other_condition_vectors[i], 'brief_title_vector');
                            totalOtherConditionScore += cosineSimilarity(params.other_condition_vectors[i], 'eligibility_criteria_vector');
                            totalOtherConditionScore += cosineSimilarity(params.other_condition_vectors[i], 'brief_summary_vector');
                        }
                        if (otherConditionCount > 0) {
                            totalOtherConditionScore /= (otherConditionCount * 4);
                        }
                        double normalizedConditionScore = (maxConditionVectorScore + 1.0) / 2.0;
                        double normalizedTitleScore = (maxTitleVectorScore + 1.0) / 2.0;
                        double normalizedSummaryScore = (maxSummaryVectorScore + 1.0) / 2.0;
                        double normalizedEligibilityScore = (maxEligibilityVectorScore + 1.0) / 2.0;
                        double normalizedOtherConditionScore = (totalOtherConditionScore + 1.0) / 2.0;
                        double combinedVectorScore = (
                            0.3 * normalizedConditionScore +
                            0.1 * normalizedTitleScore +
                            0.1 * normalizedSummaryScore +
                            0.2 * normalizedOtherConditionScore +
                            0.3 * normalizedEligibilityScore
                        );
                        if (combinedVectorScore < params.vector_score_threshold) {
                            return 0;
                        }
                        return alpha * normalizedTextScore + beta * combinedVectorScore;
                    """,
                    "params": {
                        "query_vectors": query_vectors,
                        "other_condition_vectors": other_vectors,
                        "max_text_score": max_text_score,
                        "vector_score_threshold": vector_score_threshold,
                    },
                },
            }
        }

    def search_trials(
        self,
        condition: str,
        age_input: Union[int, str],
        sex: str,
        overall_status: Optional[str] = None,
        size: int = 10,
        pre_selected_nct_ids: Optional[List[str]] = None,
        synonyms: Optional[List[str]] = None,
        other_conditions: Optional[List[str]] = None,
        vector_score_threshold: float = 0.0,
        search_mode: str = "hybrid",
    ) -> Tuple[List[Dict], List[float]]:
        if age_input not in ["all", "ALL", "All"]:
            age = self.parse_age_input(age_input)
            if age is None:
                raise ValueError("Could not parse age input.")
        else:
            age = None
        primary_synonyms = _clean_terms([condition] + (synonyms or []))
        other_conditions = _clean_terms(other_conditions or [])
        all_terms = primary_synonyms + other_conditions

        mode = (search_mode or "hybrid").lower()
        embeddings: Dict[str, List[float]] = {}
        if mode in {"vector", "hybrid"} and self.embedder is not None:
            vectors = self.embedder.embed_texts(all_terms)
            embeddings = dict(zip(all_terms, vectors))
            if not embeddings:
                logger.warning(
                    "No valid terms to embed for vector search. Falling back to BM25 only."
                )
                mode = "bm25"
        elif mode in {"vector", "hybrid"} and self.embedder is None:
            logger.warning(
                "Vector/hybrid mode selected but embedder is None. Falling back to BM25 only."
            )
            mode = "bm25"

        max_text_score = (
            1.0 if mode == "vector" else self.get_max_text_score(primary_synonyms)
        )
        query = self.create_query(
            primary_synonyms,
            embeddings,
            age if age is not None else 0,
            sex,
            overall_status,
            max_text_score,
            vector_score_threshold,
            pre_selected_nct_ids,
            other_conditions,
            search_mode=mode,
        )
        try:
            response = with_retries(
                lambda: self.es_client.search(
                    index=self.index_name, body={"size": size, "query": query}
                ),
                logger=logger,
                action="ES trial search",
            )
            hits = response["hits"]["hits"]
            trials = [hit["_source"] for hit in hits]
            scores = [hit["_score"] for hit in hits]
        except Exception:
            logger.exception("Search failed; returning empty results.")
            return [], []
        trials_with_scores = sorted(
            zip(trials, scores), key=lambda x: x[1], reverse=True
        )
        top_x_percent_index = int(len(trials_with_scores) * 1.0)
        trials = [trial for trial, score in trials_with_scores[:top_x_percent_index]]
        logger.info(
            f"[{mode}] Found {len(trials)} trials matching the search criteria."
        )
        return trials, scores


def _clean_terms(terms: List[str]) -> List[str]:
    return [term.strip() for term in terms if term and term.strip()]
