from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

from dateutil import parser as date_parser
from trialmatchai.models.embedding.text_embedder import TextEmbedder
from trialmatchai.search.lancedb_backend import TrialSearchBackend
from trialmatchai.utils.logging_config import setup_logging

logger = setup_logging(__name__)


class ClinicalTrialSearch:
    def __init__(
        self,
        search_backend: TrialSearchBackend,
        embedder: Optional[TextEmbedder],
        bio_med_ner=None,
        entity_annotator=None,
    ):
        self.search_backend = search_backend
        self.embedder = embedder
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
        max_conditions_per_query = 800  # Conservative limit
        all_conditions = synonyms + (other_conditions or [])
        if len(all_conditions) > max_conditions_per_query:
            logger.warning(
                f"Capping search conditions from {len(all_conditions)} to {max_conditions_per_query}"
            )
            capped_synonyms = synonyms[:max_conditions_per_query]
            remaining_slots = max_conditions_per_query - len(capped_synonyms)
            capped_other = (
                (other_conditions or [])[:remaining_slots]
                if remaining_slots > 0
                else []
            )
            synonyms = capped_synonyms
            other_conditions = capped_other

        return {
            "primary_terms": synonyms,
            "other_terms": other_conditions or [],
            "embeddings": embeddings,
            "age": age,
            "sex": sex,
            "overall_status": overall_status,
            "pre_selected_nct_ids": pre_selected_nct_ids or [],
            "search_mode": search_mode,
            "vector_score_threshold": vector_score_threshold,
            "max_text_score": max_text_score,
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

        query = self.create_query(
            primary_synonyms,
            embeddings,
            age if age is not None else 0,
            sex,
            overall_status,
            1.0,
            vector_score_threshold,
            pre_selected_nct_ids,
            other_conditions,
            search_mode=mode,
        )
        try:
            trials, scores = self.search_backend.search_trials(
                primary_terms=query["primary_terms"],
                other_terms=query["other_terms"],
                embeddings=query["embeddings"],
                age=age,
                sex=sex,
                overall_status=overall_status,
                pre_selected_nct_ids=pre_selected_nct_ids,
                size=size,
                vector_score_threshold=vector_score_threshold,
                search_mode=mode,
            )
        except Exception:
            logger.exception("First-level search failed; returning empty results.")
            return [], []
        logger.info(
            f"[{mode}] Found {len(trials)} trials matching the search criteria."
        )
        return trials, scores


def _clean_terms(terms: List[str]) -> List[str]:
    return [term.strip() for term in terms if term and term.strip()]
