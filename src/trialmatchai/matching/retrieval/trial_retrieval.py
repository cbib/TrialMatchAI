from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from dateutil import parser as date_parser
from trialmatchai.matching.retrieval.first_level_planner import (
    FirstLevelCandidateEvidence,
    FirstLevelQueryPlan,
    FirstLevelQueryPlanner,
    LLMQueryExpansionBackend,
    fuse_first_level_channel_hits,
)
from trialmatchai.matching.retrieval.synonyms import disease_synonyms
from trialmatchai.search.lancedb_backend import TrialSearchBackend
from trialmatchai.utils.logging_config import setup_logging

if TYPE_CHECKING:
    from trialmatchai.interop.models import PatientProfile
    from trialmatchai.models.embedding.text_embedder import TextEmbedder

logger = setup_logging(__name__)


class ClinicalTrialSearch:
    def __init__(
        self,
        search_backend: TrialSearchBackend,
        embedder: Optional[TextEmbedder],
        entity_annotator=None,
        llm_query_expander: LLMQueryExpansionBackend | None = None,
    ):
        self.search_backend = search_backend
        self.embedder = embedder
        self.entity_annotator = entity_annotator
        self.query_planner = FirstLevelQueryPlanner(
            entity_annotator=entity_annotator,
            llm_expander=llm_query_expander,
        )

    def get_synonyms(self, condition: str) -> List[str]:
        return disease_synonyms(self.entity_annotator, condition)

    def parse_age_input(self, age_input: Union[int, str]) -> Optional[int]:
        if isinstance(age_input, int):
            return age_input if 0 <= age_input <= 150 else None
        elif isinstance(age_input, str):
            age_input = age_input.strip().lower()
            age_keywords = ["year", "years", "yr", "yrs"]
            try:
                age_str = age_input
                for keyword in age_keywords:
                    if age_input.endswith(keyword):
                        age_str = age_input.replace(keyword, "").strip()
                        break
                parsed = int(age_str)
                # A purely-numeric age is authoritative; don't fall through to fuzzy
                # date parsing, which would misread "-5" as a near-today date (age 0).
                return parsed if 0 <= parsed <= 150 else None
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
        other_conditions: Optional[List[str]] = None,
    ) -> Dict:
        # Assemble the capped term lists + embeddings the backend consumes. Filters
        # (age/sex/status/nct_ids) go to the backend directly, not through this dict.
        max_conditions_per_query = 800
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
        age = None
        if str(age_input).strip().casefold() != "all":
            age = self.parse_age_input(age_input)
            if age is None:
                logger.warning(
                    "Could not parse age %r; proceeding without an age filter.",
                    age_input,
                )
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

        query = self.create_query(primary_synonyms, embeddings, other_conditions)
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

    def build_query_plan(
        self,
        *,
        profile: "PatientProfile",
        matching_summary: dict[str, Any],
        config: dict[str, Any] | None = None,
        age: int | str | None = None,
        sex: str | None = None,
        overall_status: str | None = None,
    ) -> FirstLevelQueryPlan:
        return self.query_planner.build(
            profile=profile,
            matching_summary=matching_summary,
            config=config,
            age=age,
            sex=sex,
            overall_status=overall_status,
        )

    def search_trials_with_plan(
        self,
        *,
        query_plan: FirstLevelQueryPlan,
        age_input: Union[int, str],
        sex: str,
        overall_status: Optional[str] = None,
        size: int = 1000,
        per_channel_size: int = 300,
        pre_selected_nct_ids: Optional[List[str]] = None,
        vector_score_threshold: float = 0.0,
        search_mode: str = "hybrid",
        rrf_k: int = 60,
    ) -> tuple[list[dict], list[float], list[FirstLevelCandidateEvidence]]:
        # Apply only the hard filters listed in the plan (default age/sex/overall_status),
        # making first_level.hard_filters controllable. None -> default; [] -> none, so
        # hard_filters=[] truly disables them.
        configured = query_plan.filters.get("hard_filters")
        hard_filters = set(
            configured if configured is not None else ["age", "sex", "overall_status"]
        )

        age = None
        if "age" in hard_filters and str(age_input).strip().casefold() != "all":
            age = self.parse_age_input(age_input)
            if age is None:
                logger.warning(
                    "Could not parse age %r; proceeding without an age filter.",
                    age_input,
                )
        effective_sex = sex if "sex" in hard_filters else "all"
        effective_status = overall_status if "overall_status" in hard_filters else None

        mode = (search_mode or "hybrid").lower()
        if mode in {"vector", "hybrid"} and self.embedder is None:
            logger.warning(
                "Vector/hybrid mode selected but embedder is None. Falling back to BM25 only."
            )
            mode = "bm25"

        channel_hits = []
        failed_channels = 0
        # Embed each channel's terms up front on the (single) GPU embedder -- kept sequential
        # because concurrent forward passes on one model are not thread-safe. Embedding is cheap
        # (~0.2s total); the cost is the per-channel search below.
        prepared = [
            (channel, self._embed_terms(channel.terms, mode))
            for channel in query_plan.channels
        ]

        def _search_channel(item):
            channel, embeddings = item
            try:
                trials, scores = self.search_backend.search_trials(
                    primary_terms=channel.terms,
                    other_terms=[],
                    embeddings=embeddings,
                    age=age,
                    sex=effective_sex,
                    overall_status=effective_status,
                    pre_selected_nct_ids=pre_selected_nct_ids,
                    size=per_channel_size,
                    vector_score_threshold=vector_score_threshold,
                    search_mode=mode,
                )
                return channel, trials, scores, None
            except Exception as exc:  # captured, re-raised into the caller thread's result
                return channel, None, None, exc

        # The per-channel searches are independent and dominated by LanceDB's native FTS/vector
        # scan and numpy re-ranking, both of which release the GIL -- so a thread pool turns the
        # ~21 sequential channel searches into a concurrent fan-out (the prior sequential loop was
        # ~5s x 21 = the whole first-level latency). Reads on a LanceDB table are concurrency-safe.
        if len(prepared) > 1:
            # Respect the cgroup CPU allotment (SLURM/containers) rather than the whole machine:
            # sched_getaffinity reflects --cpus-per-task; os.cpu_count() would oversubscribe.
            try:
                cpus = len(os.sched_getaffinity(0))
            except AttributeError:  # non-Linux fallback
                cpus = os.cpu_count() or 4
            workers = min(len(prepared), max(1, cpus))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                results = list(executor.map(_search_channel, prepared))
        else:
            results = [_search_channel(item) for item in prepared]

        for channel, trials, scores, exc in results:
            if exc is not None:
                failed_channels += 1
                logger.warning(
                    "First-level channel search failed for %s; skipping channel: %s",
                    channel.kind,
                    exc,
                )
                continue
            if trials:
                channel_hits.append((channel, trials, scores))

        if query_plan.channels and not channel_hits:
            logger.warning(
                "First-level retrieval produced no candidates from %d channels "
                "(%d errored). Check the search backend/tables.",
                len(query_plan.channels),
                failed_channels,
            )

        trials, scores, evidence = fuse_first_level_channel_hits(
            channel_hits,
            size=size,
            rrf_k=rrf_k,
        )
        logger.info(
            "[%s] First-level planned retrieval found %s trials across %s channels.",
            mode,
            len(trials),
            len(channel_hits),
        )
        return trials, scores, evidence

    def _embed_terms(self, terms: list[str], mode: str) -> dict[str, list[float]]:
        if mode not in {"vector", "hybrid"} or self.embedder is None:
            return {}
        vectors = self.embedder.embed_texts(terms)
        return dict(zip(terms, vectors))


def _clean_terms(terms: List[str]) -> List[str]:
    return [term.strip() for term in terms if term and term.strip()]
