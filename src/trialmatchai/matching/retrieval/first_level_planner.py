from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from trialmatchai.interop.models import ClinicalFact, PatientProfile
from trialmatchai.interop.narrative import render_patient_narrative
from trialmatchai.matching.retrieval.synonyms import disease_synonyms
from trialmatchai.utils.logging_config import setup_logging
from trialmatchai.utils.text import flatten_text

logger = setup_logging(__name__)


FirstLevelChannelKind = Literal[
    "primary_condition",
    "other_condition",
    "concept_synonym",
    "broader_disease",
    "narrative",
    "biomarker",
    "therapy",
    "llm_expansion",
]

DEFAULT_CHANNEL_WEIGHTS: dict[FirstLevelChannelKind, float] = {
    "primary_condition": 1.0,
    # 0.25 (not 0.5): comorbidity hits then fill the tail below main-condition hits instead of
    # evicting low-ranked-but-relevant main trials under the max_trials_first_level cap. Swept
    # empirically -- 0.25 is a Pareto win (every patient's recall held or rose; 0.5 hurt the
    # recall-rich patients via that eviction).
    "other_condition": 0.25,
    "concept_synonym": 0.9,
    "narrative": 0.8,
    "biomarker": 0.7,
    "therapy": 0.45,
    "broader_disease": 0.35,
    "llm_expansion": 0.5,
}

# A highly multi-morbid patient's relevant trials are dominated by the comorbidities, but one
# blended query over many distinct conditions dilutes both BM25 and the mean query vector to
# near-noise (empirically ~1/5 the recall). So each comorbidity gets its OWN focused channel.
# Cap the count to bound per-patient retrieval latency; other_conditions is importance-ordered.
_MAX_OTHER_CONDITION_CHANNELS = 25


class FirstLevelQueryChannel(BaseModel):
    kind: FirstLevelChannelKind
    terms: list[str]
    weight: float = Field(ge=0.0)
    source: str = "deterministic"

    model_config = ConfigDict(extra="forbid")


class FirstLevelQueryPlan(BaseModel):
    patient_id: str | None = None
    channels: list[FirstLevelQueryChannel] = Field(default_factory=list)
    filters: dict[str, Any] = Field(default_factory=dict)
    llm_expansion_enabled: bool = False

    model_config = ConfigDict(extra="forbid")

    def terms_for(self, *kinds: FirstLevelChannelKind) -> list[str]:
        allowed = set(kinds)
        return dedupe_terms(
            term
            for channel in self.channels
            if channel.kind in allowed
            for term in channel.terms
        )


class FirstLevelCandidateEvidence(BaseModel):
    nct_id: str
    score: float
    channels: list[dict[str, Any]] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class LLMQueryExpansion(BaseModel):
    primary_queries: list[str] = Field(default_factory=list)
    disease_aliases: list[str] = Field(default_factory=list)
    broader_queries: list[str] = Field(default_factory=list)
    biomarker_queries: list[str] = Field(default_factory=list)
    treatment_queries: list[str] = Field(default_factory=list)
    discarded_or_uncertain: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class LLMQueryExpansionBackend(Protocol):
    def expand_first_level_queries(
        self,
        *,
        profile: PatientProfile,
        matching_summary: dict[str, Any],
    ) -> str | dict[str, Any]:
        ...


class FirstLevelQueryPlanner:
    def __init__(
        self,
        *,
        entity_annotator: Any = None,
        llm_expander: LLMQueryExpansionBackend
        | Callable[..., str | dict[str, Any]]
        | None = None,
    ) -> None:
        self.entity_annotator = entity_annotator
        self.llm_expander = llm_expander

    def build(
        self,
        *,
        profile: PatientProfile,
        matching_summary: dict[str, Any],
        config: dict[str, Any] | None = None,
        age: int | str | None = None,
        sex: str | None = None,
        overall_status: str | None = None,
    ) -> FirstLevelQueryPlan:
        cfg = config or {}
        channels: list[FirstLevelQueryChannel] = []
        primary_terms = dedupe_terms(
            [
                *matching_summary.get("main_conditions", []),
                *positive_fact_labels(profile.conditions),
            ]
        )
        channels.append(
            self._channel("primary_condition", primary_terms, "matching_summary")
        )

        # One focused channel per comorbidity (see _MAX_OTHER_CONDITION_CHANNELS): a blended
        # channel over all of them dilutes to noise, so each retrieves its own trials at full
        # strength. Terms already in the primary channel are dropped to avoid a redundant channel.
        other_condition_terms = [
            term
            for term in dedupe_terms(matching_summary.get("other_conditions", []))
            if term.casefold() not in {p.casefold() for p in primary_terms}
        ]
        for term in other_condition_terms[:_MAX_OTHER_CONDITION_CHANNELS]:
            channels.append(self._channel("other_condition", [term], "matching_summary"))

        synonym_terms = self._synonym_terms(primary_terms)
        channels.append(self._channel("concept_synonym", synonym_terms, "concept_linker"))

        broader_terms = broader_disease_terms([*primary_terms, *synonym_terms])
        channels.append(self._channel("broader_disease", broader_terms, "deterministic"))

        narrative_terms = narrative_terms_from_summary_or_profile(
            profile,
            matching_summary,
        )
        channels.append(self._channel("narrative", narrative_terms, "patient_narrative"))

        biomarker_terms = dedupe_terms(
            [
                *positive_fact_labels(profile.genomic_findings),
                *biomarker_terms_from_facts(profile.cancer_profile),
            ]
        )
        channels.append(self._channel("biomarker", biomarker_terms, "patient_profile"))

        therapy_terms = therapy_terms_from_profile(profile)
        channels.append(self._channel("therapy", therapy_terms, "patient_profile"))

        llm_enabled = bool(cfg.get("llm_expansion_enabled", False))
        if llm_enabled:
            llm_terms = self._llm_terms(
                profile=profile,
                matching_summary=matching_summary,
                max_terms=int(cfg.get("llm_max_terms", 12)),
            )
            channels.append(self._channel("llm_expansion", llm_terms, "llm"))

        return FirstLevelQueryPlan(
            patient_id=profile.patient_id,
            channels=[channel for channel in channels if channel.terms],
            filters={
                "age": age,
                "sex": sex,
                "overall_status": overall_status,
                "hard_filters": cfg.get(
                    "hard_filters",
                    ["age", "sex", "overall_status"],
                ),
            },
            llm_expansion_enabled=llm_enabled,
        )

    def _channel(
        self,
        kind: FirstLevelChannelKind,
        terms: Sequence[str],
        source: str,
    ) -> FirstLevelQueryChannel:
        return FirstLevelQueryChannel(
            kind=kind,
            terms=dedupe_terms(terms),
            weight=DEFAULT_CHANNEL_WEIGHTS[kind],
            source=source,
        )

    def _synonym_terms(self, primary_terms: Sequence[str]) -> list[str]:
        output: list[str] = []
        for term in primary_terms[:5]:
            output.extend(disease_synonyms(self.entity_annotator, term))
        return dedupe_terms(output)

    def _llm_terms(
        self,
        *,
        profile: PatientProfile,
        matching_summary: dict[str, Any],
        max_terms: int,
    ) -> list[str]:
        if self.llm_expander is None:
            logger.warning(
                "First-level LLM expansion is enabled but no expander is configured."
            )
            return []
        try:
            if hasattr(self.llm_expander, "expand_first_level_queries"):
                raw = self.llm_expander.expand_first_level_queries(
                    profile=profile,
                    matching_summary=matching_summary,
                )
            else:
                raw = self.llm_expander(
                    profile=profile,
                    matching_summary=matching_summary,
                )
            parsed = parse_llm_query_expansion(raw, max_terms=max_terms)
        except Exception:
            logger.exception("Discarding invalid first-level LLM query expansion.")
            return []
        return dedupe_terms(
            [
                *parsed.primary_queries,
                *parsed.disease_aliases,
                *parsed.broader_queries,
                *parsed.biomarker_queries,
                *parsed.treatment_queries,
            ]
        )[:max_terms]


def parse_llm_query_expansion(
    raw: str | dict[str, Any],
    *,
    max_terms: int,
) -> LLMQueryExpansion:
    payload = json.loads(raw) if isinstance(raw, str) else raw
    parsed = LLMQueryExpansion.model_validate(payload)
    capped: dict[str, list[str]] = {}
    remaining = max(0, max_terms)
    for field in (
        "primary_queries",
        "disease_aliases",
        "broader_queries",
        "biomarker_queries",
        "treatment_queries",
    ):
        values = dedupe_terms(getattr(parsed, field))
        if remaining <= 0:
            capped[field] = []
            continue
        capped[field] = values[:remaining]
        remaining -= len(capped[field])
    capped["discarded_or_uncertain"] = dedupe_terms(parsed.discarded_or_uncertain)
    return LLMQueryExpansion.model_validate(capped)


def fuse_first_level_channel_hits(
    channel_hits: Sequence[tuple[FirstLevelQueryChannel, list[dict], list[float]]],
    *,
    size: int,
    rrf_k: int = 60,
) -> tuple[list[dict], list[float], list[FirstLevelCandidateEvidence]]:
    candidates: dict[str, dict[str, Any]] = {}
    evidence: dict[str, list[dict[str, Any]]] = {}
    for channel, trials, scores in channel_hits:
        for rank, (trial, raw_score) in enumerate(zip(trials, scores), start=1):
            nct_id = str(trial.get("nct_id") or "")
            if not nct_id:
                continue
            contribution = channel.weight / (rrf_k + rank)
            if nct_id not in candidates:
                candidates[nct_id] = {
                    "trial": trial,
                    "score": 0.0,
                }
                evidence[nct_id] = []
            candidates[nct_id]["score"] += contribution
            evidence[nct_id].append(
                {
                    "channel": channel.kind,
                    "rank": rank,
                    "raw_score": float(raw_score),
                    "weight": channel.weight,
                    "terms": channel.terms,
                    "contribution": contribution,
                }
            )

    ranked = sorted(
        candidates.values(),
        key=lambda item: item["score"],
        reverse=True,
    )[:size]
    trials = [item["trial"] for item in ranked]
    scores = [float(item["score"]) for item in ranked]
    candidate_evidence = [
        FirstLevelCandidateEvidence(
            nct_id=str(item["trial"].get("nct_id")),
            score=float(item["score"]),
            channels=evidence.get(str(item["trial"].get("nct_id")), []),
        )
        for item in ranked
    ]
    return trials, scores, candidate_evidence


def positive_fact_labels(facts: Sequence[ClinicalFact]) -> list[str]:
    return dedupe_terms(fact.label for fact in facts if not fact.negated)


def narrative_terms_from_summary_or_profile(
    profile: PatientProfile,
    matching_summary: dict[str, Any],
    *,
    max_chars: int = 1500,
) -> list[str]:
    sentences = matching_summary.get("patient_narrative") or render_patient_narrative(
        profile
    )
    sentences = [
        sentence
        for sentence in sentences
        if " absent" not in str(sentence).casefold()
    ]
    narrative = flatten_text(sentences)
    if not narrative:
        return []
    return [narrative[:max_chars].rsplit(" ", 1)[0].strip() or narrative[:max_chars]]


def biomarker_terms_from_facts(facts: Sequence[ClinicalFact]) -> list[str]:
    terms: list[str] = []
    gene_pattern = re.compile(
        r"\b(ALK|BRAF|BRCA1|BRCA2|EGFR|ERBB2|HER2|KRAS|NTRK|PD-L1|PIK3CA|ROS1|TP53)\b",
        re.IGNORECASE,
    )
    for fact in facts:
        if fact.negated:
            continue
        text = flatten_text([fact.label, fact.description, fact.evidence_text])
        terms.extend(match.group(1).upper() for match in gene_pattern.finditer(text))
    return dedupe_terms(terms)


def therapy_terms_from_profile(profile: PatientProfile) -> list[str]:
    terms: list[str] = []
    for fact in [*profile.medications, *profile.procedures]:
        if fact.negated:
            continue
        terms.append(fact.label)
        if fact.temporality and "prior" in fact.temporality.casefold():
            terms.append(f"prior {fact.label}")
    return dedupe_terms(terms)


def broader_disease_terms(terms: Sequence[str]) -> list[str]:
    output: list[str] = []
    for term in terms:
        normalized = term.casefold()
        if any(
            marker in normalized
            for marker in (
                "cancer",
                "carcinoma",
                "melanoma",
                "sarcoma",
                "tumor",
                "tumour",
                "neoplasm",
            )
        ):
            output.extend(["cancer", "malignant neoplasm", "solid tumor"])
        if any(marker in normalized for marker in ("leukemia", "lymphoma", "myeloma")):
            output.extend(["hematologic malignancy", "blood cancer"])
        if "lung" in normalized and any(
            marker in normalized for marker in ("cancer", "carcinoma", "nsclc")
        ):
            output.extend(["thoracic cancer", "solid tumor"])
    return dedupe_terms(output)


def dedupe_terms(values: Any) -> list[str]:
    if values is None:
        return []
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        cleaned = re.sub(r"\s+", " ", str(value or "")).strip()
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        output.append(cleaned)
    return output
