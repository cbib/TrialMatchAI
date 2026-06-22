from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


ConstraintKind = Literal[
    "age",
    "sex",
    "condition",
    "phenotype",
    "medication",
    "procedure",
    "lab",
    "biomarker",
    "performance_status",
    "temporal",
]
ConstraintPolarity = Literal["inclusion", "exclusion", "unknown"]
ConstraintComparator = Literal[
    "present",
    "absent",
    "eq",
    "ne",
    "gt",
    "ge",
    "lt",
    "le",
    "between",
    "positive",
    "negative",
    "mutated",
    "wildtype",
    "prior",
    "current",
]
EvaluationStatus = Literal["matched", "violated", "unknown", "not_applicable"]


class Constraint(BaseModel):
    kind: ConstraintKind
    label: str
    comparator: ConstraintComparator = "present"
    value: float | str | None = None
    min_value: float | None = None
    max_value: float | None = None
    unit: str | None = None
    normalized_codes: list[dict[str, Any]] = Field(default_factory=list)
    temporal_window: str | None = None
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    evidence_text: str | None = None
    evidence_start: int | None = Field(default=None, ge=0)
    evidence_end: int | None = Field(default=None, ge=0)

    model_config = ConfigDict(extra="forbid")


class ConstraintSet(BaseModel):
    nct_id: str
    criteria_id: str
    polarity: ConstraintPolarity
    source_text: str
    source_start: int | None = Field(default=None, ge=0)
    source_end: int | None = Field(default=None, ge=0)
    constraints: list[Constraint] = Field(default_factory=list)
    extractor_version: str = "deterministic-v1"

    model_config = ConfigDict(extra="forbid")


class PatientConstraintFact(BaseModel):
    kind: ConstraintKind
    label: str
    normalized_codes: list[dict[str, Any]] = Field(default_factory=list)
    value: float | None = None
    unit: str | None = None
    negated: bool = False
    temporality: str | None = None
    evidence_text: str | None = None
    source: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class PatientConstraintContext(BaseModel):
    patient_id: str
    age_years: float | None = None
    sex: str | None = None
    gender: str | None = None
    facts: list[PatientConstraintFact] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class ConstraintEvaluation(BaseModel):
    nct_id: str
    criteria_id: str
    polarity: ConstraintPolarity
    constraint: Constraint
    status: EvaluationStatus
    score_signal: float = Field(0.0, ge=-1.0, le=1.0)
    patient_evidence: str | None = None
    reason: str

    model_config = ConfigDict(extra="forbid")


class CriterionConstraintEvaluation(BaseModel):
    nct_id: str
    criteria_id: str
    criterion: str
    polarity: ConstraintPolarity
    evaluations: list[ConstraintEvaluation] = Field(default_factory=list)
    constraint_signal: float = Field(0.0, ge=-1.0, le=1.0)
    matched_count: int = 0
    violated_count: int = 0
    unknown_count: int = 0
    not_applicable_count: int = 0

    model_config = ConfigDict(extra="forbid")
