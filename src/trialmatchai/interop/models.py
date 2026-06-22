from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


MappingStatus = Literal[
    "exact",
    "normalized",
    "broader",
    "narrower",
    "inferred",
    "unsupported",
    "unmapped",
]


class Provenance(BaseModel):
    source_format: str
    source_id: str | None = None
    source_path: str | None = None
    source_resource: str | None = None
    source_table: str | None = None
    source_field: str | None = None

    model_config = ConfigDict(extra="allow")


class NormalizedCode(BaseModel):
    vocabulary: str
    code: str
    label: str | None = None
    system: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    mapping_status: MappingStatus = "unmapped"

    model_config = ConfigDict(extra="allow")


class Demographics(BaseModel):
    sex: str | None = None
    gender: str | None = None
    birth_date: date | None = None
    age_years: float | None = Field(default=None, ge=0)
    species: str | None = None
    description: str | None = None

    model_config = ConfigDict(extra="allow")


class Location(BaseModel):
    """A patient's geographic location, used for optional site-based filtering."""

    country: str | None = None
    state: str | None = None
    city: str | None = None

    model_config = ConfigDict(extra="allow")


class ClinicalFact(BaseModel):
    fact_id: str
    category: str
    label: str
    description: str | None = None
    original_code: NormalizedCode | None = None
    normalized_codes: list[NormalizedCode] = Field(default_factory=list)
    vocabulary: str | None = None
    evidence_text: str | None = None
    evidence_start: int | None = Field(default=None, ge=0)
    evidence_end: int | None = Field(default=None, ge=0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    negated: bool = False
    temporality: str | None = None
    mapping_status: MappingStatus = "unmapped"
    provenance: Provenance
    extra: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow")

    @field_validator("evidence_end")
    @classmethod
    def validate_evidence_end(cls, value: int | None, info):
        start = info.data.get("evidence_start")
        if value is not None and start is not None and value < start:
            raise ValueError("evidence_end must be >= evidence_start")
        return value


class PatientNote(BaseModel):
    note_id: str
    text: str
    note_type: str = "clinical-note"
    entities: list[dict[str, Any]] = Field(default_factory=list)
    provenance: Provenance

    model_config = ConfigDict(extra="allow")


class SourceDocument(BaseModel):
    document_id: str
    title: str | None = None
    document_type: str | None = None
    url: str | None = None
    text: str | None = None
    provenance: Provenance

    model_config = ConfigDict(extra="allow")


class PatientProfile(BaseModel):
    patient_id: str
    demographics: Demographics = Field(default_factory=Demographics)
    location: Location | None = None
    conditions: list[ClinicalFact] = Field(default_factory=list)
    phenotypes: list[ClinicalFact] = Field(default_factory=list)
    observations: list[ClinicalFact] = Field(default_factory=list)
    medications: list[ClinicalFact] = Field(default_factory=list)
    procedures: list[ClinicalFact] = Field(default_factory=list)
    diagnostic_reports: list[ClinicalFact] = Field(default_factory=list)
    genomic_findings: list[ClinicalFact] = Field(default_factory=list)
    cancer_profile: list[ClinicalFact] = Field(default_factory=list)
    family_history: list[ClinicalFact] = Field(default_factory=list)
    notes: list[PatientNote] = Field(default_factory=list)
    source_documents: list[SourceDocument] = Field(default_factory=list)
    provenance: list[Provenance] = Field(default_factory=list)
    unsupported: list[dict[str, Any]] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow")

    def add_fact(self, fact: ClinicalFact) -> None:
        bucket = {
            "condition": self.conditions,
            "phenotype": self.phenotypes,
            "observation": self.observations,
            "medication": self.medications,
            "procedure": self.procedures,
            "diagnostic_report": self.diagnostic_reports,
            "genomic_finding": self.genomic_findings,
            "cancer": self.cancer_profile,
            "family_history": self.family_history,
        }.get(fact.category)
        if bucket is None:
            self.observations.append(fact)
        else:
            bucket.append(fact)
