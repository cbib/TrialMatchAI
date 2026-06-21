from __future__ import annotations

import hashlib
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from trialmatchai.interop.models import ClinicalFact, NormalizedCode, Provenance


FHIR_SYSTEM_TO_VOCABULARY = {
    "http://snomed.info/sct": "SNOMED",
    "http://loinc.org": "LOINC",
    "http://www.nlm.nih.gov/research/umls/rxnorm": "RxNorm",
    "http://hl7.org/fhir/sid/icd-10-cm": "ICD10CM",
    "http://hl7.org/fhir/sid/icd-10": "ICD10",
    "https://hpo.jax.org/app/browse/term": "HP",
    "http://purl.obolibrary.org/obo/hp.owl": "HP",
    "http://purl.obolibrary.org/obo/mondo.owl": "MONDO",
}


def stable_id(*parts: Any) -> str:
    payload = "|".join(str(part) for part in parts if part not in (None, ""))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value).strip()
    if isinstance(value, Mapping):
        return clean_text(" ".join(clean_text(item) for item in value.values()))
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        return clean_text(" ".join(clean_text(item) for item in value))
    return clean_text(str(value))


def normalize_gender(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().casefold()
    if text in {"female", "f", "woman", "women"}:
        return "Female"
    if text in {"male", "m", "man", "men"}:
        return "Male"
    if text in {"all", "any", "unknown", "other", "undifferentiated"}:
        return text.title()
    return str(value).strip() or None


def parse_date(value: Any) -> date | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).date()
    except ValueError:
        return None


def age_years_from_birth_date(birth_date: date | None) -> float | None:
    if birth_date is None:
        return None
    today = date.today()
    years = today.year - birth_date.year
    if (today.month, today.day) < (birth_date.month, birth_date.day):
        years -= 1
    return float(years) if years >= 0 else None


def parse_iso8601_age_years(value: Any) -> float | None:
    if not value:
        return None
    text = str(value)
    match = re.fullmatch(r"P(?:(\d+(?:\.\d+)?)Y)?(?:(\d+(?:\.\d+)?)M)?", text)
    if not match:
        return None
    years = float(match.group(1) or 0)
    months = float(match.group(2) or 0)
    return round(years + months / 12, 2)


def code_from_ontology_class(value: Mapping[str, Any] | None) -> NormalizedCode | None:
    if not value:
        return None
    identifier = str(value.get("id") or value.get("code") or "").strip()
    label = str(value.get("label") or value.get("display") or "").strip() or None
    if not identifier and not label:
        return None
    vocabulary = "local"
    code = identifier or label or ""
    if ":" in identifier:
        vocabulary, code = identifier.split(":", 1)
    elif "/" in identifier:
        code = identifier.rstrip("/").split("/")[-1]
    return NormalizedCode(
        vocabulary=vocabulary,
        code=code,
        label=label,
        system=value.get("system"),
        mapping_status="exact" if identifier else "unmapped",
    )


def code_from_fhir_codeable(value: Mapping[str, Any] | None) -> NormalizedCode | None:
    if not value:
        return None
    text = clean_text(value.get("text"))
    codings = value.get("coding") or []
    if isinstance(codings, list) and codings:
        coding = codings[0] or {}
        system = str(coding.get("system") or "")
        vocabulary = FHIR_SYSTEM_TO_VOCABULARY.get(system, system.rsplit("/", 1)[-1])
        code = str(coding.get("code") or "").strip()
        label = clean_text(coding.get("display")) or text or None
        if code:
            return NormalizedCode(
                vocabulary=vocabulary or "FHIR",
                code=code,
                label=label,
                system=system or None,
                mapping_status="exact",
            )
    if text:
        return NormalizedCode(
            vocabulary="FHIR",
            code=text,
            label=text,
            mapping_status="unmapped",
        )
    return None


def label_from_fhir_codeable(value: Mapping[str, Any] | None) -> str:
    if not value:
        return ""
    text = clean_text(value.get("text"))
    if text:
        return text
    codings = value.get("coding") or []
    if isinstance(codings, list):
        for coding in codings:
            label = clean_text((coding or {}).get("display"))
            if label:
                return label
            code = clean_text((coding or {}).get("code"))
            if code:
                return code
    return ""


def make_fact(
    *,
    category: str,
    label: str,
    provenance: Provenance,
    description: str | None = None,
    original_code: NormalizedCode | None = None,
    normalized_codes: list[NormalizedCode] | None = None,
    evidence_text: str | None = None,
    evidence_start: int | None = None,
    evidence_end: int | None = None,
    confidence: float | None = None,
    negated: bool = False,
    temporality: str | None = None,
    extra: dict[str, Any] | None = None,
) -> ClinicalFact:
    cleaned_label = clean_text(label) or "Unknown"
    codes = normalized_codes or ([] if original_code is None else [original_code])
    return ClinicalFact(
        fact_id=stable_id(
            category,
            cleaned_label,
            provenance.source_format,
            provenance.source_path,
            provenance.source_resource,
            provenance.source_table,
            evidence_start,
            evidence_end,
        ),
        category=category,
        label=cleaned_label,
        description=clean_text(description) or None,
        original_code=original_code,
        normalized_codes=codes,
        vocabulary=(codes[0].vocabulary if codes else None),
        evidence_text=clean_text(evidence_text) or None,
        evidence_start=evidence_start,
        evidence_end=evidence_end,
        confidence=confidence,
        negated=negated,
        temporality=temporality,
        mapping_status=codes[0].mapping_status if codes else "unmapped",
        provenance=provenance,
        extra=extra or {},
    )


def source_path_string(path: str | Path | None) -> str | None:
    return str(Path(path).resolve()) if path else None


def safe_patient_id(value: Any, fallback: str) -> str:
    candidate = clean_text(value) or fallback
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", candidate).strip("-") or fallback
