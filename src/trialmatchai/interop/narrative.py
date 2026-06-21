from __future__ import annotations

import re

from trialmatchai.interop.models import ClinicalFact, PatientProfile


def render_patient_narrative(
    profile: PatientProfile,
    *,
    style: str = "rag",
) -> list[str]:
    """Render structured profile facts into deterministic LLM-ready sentences."""
    lines: list[str] = []
    demographics = profile.demographics
    demo_bits = []
    if demographics.age_years is not None:
        demo_bits.append(f"age {demographics.age_years:g} years")
    if demographics.sex:
        demo_bits.append(f"sex {demographics.sex}")
    if demographics.gender and demographics.gender != demographics.sex:
        demo_bits.append(f"gender {demographics.gender}")
    if demographics.species:
        demo_bits.append(f"species {demographics.species}")
    if demographics.description:
        demo_bits.append(demographics.description)
    if demo_bits:
        lines.append("Patient demographics: " + "; ".join(demo_bits) + ".")

    lines.extend(_render_fact_group("Diagnoses", profile.conditions))
    lines.extend(_render_fact_group("Phenotypes", profile.phenotypes))
    lines.extend(_render_fact_group("Observations", profile.observations))
    lines.extend(_render_fact_group("Medications", profile.medications))
    lines.extend(_render_fact_group("Procedures", profile.procedures))
    lines.extend(_render_fact_group("Diagnostic reports", profile.diagnostic_reports))
    lines.extend(_render_fact_group("Genomic findings", profile.genomic_findings))
    lines.extend(_render_fact_group("Cancer profile", profile.cancer_profile))
    lines.extend(_render_fact_group("Family history", profile.family_history))

    if style == "audit":
        for note in profile.notes:
            lines.append(f"Source note {note.note_id}: {note.text}")
    elif profile.notes:
        note_text = " ".join(note.text for note in profile.notes[:3])
        if note_text:
            lines.append(f"Clinical note context: {note_text[:2000]}")

    return lines or ["No structured patient facts were available."]


def render_search_terms(profile: PatientProfile) -> tuple[list[str], list[str]]:
    main_conditions = _dedupe(
        fact.label for fact in profile.conditions if not fact.negated
    )
    if not main_conditions:
        main_conditions = _dedupe(
            fact.label for fact in profile.phenotypes if not fact.negated
        )
    if not main_conditions:
        main_conditions = _dedupe(
            _note_search_term(note.text) for note in profile.notes if note.text
        )[:1]
    other_terms = _dedupe(
        [
            *[fact.label for fact in profile.phenotypes if not fact.negated],
            *[fact.label for fact in profile.observations if not fact.negated],
            *[fact.label for fact in profile.medications if not fact.negated],
            *[fact.label for fact in profile.procedures if not fact.negated],
            *[fact.label for fact in profile.genomic_findings if not fact.negated],
            *[fact.label for fact in profile.cancer_profile if not fact.negated],
            *[fact.label for fact in profile.family_history if not fact.negated],
        ]
    )
    return main_conditions, [term for term in other_terms if term not in main_conditions]


def _render_fact_group(label: str, facts: list[ClinicalFact]) -> list[str]:
    if not facts:
        return []
    rendered = []
    for fact in facts:
        status = "absent" if fact.negated else "present"
        pieces = [fact.label, status]
        if fact.description:
            pieces.append(fact.description)
        if fact.temporality:
            pieces.append(f"timing {fact.temporality}")
        code = fact.normalized_codes[0] if fact.normalized_codes else None
        if code and code.code:
            pieces.append(f"code {code.vocabulary}:{code.code}")
        rendered.append(f"{label}: " + "; ".join(pieces) + ".")
    return rendered


def _dedupe(values) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        cleaned = str(value).strip()
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        output.append(cleaned)
    return output


def _note_search_term(text: str, *, max_chars: int = 500) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    truncated = cleaned[:max_chars].rsplit(" ", 1)[0].strip()
    return truncated or cleaned[:max_chars].strip()
