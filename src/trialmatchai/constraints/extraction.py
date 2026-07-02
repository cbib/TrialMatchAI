from __future__ import annotations

import re
from typing import Any, Mapping

from trialmatchai.constraints.models import (
    Constraint,
    ConstraintPolarity,
    ConstraintSet,
)
from trialmatchai.utils.text import flatten_text


GENE_SYMBOLS = (
    "ALK",
    "BRAF",
    "BRCA1",
    "BRCA2",
    "EGFR",
    "ERBB2",
    "HER2",
    "KRAS",
    "NTRK",
    "PD-L1",
    "PIK3CA",
    "ROS1",
    "TP53",
)
LAB_ALIASES = (
    "absolute neutrophil count",
    "anc",
    "alt",
    "ast",
    "bilirubin",
    "creatinine",
    "hemoglobin",
    "platelet count",
    "platelets",
)


def extract_constraint_set(
    *,
    nct_id: str,
    criteria_id: str,
    criterion: str,
    eligibility_type: str,
    entities: Any = None,
) -> ConstraintSet:
    text = flatten_text(criterion)
    polarity = normalize_polarity(eligibility_type)
    constraints: list[Constraint] = []
    constraints.extend(_age_constraints(text))
    constraints.extend(_sex_constraints(text))
    constraints.extend(_performance_constraints(text))
    constraints.extend(_lab_constraints(text))
    constraints.extend(_biomarker_constraints(text))
    constraints.extend(_prior_therapy_constraints(text))
    constraints.extend(_temporal_constraints(text))
    constraints.extend(_entity_constraints(entities, text))
    constraints = _dedupe_constraints(constraints)
    return ConstraintSet(
        nct_id=str(nct_id),
        criteria_id=str(criteria_id),
        polarity=polarity,
        source_text=text,
        source_start=0 if text else None,
        source_end=len(text) if text else None,
        constraints=constraints,
    )


def normalize_polarity(value: str | None) -> ConstraintPolarity:
    normalized = (value or "").casefold()
    if "excl" in normalized:
        return "exclusion"
    if "incl" in normalized:
        return "inclusion"
    return "unknown"


def _age_constraints(text: str) -> list[Constraint]:
    constraints: list[Constraint] = []
    lower = text.casefold()
    # Require an explicit age signal (trailing year unit or age/aged cue) so dosing/lab numbers aren't read as ages.
    year = r"(?:years?|yrs?|y/?o)"
    age_cue = r"(?:age[d]?|years?\s+of\s+age)"

    for pattern in (
        rf"\b(\d{{1,3}})\s*(?:-|to|through)\s*(\d{{1,3}})\s*{year}\b",
        rf"\b{age_cue}\s*(\d{{1,3}})\s*(?:-|to|through)\s*(\d{{1,3}})\b",
    ):
        for match in re.finditer(pattern, lower):
            start, end = match.span()
            constraints.append(
                Constraint(
                    kind="age",
                    label="age",
                    comparator="between",
                    min_value=float(match.group(1)),
                    max_value=float(match.group(2)),
                    unit="years",
                    evidence_text=text[start:end],
                    evidence_start=start,
                    evidence_end=end,
                )
            )
    for pattern in (
        rf"\b(\d{{1,3}})\s*{year}\s*(?:or\s+)?(?:older|greater|above)\b",
        rf"\b(?:>=|≥|at\s+least)\s*(\d{{1,3}})\s*{year}\b",
        rf"\b{age_cue}\s*(\d{{1,3}})\s*(?:or\s+)?(?:older|greater|above)\b",
        rf"\b{age_cue}\s*(?:>=|≥|at\s+least)\s*(\d{{1,3}})\b",
    ):
        for match in re.finditer(pattern, lower):
            constraints.append(_age_bound(text, match, "ge", float(match.group(1))))
    for pattern in (
        rf"\b(?:younger|less|under|below)\s+than\s+(\d{{1,3}})\s*{year}\b",
        rf"\b{age_cue}\s*(?:younger|less|under|below)\s+than\s+(\d{{1,3}})\b",
    ):
        for match in re.finditer(pattern, lower):
            constraints.append(_age_bound(text, match, "lt", float(match.group(1))))
    for pattern in (
        rf"\b(\d{{1,3}})\s*{year}\s*(?:or\s+)?(?:younger|less|under)\b",
        rf"\b{age_cue}\s*(\d{{1,3}})\s*(?:or\s+)?(?:younger|less|under)\b",
    ):
        for match in re.finditer(pattern, lower):
            constraints.append(_age_bound(text, match, "le", float(match.group(1))))
    if not constraints and re.search(r"\badults?\b", lower):
        match = re.search(r"\badults?\b", lower)
        start, end = match.span() if match else (None, None)
        constraints.append(
            Constraint(
                kind="age",
                label="adult",
                comparator="ge",
                value=18.0,
                unit="years",
                confidence=0.7,
                evidence_text=text[start:end] if start is not None else None,
                evidence_start=start,
                evidence_end=end,
            )
        )
    return constraints


def _age_bound(text: str, match: re.Match[str], comparator: str, value: float) -> Constraint:
    start, end = match.span()
    return Constraint(
        kind="age",
        label="age",
        comparator=comparator,  # type: ignore[arg-type]
        value=value,
        unit="years",
        evidence_text=text[start:end],
        evidence_start=start,
        evidence_end=end,
    )


def _sex_constraints(text: str) -> list[Constraint]:
    lower = text.casefold()
    # In a pregnancy/contraception criterion, female/male describes the condition, not a sex restriction;
    # emitting sex==female here would trip the near-ubiquitous pregnancy exclusion for every woman.
    if re.search(r"\b(pregnan|breast[\s-]?feed|lactat|contracept|childbearing)", lower):
        return []
    constraints: list[Constraint] = []
    female = re.search(r"\b(female|females|woman|women)\b", lower)
    male = re.search(r"\b(male|males|man|men)\b", lower)
    if female and not male:
        constraints.append(_sex_constraint(text, female, "female"))
    elif male and not female:
        constraints.append(_sex_constraint(text, male, "male"))
    return constraints


def _sex_constraint(text: str, match: re.Match[str], value: str) -> Constraint:
    start, end = match.span()
    return Constraint(
        kind="sex",
        label="sex",
        comparator="eq",
        value=value,
        evidence_text=text[start:end],
        evidence_start=start,
        evidence_end=end,
    )


def _performance_constraints(text: str) -> list[Constraint]:
    constraints: list[Constraint] = []
    lower = text.casefold()
    for match in re.finditer(
        r"\becog(?:\s+performance\s+status)?(?:\s+of)?\s*(\d)\s*(?:-|to|or)\s*(\d)\b",
        lower,
    ):
        start, end = match.span()
        constraints.append(
            Constraint(
                kind="performance_status",
                label="ECOG",
                comparator="between",
                min_value=float(match.group(1)),
                max_value=float(match.group(2)),
                evidence_text=text[start:end],
                evidence_start=start,
                evidence_end=end,
            )
        )
    for match in re.finditer(
        r"\becog(?:\s+performance\s+status)?\s*(?:<=|≤|no\s+greater\s+than|up\s+to)\s*(\d)\b",
        lower,
    ):
        start, end = match.span()
        constraints.append(
            Constraint(
                kind="performance_status",
                label="ECOG",
                comparator="le",
                value=float(match.group(1)),
                evidence_text=text[start:end],
                evidence_start=start,
                evidence_end=end,
            )
        )
    for match in re.finditer(
        r"\bkarnofsky\b.*?(?:>=|≥|at\s+least)\s*(\d{2,3})\b",
        lower,
    ):
        start, end = match.span()
        constraints.append(
            Constraint(
                kind="performance_status",
                label="Karnofsky",
                comparator="ge",
                value=float(match.group(1)),
                evidence_text=text[start:end],
                evidence_start=start,
                evidence_end=end,
            )
        )
    return constraints


def _lab_constraints(text: str) -> list[Constraint]:
    constraints: list[Constraint] = []
    aliases = "|".join(re.escape(alias) for alias in LAB_ALIASES)
    pattern = re.compile(
        rf"\b({aliases})\b(?:[^0-9<>≤≥=]{{0,40}})(>=|<=|≥|≤|>|<|=|at least|greater than|less than)?\s*([0-9]+(?:,[0-9]+)*(?:\.[0-9]+)?)\s*([A-Za-z/%0-9^µμ.-]+)?",
        re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        # Require an explicit comparator: bare "creatinine 1.5" is directionally ambiguous.
        if not match.group(2):
            continue
        label = _canonical_lab_label(match.group(1))
        comparator = _normalize_comparator(match.group(2))
        start, end = match.span()
        constraints.append(
            Constraint(
                kind="lab",
                label=label,
                comparator=comparator,
                # Strip thousands separators ("1,500" -> 1500) so comma-grouped thresholds keep their magnitude.
                value=float(match.group(3).replace(",", "")),
                unit=(match.group(4) or "").strip() or None,
                evidence_text=text[start:end],
                evidence_start=start,
                evidence_end=end,
            )
        )
    return constraints


def _canonical_lab_label(value: str) -> str:
    normalized = value.casefold()
    if normalized == "anc":
        return "absolute neutrophil count"
    if normalized == "platelets":
        return "platelet count"
    return normalized


def _normalize_comparator(value: str) -> str:
    normalized = value.strip().casefold()
    if normalized in {">=", "≥", "at least"}:
        return "ge"
    if normalized in {">", "greater than", "more than", "above"}:
        return "gt"  # exclusive: a boundary value does not satisfy "greater than"
    if normalized in {"<=", "≤"}:
        return "le"
    if normalized in {"<", "less than"}:
        return "lt"
    if normalized == "=":
        return "eq"
    return "ge"


def _biomarker_constraints(text: str) -> list[Constraint]:
    constraints: list[Constraint] = []
    gene_pattern = "|".join(re.escape(gene) for gene in GENE_SYMBOLS)
    pattern = re.compile(
        rf"\b({gene_pattern})\b(?:[-\s]*(mutated|mutation|positive|negative|wild[-\s]?type|amplification|overexpression))?",
        re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        # Require an explicit status token: a bare gene in a drug phrase ("EGFR inhibitor") isn't a biomarker requirement.
        if not match.group(2):
            continue
        status = (match.group(2) or "").casefold().replace(" ", "-")
        comparator = "present"
        if status in {"mutated", "mutation"}:
            comparator = "mutated"
        elif status in {"positive", "amplification", "overexpression"}:
            comparator = "positive"
        elif status == "negative":
            comparator = "negative"
        elif status in {"wild-type", "wildtype"}:
            comparator = "wildtype"
        start, end = match.span()
        constraints.append(
            Constraint(
                kind="biomarker",
                label=match.group(1).upper(),
                comparator=comparator,  # type: ignore[arg-type]
                evidence_text=text[start:end],
                evidence_start=start,
                evidence_end=end,
                confidence=0.9 if comparator != "present" else 0.75,
            )
        )
    return constraints


def _prior_therapy_constraints(text: str) -> list[Constraint]:
    constraints: list[Constraint] = []
    patterns = (
        r"\bprior\s+(?:treatment|therapy)\s+with\s+([^.;,]+)",
        r"\bpreviously\s+treated\s+with\s+([^.;,]+)",
        r"\breceived\s+prior\s+([^.;,]+)",
        r"\bprior\s+investigational\s+therapy\b",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            if match.lastindex:
                label = match.group(1)
            else:
                label = "investigational therapy"
            start, end = match.span()
            constraints.append(
                Constraint(
                    kind="medication",
                    label=_clean_label(label),
                    comparator="prior",
                    evidence_text=text[start:end],
                    evidence_start=start,
                    evidence_end=end,
                    confidence=0.8,
                )
            )
    return constraints


def _temporal_constraints(text: str) -> list[Constraint]:
    constraints: list[Constraint] = []
    for match in re.finditer(
        r"\bwithin\s+(?:the\s+last\s+)?(\d+)\s+(days?|weeks?|months?|years?)\b",
        text,
        re.IGNORECASE,
    ):
        start, end = match.span()
        constraints.append(
            Constraint(
                kind="temporal",
                label="time window",
                comparator="current",
                value=float(match.group(1)),
                unit=match.group(2).lower(),
                temporal_window=text[start:end],
                evidence_text=text[start:end],
                evidence_start=start,
                evidence_end=end,
                confidence=0.7,
            )
        )
    return constraints


def _entity_constraints(entities: Any, text: str) -> list[Constraint]:
    if not isinstance(entities, list):
        return []
    constraints: list[Constraint] = []
    for entity in entities:
        if not isinstance(entity, Mapping):
            continue
        group = str(entity.get("entity_group") or entity.get("class") or "").casefold()
        label = _clean_label(entity.get("text") or entity.get("entity") or "")
        if not label:
            continue
        kind = _entity_group_to_kind(group)
        if kind is None:
            continue
        start = _safe_int(entity.get("start"))
        end = _safe_int(entity.get("end"))
        constraints.append(
            Constraint(
                kind=kind,
                label=label,
                comparator=_entity_comparator(kind, text),
                normalized_codes=_entity_codes(entity),
                evidence_text=text[start:end] if start is not None and end is not None else label,
                evidence_start=start,
                evidence_end=end,
                confidence=float(entity.get("score") or 0.8),
            )
        )
    return constraints


def _entity_group_to_kind(group: str) -> str | None:
    if group in {"disease", "condition", "sign symptom", "sign_symptom"}:
        return "condition"
    if group in {"drug", "medication"}:
        return "medication"
    if group == "procedure":
        return "procedure"
    if group == "gene":
        return "biomarker"
    if group in {"laboratory test", "laboratory_test", "diagnostic test", "diagnostic_test"}:
        return "lab"
    return None


def _entity_comparator(kind: str, text: str) -> str:
    if kind == "biomarker":
        lower = text.casefold()
        if re.search(r"\b(mutated|mutation)\b", lower):
            return "mutated"
        if re.search(r"\b(positive|amplification|overexpression)\b", lower):
            return "positive"
        if re.search(r"\bnegative\b", lower):
            return "negative"
    if kind == "medication" and re.search(r"\b(prior|previous|previously)\b", text, re.IGNORECASE):
        return "prior"
    return "present"


def _entity_codes(entity: Mapping[str, Any]) -> list[dict[str, Any]]:
    codes = entity.get("normalized_id") or []
    if isinstance(codes, str):
        codes = [codes]
    output: list[dict[str, Any]] = []
    for code in codes:
        text = str(code)
        if not text or text == "CUI-less":
            continue
        if ":" in text:
            vocabulary, value = text.split(":", 1)
        else:
            vocabulary, value = "local", text
        output.append({"vocabulary": vocabulary, "code": value})
    return output


def _dedupe_constraints(constraints: list[Constraint]) -> list[Constraint]:
    seen: set[tuple[Any, ...]] = set()
    output: list[Constraint] = []
    for constraint in constraints:
        # Include codes in the key so same-label constraints differing only in normalized_codes aren't collapsed.
        codes_sig = tuple(
            sorted(
                (str(c.get("vocabulary", "")), str(c.get("code", "")))
                for c in (constraint.normalized_codes or [])
            )
        )
        key = (
            constraint.kind,
            constraint.label.casefold(),
            constraint.comparator,
            constraint.value,
            constraint.min_value,
            constraint.max_value,
            codes_sig,
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(constraint)
    return output


def _clean_label(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip(" .;,:")


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
