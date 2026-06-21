from __future__ import annotations

import re
from typing import Any


def normalize_study(study: dict[str, Any]) -> dict[str, Any]:
    """Normalize a ClinicalTrials.gov v2 study into TrialMatchAI trial JSON."""
    protocol = _mapping(study.get("protocolSection"))
    identification = _mapping(protocol.get("identificationModule"))
    status = _mapping(protocol.get("statusModule"))
    description = _mapping(protocol.get("descriptionModule"))
    conditions = _mapping(protocol.get("conditionsModule"))
    design = _mapping(protocol.get("designModule"))
    eligibility = _mapping(protocol.get("eligibilityModule"))
    interventions = _mapping(protocol.get("armsInterventionsModule"))
    contacts_locations = _mapping(protocol.get("contactsLocationsModule"))
    references = _mapping(protocol.get("referencesModule"))

    nct_id = _text(identification.get("nctId"))
    if not nct_id:
        raise ValueError("ClinicalTrials.gov study is missing protocolSection.identificationModule.nctId")

    criteria_text = _multiline_text(eligibility.get("eligibilityCriteria"))
    condition_values = _string_list(conditions.get("conditions"))
    normalized = {
        "nct_id": nct_id,
        "source": "clinicaltrials.gov",
        "source_url": f"https://clinicaltrials.gov/study/{nct_id}",
        "brief_title": _text(identification.get("briefTitle")),
        "official_title": _text(identification.get("officialTitle")),
        "brief_summary": _text(description.get("briefSummary")),
        "detailed_description": _text(description.get("detailedDescription")),
        "condition": condition_values,
        "eligibility_criteria": criteria_text,
        "criteria": split_eligibility_criteria(criteria_text),
        "overall_status": _text(status.get("overallStatus")),
        "phase": _join(_string_list(design.get("phases"))),
        "study_type": _text(design.get("studyType")),
        "gender": _text(eligibility.get("sex")) or "All",
        "minimum_age": _text(eligibility.get("minimumAge")),
        "maximum_age": _text(eligibility.get("maximumAge")),
        "start_date": _date_struct(status.get("startDateStruct")),
        "completion_date": _date_struct(status.get("completionDateStruct")),
        "last_update_posted": _date_struct(status.get("lastUpdatePostDateStruct")),
        "intervention": _intervention_rows(interventions.get("interventions")),
        "location": _location_rows(contacts_locations.get("locations")),
        "reference": _reference_rows(references.get("references")),
    }
    return {key: value for key, value in normalized.items() if value not in (None, "", [])}


def split_eligibility_criteria(text: str) -> list[dict[str, str]]:
    if not text or not text.strip():
        return []

    current_type = "unknown"
    entries: list[dict[str, str]] = []
    buffered: list[str] = []

    def flush() -> None:
        if not buffered:
            return
        criterion = " ".join(buffered).strip()
        buffered.clear()
        if _is_useful_criterion(criterion):
            entries.append({"type": current_type, "criterion": criterion})

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            flush()
            continue
        detected_type = _detect_criteria_header(line)
        if detected_type:
            flush()
            current_type = detected_type
            continue
        cleaned = _clean_criterion_line(line)
        if not cleaned:
            flush()
            continue
        if _starts_new_criterion(line):
            flush()
            buffered.append(cleaned)
        else:
            buffered.append(cleaned)

    flush()
    if entries:
        return entries

    fallback = _clean_criterion_line(text)
    return [{"type": "unknown", "criterion": fallback}] if fallback else []


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return " ".join(value.split())
    return str(value)


def _multiline_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in value.splitlines()]
        return "\n".join(lines).strip()
    return _text(value)


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [_text(item) for item in value if _text(item)]
    text = _text(value)
    return [text] if text else []


def _join(values: list[str]) -> str:
    return ", ".join(values)


def _date_struct(value: Any) -> str:
    if isinstance(value, dict):
        return _text(value.get("date"))
    return _text(value)


def _intervention_rows(value: Any) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in value if isinstance(value, list) else []:
        if not isinstance(item, dict):
            continue
        row = {
            "name": _text(item.get("name")),
            "type": _text(item.get("type")),
            "description": _text(item.get("description")),
        }
        rows.append({key: val for key, val in row.items() if val})
    return rows


def _location_rows(value: Any) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in value if isinstance(value, list) else []:
        if not isinstance(item, dict):
            continue
        row = {
            "facility": _text(item.get("facility")),
            "city": _text(item.get("city")),
            "state": _text(item.get("state")),
            "country": _text(item.get("country")),
            "status": _text(item.get("status")),
        }
        rows.append({key: val for key, val in row.items() if val})
    return rows


def _reference_rows(value: Any) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in value if isinstance(value, list) else []:
        if not isinstance(item, dict):
            continue
        row = {
            "pmid": _text(item.get("pmid")),
            "type": _text(item.get("type")),
            "citation": _text(item.get("citation")),
        }
        rows.append({key: val for key, val in row.items() if val})
    return rows


def _detect_criteria_header(line: str) -> str | None:
    normalized = re.sub(r"[^a-z]+", " ", line.casefold()).strip()
    if normalized in {"inclusion criteria", "inclusion"}:
        return "inclusion"
    if normalized in {"exclusion criteria", "exclusion"}:
        return "exclusion"
    return None


def _starts_new_criterion(line: str) -> bool:
    return bool(re.match(r"^\s*(?:[-*•]|\d+[\).]|[a-zA-Z][\).])\s+", line))


def _clean_criterion_line(line: str) -> str:
    cleaned = re.sub(r"^\s*(?:[-*•]|\d+[\).]|[a-zA-Z][\).])\s+", "", line)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" :-\t")
    if _detect_criteria_header(cleaned):
        return ""
    return cleaned


def _is_useful_criterion(text: str) -> bool:
    if len(text) < 3:
        return False
    if _detect_criteria_header(text):
        return False
    return True
