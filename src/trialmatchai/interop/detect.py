from __future__ import annotations

import json
from pathlib import Path
from typing import Literal


PatientInputFormat = Literal["text", "phenopacket", "fhir", "fhir-ndjson", "omop"]

OMOP_TABLE_NAMES = {
    "person",
    "condition_occurrence",
    "measurement",
    "drug_exposure",
    "procedure_occurrence",
    "observation",
    "note",
    "note_nlp",
}


def detect_patient_input_format(path: str | Path) -> PatientInputFormat:
    candidate = Path(path)
    if candidate.is_dir():
        table_stems = {
            item.stem.casefold()
            for item in candidate.iterdir()
            if item.suffix.casefold() in {".csv", ".parquet"}
        }
        if table_stems & OMOP_TABLE_NAMES:
            return "omop"
        raise ValueError(f"Could not detect patient input format for directory: {path}")

    suffix = candidate.suffix.casefold()
    if suffix in {".txt", ".md"}:
        return "text"
    if suffix == ".ndjson":
        return "fhir-ndjson"
    if suffix not in {".json", ".jsonl"}:
        raise ValueError(f"Unsupported patient input file extension: {suffix}")

    with candidate.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        if data.get("resourceType") == "Bundle":
            return "fhir"
        if data.get("resourceType"):
            return "fhir"
        if "metaData" in data and ("subject" in data or "phenotypicFeatures" in data):
            return "phenopacket"
        if {"id", "subject"} <= set(data):
            return "phenopacket"
    if isinstance(data, list) and any(
        isinstance(item, dict) and item.get("resourceType") for item in data
    ):
        return "fhir"
    raise ValueError(f"Could not detect patient input format for file: {path}")
