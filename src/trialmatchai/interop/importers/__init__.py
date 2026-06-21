from __future__ import annotations

from pathlib import Path
from typing import Any

from trialmatchai.interop.detect import detect_patient_input_format
from trialmatchai.interop.importers.fhir import import_fhir
from trialmatchai.interop.importers.omop import import_omop_extract
from trialmatchai.interop.importers.phenopacket import import_phenopacket
from trialmatchai.interop.importers.text import import_text_note
from trialmatchai.interop.models import PatientProfile


def import_patient_path(
    path: str | Path,
    *,
    input_format: str = "auto",
    entity_annotator: Any | None = None,
    strict: bool = False,
) -> list[PatientProfile]:
    resolved_format = (
        detect_patient_input_format(path) if input_format == "auto" else input_format
    )
    if resolved_format == "text":
        return [import_text_note(path, entity_annotator=entity_annotator)]
    if resolved_format == "phenopacket":
        return [import_phenopacket(path, strict=strict)]
    if resolved_format in {"fhir", "fhir-ndjson"}:
        return [import_fhir(path, input_format=resolved_format, strict=strict)]
    if resolved_format == "omop":
        return import_omop_extract(path, strict=strict)
    raise ValueError(f"Unsupported patient input format: {resolved_format}")


__all__ = ["import_patient_path"]
