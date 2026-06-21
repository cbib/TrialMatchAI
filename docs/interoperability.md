# Patient Interoperability

TrialMatchAI uses a canonical `PatientProfile` for patient data. Source-specific
importers preserve raw evidence and provenance, then exporters render the profile
for matching, LLM context, and optional standards output.

## Supported v1 Inputs

- Free text notes: `.txt` and `.md` files.
- GA4GH Phenopacket JSON.
- HL7 FHIR R4 Bundle JSON, individual FHIR resource JSON, and NDJSON.
- OMOP CDM extract folders with CSV or Parquet tables.

The first import milestone intentionally does not implement C-CDA, HL7 v2,
DICOMweb, or CDISC adapters. Those formats should be added as importers on top of
the same `PatientProfile` model, not as new runtime data models.

## Canonical Profile

Each imported patient is written to `data/patients/profiles/<patient_id>.json`.
The profile contains:

- demographics
- conditions
- phenotypes
- observations, labs, and vitals
- medications
- procedures
- diagnostic reports
- genomic findings
- cancer profile fields
- family history
- notes
- source documents
- provenance and unsupported source elements

Every clinical fact carries source format, source path, source resource or table,
original code, normalized code candidates, evidence text and offsets when
available, confidence, negation, temporality, and mapping status.

## Narrative Generation

Narrative rendering is deliberately separate from import. Importers create
structured facts. `trialmatchai.interop.narrative` turns those facts into
deterministic sentences for retrieval and LLM context. This keeps source parsing,
normalization, and prompt context generation independently testable.

The matching exporter writes `data/patients/summaries/<patient_id>.json` with the
runtime summary fields:

- `main_conditions`
- `other_conditions`
- `expanded_sentences`
- `age`
- `gender`
- `provenance`

`trialmatchai-run` consumes canonical profiles from `patient_inputs.profile_dir`.
Import source files explicitly with `trialmatchai-import-patient` before running
matching.

## Commands

Import a free text note:

```bash
uv run trialmatchai-import-patient \
  --input data/patients/raw/patient-1.txt \
  --format text
```

Import a Phenopacket:

```bash
uv run trialmatchai import-patient \
  --input data/patients/raw/patient-1.phenopacket.json
```

Import a FHIR Bundle:

```bash
uv run trialmatchai-import-patient \
  --input data/patients/raw/patient-1.fhir.json
```

Import an OMOP extract folder:

```bash
uv run trialmatchai-import-patient \
  --input data/patients/omop_extract \
  --format omop
```

Then run matching:

```bash
uv run trialmatchai-run
```

## Conversion Limits

FHIR and Phenopacket exports are best-effort views of `PatientProfile`.
Unsupported or lossy fields remain in the profile and are reported instead of
being silently dropped. OMOP support is file-based for v1; live database
connectors are out of scope.
