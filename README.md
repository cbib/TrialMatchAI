# TrialMatchAI

<img src="img/logo.png" alt="TrialMatchAI logo" align="right" width="170" height="170">

TrialMatchAI is an AI-driven clinical trial matching pipeline. It imports patient
data, retrieves relevant trials from local LanceDB tables, and produces ranked
trial recommendations with criterion-level eligibility explanations.

The supported deployment is a single Python 3.11 GPU server. Trial search is
embedded and file-backed; there is no Elasticsearch, hosted vector database, or
separate search service to run.

[Install](#install) | [Quickstart](#quickstart) | [How It Works](#how-it-works) | [Configuration](#configuration) | [CLI](#cli-reference)

> For research and informational use only. TrialMatchAI is not medical advice,
> not a medical device, and must not replace review by qualified healthcare
> professionals.

## Requirements

- Python 3.11 (`pyproject.toml` requires `>=3.11,<3.12`)
- `uv` recommended, or `pip` with an editable install
- NVIDIA GPU for vLLM-backed matching and fine-tuning
- Around 100 GB disk for datasets, model artifacts, LanceDB tables, manifests,
  and run outputs
- OMOP vocabulary files if you want to build the concept-linking table locally

## Install

Clone the repository and install from the project root:

```bash
git clone <repo-url>
cd TrialMatchAI
```

Base install with `uv` gives the package and CLI entry points without the heavy
model stack:

```bash
uv sync
uv run trialmatchai --help
```

Install the full model-backed runtime:

```bash
uv sync --extra llm --extra gpu --extra entity
```

Install the fine-tuning stack:

```bash
uv sync --extra finetune
```

Editable install with `pip` is also supported:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional extras with `pip`:

```bash
pip install -e ".[entity]"
pip install -e ".[llm,entity]"
pip install -e ".[llm,gpu,entity]"
pip install -e ".[finetune]"
```

| Extra | Adds |
| --- | --- |
| `entity` | GLiNER2 biomedical NER |
| `llm` | local embedding and LLM dependencies |
| `gpu` | vLLM and bitsandbytes; intended for Linux CUDA hosts |
| `finetune` | training dependencies for `trialmatchai-finetune` |

Installing the package is only the first step. Real matching also needs runtime
data, model artifacts, a concept table, normalized trials, and LanceDB search
tables.

## Quickstart

Copy the environment template if you need local overrides:

```bash
cp .env.example .env
```

Check the installation and configured paths:

```bash
uv run trialmatchai-healthcheck
```

Download packaged artifacts:

```bash
uv run trialmatchai-bootstrap-data
```

Build the concept-linking table from OMOP vocabulary files:

```bash
uv run trialmatchai-build-concepts \
  --concept-csv data/omop/CONCEPT.csv \
  --synonym-csv data/omop/CONCEPT_SYNONYM.csv
```

Fetch recent ClinicalTrials.gov studies and update local trial JSON plus
LanceDB:

```bash
uv run trialmatchai-update-registry --since 2026-06-01 --max-studies 100
```

Build or rebuild search tables from normalized trial JSON:

```bash
uv run trialmatchai-index --prepare
```

Import patients into canonical TrialMatchAI profiles:

```bash
uv run trialmatchai-import-patient --input data/patients/raw/patient-1.txt --format text
uv run trialmatchai-import-patient --input data/patients/raw/patient-1.fhir.json
uv run trialmatchai-import-patient --input data/patients/omop_extract --format omop
```

Run batch matching:

```bash
uv run trialmatchai-run
```

Results are written under `results/<patient_id>/`.

## How It Works

```text
Patient data (text / FHIR / Phenopacket / OMOP)
      |
      v
Interop importers -> canonical PatientProfile
      |
      v
GLiNER2 NER + deterministic variant patterns -> concept linking
      |
      v
First-level trial retrieval in LanceDB (BM25 + embeddings)
      |
      v
Multi-channel query fusion for broad candidate recall
      |
      v
Criterion retrieval + vLLM Yes/No reranker
      |
      v
Constraint-aware criterion scoring
      |
      v
vLLM eligibility reasoning per criterion
      |
      v
Final ranking + explanations in results/
```

The generative LLM stages, reranker and eligibility reasoning, run on vLLM.
LoRA adapters are served natively through vLLM. NER, reranker, and eligibility
reasoning are configurable and fine-tunable.

## Data And Storage

TrialMatchAI uses embedded LanceDB tables by default:

- Search DB: `data/search`
- Trial table: `trials`
- Criteria table: `criteria`
- Concept-linking DB: `data/concepts`
- Concept table: `concepts`

ClinicalTrials.gov records are normalized to JSON files under
`data/trials_jsons/<NCT_ID>.json`. During indexing, TrialMatchAI prepares:

- one trial row per NCT ID, including text fields, metadata filters, date/age
  fields, and embedding vectors
- one criteria row per eligibility criterion, including criterion text,
  criterion embedding, eligibility type, entity annotations, and parsed
  eligibility constraints

The trial and criteria tables each get full-text search fields and vector
columns, so retrieval can run in `bm25`, `vector`, or `hybrid` mode.

Patient inputs are imported before matching. Each imported patient is stored as
a canonical profile under `data/patients/profiles/<patient_id>.json`, with a
matching summary under `data/patients/summaries/<patient_id>.json`.

## Patient Inputs

The importer supports:

- free-text notes: `.txt` and `.md`
- GA4GH Phenopacket JSON
- HL7 FHIR R4 Bundle JSON, individual FHIR resource JSON, NDJSON, and JSONL
- OMOP CDM extract folders with CSV or Parquet tables

Importers preserve provenance and unsupported source elements where possible.
The matching summary is rendered deterministically from the canonical
`PatientProfile`; raw patient files are not consumed directly by
`trialmatchai-run`.

See [docs/interoperability.md](docs/interoperability.md) for format details.

## First-Level Retrieval

First-level retrieval is recall-oriented. It builds a multi-channel query plan
from the canonical `PatientProfile` and matching summary, then searches each
channel separately and fuses candidates with reciprocal rank fusion.

Channels include primary conditions, linked concept synonyms, broader disease
terms, patient narrative text, biomarkers, prior therapy or procedures, and
optional LLM-generated expansions. LLM expansion is off by default; deterministic
concept and patient-profile expansion are the default path.

The first level only hard-filters by age, sex/gender, and recruitment status.
Location, biomarkers, phase, prior therapy, and eligibility constraints remain
soft signals for later retrieval and reasoning stages.

When enabled, first-level artifacts are written under `results/<patient_id>/`:

- `first_level_query_plan.json`
- `first_level_candidates.json`

## Constraint-Aware Retrieval

TrialMatchAI parses common eligibility logic from criteria rows and compares it
with the canonical `PatientProfile` during second-stage retrieval. V1 supports
age, sex or gender, conditions, medications and prior therapy, procedures, labs,
biomarkers, ECOG/Karnofsky-style performance status, temporal phrases, and
inclusion/exclusion polarity.

Constraints are a soft ranking signal. They can boost matching inclusion
criteria, penalize violated inclusion or exclusion criteria, and leave unknown
facts neutral. They do not hard-exclude trials and they are not medical advice;
the final vLLM eligibility reasoning remains the final judge.

When enabled, per-patient reports are written under `results/<patient_id>/`:

- `constraint_evaluations.json`
- `constraint_summary.md`
- `top_trials_explained.json`

## Bring Your Own Models

Defaults are starting points. Point the pipeline at your own checkpoints or
adapters through config or environment variables.

| Component | Default | Config key |
| --- | --- | --- |
| Biomedical NER | `fastino/gliner2-base` | `entity_extraction.model_name` |
| Reranker | `google/gemma-2-2b-it` | `model.reranker_adapter_path` |
| CoT eligibility | configured CoT model | `model.cot_adapter_path` |

Fine-tune model components with:

```bash
uv sync --extra finetune
uv run trialmatchai-finetune cot \
  --base-model microsoft/phi-4 \
  --train-data data/finetune/cot.jsonl \
  --output-dir models/cot-adapter
uv run trialmatchai-finetune reranker \
  --base-model google/gemma-2-2b-it \
  --train-data data/finetune/reranker.jsonl \
  --output-dir models/reranker-adapter
uv run trialmatchai-finetune ner \
  --base-model fastino/gliner2-base-v1 \
  --train-data data/finetune/ner.jsonl \
  --output-dir models/ner
```

See [docs/finetuning.md](docs/finetuning.md) for accepted training formats and
adapter configuration.

## Configuration

Defaults live in `src/trialmatchai/config/config.json`. Runtime overrides can be
set in `.env` or as environment variables.

Common overrides:

```bash
TRIALMATCHAI_OUTPUT_DIR=results
TRIALMATCHAI_TRIALS_JSON_FOLDER=data/trials_jsons
TRIALMATCHAI_SEARCH_DB_PATH=data/search
TRIALMATCHAI_SEARCH_MODE=hybrid
TRIALMATCHAI_FIRST_LEVEL_MAX_TRIALS=1000
TRIALMATCHAI_FIRST_LEVEL_PER_CHANNEL_SIZE=300
TRIALMATCHAI_FIRST_LEVEL_VECTOR_SCORE_THRESHOLD=0.0
TRIALMATCHAI_FIRST_LEVEL_LLM_EXPANSION_ENABLED=false
TRIALMATCHAI_ENTITY_BACKEND=gliner2
TRIALMATCHAI_ENTITY_SCHEMA_PATH=entity_schemas/trialmatchai.yaml
TRIALMATCHAI_CONCEPT_DB_PATH=data/concepts
TRIALMATCHAI_LINK_ACCEPT=0.80
TRIALMATCHAI_LINK_REJECT=0.30
TRIALMATCHAI_CONSTRAINTS_ENABLED=true
TRIALMATCHAI_CONSTRAINTS_SCORE_WEIGHT=0.25
TRIALMATCHAI_CONSTRAINTS_LLM_EXTRACTION_ENABLED=false
TRIALMATCHAI_CONSTRAINTS_WRITE_REPORTS=true
TRIALMATCHAI_MODEL_TRUST_REMOTE_CODE=false
TRIALMATCHAI_LOG_JSON=1
```

The full override list is in [`.env.example`](.env.example).

## CLI Reference

| Command | Purpose |
| --- | --- |
| `trialmatchai` | Command group for the main subcommands |
| `trialmatchai-healthcheck` | Validate config, paths, optional model deps, and LanceDB tables |
| `trialmatchai-bootstrap-data` | Download and extract runtime data/model artifacts |
| `trialmatchai-build-concepts` | Build the LanceDB concept table for entity normalization |
| `trialmatchai-update-registry` | Fetch changed ClinicalTrials.gov studies and upsert LanceDB |
| `trialmatchai-index` | Prepare/index trial and criteria search tables |
| `trialmatchai-import-patient` | Import text, FHIR, Phenopacket, or OMOP patient data |
| `trialmatchai-run` | Run the batch matching pipeline |
| `trialmatchai-finetune` | Fine-tune NER, reranker, or eligibility reasoning models |

The first seven commands are also available as subcommands:

```bash
uv run trialmatchai healthcheck
uv run python -m trialmatchai healthcheck
```

## Deployment

The supported deployment is a single Python 3.11 GPU server or VM. Search tables
are local LanceDB files under `data/search`, and concept linking uses a separate
LanceDB database under `data/concepts`.

The registry updater is designed for cron, systemd timers, or GitHub Actions.
See [docs/registry-updater.md](docs/registry-updater.md).

## Development

```bash
uv sync
uv run ruff check .
uv run pytest
uv run python scripts/scan_secrets.py
uv run pip-audit --progress-spinner off --ignore-vuln CVE-2025-3000
```

## Security

Never commit real credentials, private keys, datasets, models, local LanceDB
data, run manifests, or results. Keep runtime values local:

```bash
cp .env.example .env
```

Artifact bootstrap supports optional SHA-256 verification through:

- `TRIALMATCHAI_PROCESSED_TRIALS_SHA256`
- `TRIALMATCHAI_MODELS_SHA256`
- `TRIALMATCHAI_CRITERIA_PART_<N>_SHA256`

Dependency auditing currently ignores `CVE-2025-3000` because vLLM 0.23 pins
Torch 2.11.0 and the advisory lists no fixed Torch version. Revisit this when
upgrading vLLM or Torch.

## Citation

If you use TrialMatchAI in your research, please cite the Nature Communications
paper:

> Abdallah, M. _et al._ TrialMatchAI. _Nature Communications_ (2026).
> <https://www.nature.com/articles/s41467-026-70509-w>

```bibtex
@article{trialmatchai,
  title   = {TrialMatchAI},
  author  = {Abdallah, Majd and others},
  journal = {Nature Communications},
  year    = {2026},
  doi     = {10.1038/s41467-026-70509-w},
  url     = {https://www.nature.com/articles/s41467-026-70509-w}
}
```

## Support

- Email: abdallahmajd7@gmail.com
- Software archive (DOI): <https://doi.org/10.5281/zenodo.18329084>
