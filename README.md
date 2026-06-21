# TrialMatchAI

<img src="img/logo.png" alt="Logo" align="right" width="200" height="200">

TrialMatchAI is a batch-oriented clinical trial matching pipeline. It combines local LanceDB retrieval, schema-driven biomedical entity extraction, concept linking, embeddings, LLM reranking, and eligibility reasoning to produce ranked trial recommendations with criterion-level explanations.

## Disclaimer

This software is for research and informational use only. It is not medical advice, is not a medical device, and must not replace review by qualified healthcare professionals.

## Deployment Target

The supported v1 deployment path is a single Python 3.11 GPU server or VM. Trial and criteria search use embedded LanceDB tables under `data/search`, so no separate search service, container, socket, TLS certificate, or service credential is required. Docker remains optional for packaging the worker.

## Requirements

- Python 3.11
- `uv` recommended, or `pip` with editable install
- NVIDIA GPU with enough VRAM for the selected LLM backend
- 100 GB+ disk space for datasets, models, LanceDB tables, and results
- A LanceDB concept table built from OMOP/legacy dictionaries for entity normalization

## Security First

No real credentials, generated private keys, datasets, models, local LanceDB data, run manifests, or results should be committed. Copy the template and keep runtime values local:

```bash
cp .env.example .env
```

Dependency auditing currently ignores `CVE-2025-3000` because vLLM 0.23 pins Torch 2.11.0 and the advisory has no fixed Torch version listed. Revisit that exception whenever upgrading vLLM or Torch.

## Quickstart

Install the package for local development and operational CLIs:

```bash
uv sync
```

Install deployment dependencies for model-backed indexing, entity extraction, and vLLM reasoning:

```bash
uv sync --extra llm --extra gpu --extra entity
```

Optional tooling:

```bash
uv sync --extra entity    # GLiNER/GLiNER2 entity extraction
uv sync --extra llm       # local embedding and LLM stack
```

Run a config and backend healthcheck:

```bash
uv run trialmatchai-healthcheck
uv run trialmatchai-healthcheck --require-tables
```

Provision data, models, concept KB, and search tables:

```bash
uv run trialmatchai-bootstrap-data
uv run trialmatchai-build-concepts --concept-csv data/omop/CONCEPT.csv --synonym-csv data/omop/CONCEPT_SYNONYM.csv
uv run trialmatchai-update-registry --since 2026-06-01 --max-studies 100
uv run trialmatchai-index --prepare
```

Run the batch matcher:

```bash
uv run trialmatchai-run
```

Results are written under `results/`.

## Docker Worker

Docker is optional. The worker container uses mounted local folders and the same embedded LanceDB tables:

```bash
docker compose build trialmatchai-worker
docker compose run --rm trialmatchai-worker trialmatchai-healthcheck
docker compose run --rm trialmatchai-worker trialmatchai-update-registry --max-studies 100
docker compose run --rm trialmatchai-worker trialmatchai-run
```

## Configuration

Configuration defaults live in `src/trialmatchai/config/config.json`. Runtime overrides use `.env` or environment variables:

```bash
TRIALMATCHAI_PATIENTS_DIR=example
TRIALMATCHAI_OUTPUT_DIR=results
TRIALMATCHAI_TRIALS_JSON_FOLDER=data/trials_jsons

TRIALMATCHAI_SEARCH_BACKEND=lancedb
TRIALMATCHAI_SEARCH_DB_PATH=data/search
TRIALMATCHAI_SEARCH_TRIALS_TABLE=trials
TRIALMATCHAI_SEARCH_CRITERIA_TABLE=criteria
TRIALMATCHAI_SEARCH_CANDIDATE_LIMIT=1000
TRIALMATCHAI_SEARCH_MODE=hybrid

TRIALMATCHAI_MODEL_TRUST_REMOTE_CODE=false
TRIALMATCHAI_ENTITY_BACKEND=gliner2
TRIALMATCHAI_ENTITY_SCHEMA_PATH=src/trialmatchai/entity_schemas/trialmatchai.yaml
TRIALMATCHAI_CONCEPT_DB_PATH=data/concepts
TRIALMATCHAI_CONCEPT_TABLE=concepts
TRIALMATCHAI_LINK_ACCEPT=0.80
TRIALMATCHAI_LINK_REJECT=0.30

TRIALMATCHAI_REGISTRY_SOURCE=clinicaltrials.gov
TRIALMATCHAI_REGISTRY_KEYWORDS_FILE=
TRIALMATCHAI_REGISTRY_SINCE_DAYS=7
TRIALMATCHAI_REGISTRY_MAX_STUDIES=
TRIALMATCHAI_REGISTRY_REQUEST_TIMEOUT=30
TRIALMATCHAI_REGISTRY_RATE_LIMIT_PER_SECOND=2
TRIALMATCHAI_REGISTRY_RAW_DIR=data/registry/raw
TRIALMATCHAI_REGISTRY_MANIFEST_PATH=data/registry/manifest.jsonl
TRIALMATCHAI_REGISTRY_REPORTS_DIR=data/registry/runs
TRIALMATCHAI_LOG_JSON=1
```

Use `TRIALMATCHAI_MODEL_TRUST_REMOTE_CODE=true` only when a selected model explicitly requires custom remote code.

## CLI Commands

- `trialmatchai-healthcheck`: validate config, paths, and optionally LanceDB search tables.
- `trialmatchai-bootstrap-data`: download and extract external data/model artifacts.
- `trialmatchai-build-concepts`: build the LanceDB concept table used for entity normalization.
- `trialmatchai-update-registry`: fetch new/changed ClinicalTrials.gov studies, write normalized JSON, and upsert LanceDB.
- `trialmatchai-index`: build the LanceDB trial and criteria search tables from prepared rows or `--prepare` normalized JSON.
- `trialmatchai-run`: run the batch matching pipeline.

The command group is also available as:

```bash
uv run python -m trialmatchai healthcheck
uv run trialmatchai update-registry --dry-run --max-studies 25
```

## Registry Updater

The registry updater is designed for cron, systemd timers, and GitHub Actions. It uses ClinicalTrials.gov v2 `/api/v2/studies`, stores raw source JSON under `data/registry/raw`, writes normalized trial JSON under `data/trials_jsons`, appends idempotency records to `data/registry/manifest.jsonl`, and upserts only new or changed studies into LanceDB.

Run with explicit keywords:

```bash
uv run trialmatchai-update-registry \
  --keyword "lung cancer" \
  --keyword "EGFR" \
  --since 2026-06-01 \
  --max-studies 250
```

Run with broad defaults:

```bash
uv run trialmatchai-update-registry --max-studies 500
```

Dry-run without writes or table changes:

```bash
uv run trialmatchai-update-registry --dry-run --max-studies 25
```

See `docs/registry-updater.md` for scheduler examples.

## Tests and Checks

```bash
uv run ruff check .
uv run pytest
uv run python scripts/scan_secrets.py
uv run pip-audit --progress-spinner off --ignore-vuln CVE-2025-3000
docker compose config
docker build .
```

## Support

- Email: abdallahmajd7@gmail.com
- DOI: https://doi.org/10.5281/zenodo.18329084
- arXiv: https://arxiv.org/abs/2505.08508
