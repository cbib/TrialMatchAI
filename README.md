# TrialMatchAI

<img src="img/logo.png" alt="Logo" align="right" width="200" height="200">

TrialMatchAI is a batch-oriented clinical trial matching pipeline. It combines Elasticsearch retrieval, biomedical NLP, embeddings, LLM reranking, and eligibility reasoning to produce ranked trial recommendations with criterion-level explanations.

## Disclaimer

This software is for research and informational use only. It is not medical advice, is not a medical device, and must not replace review by qualified healthcare professionals.

## Deployment Target

The supported v1 deployment path is a single Linux GPU server or VM with Docker Compose for Elasticsearch and a containerized TrialMatchAI worker. HPC/Apptainer support remains available through the scripts under `elasticsearch/`, but production runtime does not auto-start local services by default.

## Requirements

- Python 3.11
- `uv` recommended, or `pip` with editable install
- Docker Compose for the default Elasticsearch deployment
- NVIDIA GPU with enough VRAM for the selected LLM backend
- 100 GB+ disk space for datasets, models, indices, and results
- Java for BioMedNER/normalization components when those services are enabled

## Security First

No real credentials, generated TLS keys, Elasticsearch keystores, Parser outputs, or local indexing state should be committed. Copy templates and rotate any previously exposed credentials before deployment:

```bash
cp .env.example .env
cp elasticsearch/.env.example elasticsearch/.env
```

Set strong local values for `TRIALMATCHAI_ES_PASSWORD`, `ELASTIC_PASSWORD`, and `KIBANA_PASSWORD`.

Dependency auditing currently ignores `CVE-2025-3000` because vLLM 0.23 pins Torch 2.11.0 and the advisory has no fixed Torch version listed. Revisit that exception whenever upgrading vLLM or Torch.

## Quickstart

Install deployment dependencies:

```bash
uv sync --extra gpu
```

For local development, tests, healthchecks, or `TRIALMATCHAI_COT_BACKEND=default`, the default dependency set is enough:

```bash
uv sync
```

Optional tooling is split out of the default runtime:

```bash
uv sync --extra llm       # OpenAI/LangChain data-generation utilities
uv sync --extra training  # fine-tuning and evaluation utilities
```

Start Elasticsearch with the root Compose stack:

```bash
docker compose up -d elasticsearch
```

Run a healthcheck:

```bash
uv run trialmatchai-healthcheck
```

Provision data, models, and indices:

```bash
uv run trialmatchai-bootstrap-data
uv run trialmatchai-index
```

Run the batch matcher:

```bash
uv run trialmatchai-run
```

Results are written under `results/`.

## Docker Worker

Build and run the worker healthcheck through Compose:

```bash
docker compose build trialmatchai-worker
docker compose up trialmatchai-worker
```

To run the full pipeline in the container after provisioning data/models/indices:

```bash
docker compose run --rm trialmatchai-worker trialmatchai-run
```

## Configuration

Configuration defaults live in `source/Matcher/config/config.json`. Runtime overrides use `.env` or environment variables:

```bash
TRIALMATCHAI_ES_HOST=https://localhost:9200
TRIALMATCHAI_ES_USERNAME=elastic
TRIALMATCHAI_ES_PASSWORD=change-me
TRIALMATCHAI_ES_CA_CERTS=elasticsearch/certs/ca/ca.crt
TRIALMATCHAI_ES_AUTO_START=false

TRIALMATCHAI_PATIENTS_DIR=example
TRIALMATCHAI_OUTPUT_DIR=results
TRIALMATCHAI_TRIALS_JSON_FOLDER=data/trials_jsons
TRIALMATCHAI_INDEX_TRIALS=clinical_trials
TRIALMATCHAI_INDEX_TRIALS_ELIGIBILITY=trials_eligibility

TRIALMATCHAI_MODEL_TRUST_REMOTE_CODE=false
TRIALMATCHAI_BIOMEDNER_AUTO_START=false
TRIALMATCHAI_LOG_JSON=1
```

Use `TRIALMATCHAI_MODEL_TRUST_REMOTE_CODE=true` only when a selected model explicitly requires custom remote code.

## CLI Commands

- `trialmatchai-healthcheck`: validate config, paths, Elasticsearch reachability, and optionally indices.
- `trialmatchai-bootstrap-data`: download and extract external data/model artifacts.
- `trialmatchai-index`: index prepared data into Elasticsearch.
- `trialmatchai-run`: run the batch matching pipeline.

## Tests and Checks

```bash
uv run ruff check .
uv run pytest
uv run python scripts/scan_secrets.py
uv run pip-audit --progress-spinner off --ignore-vuln CVE-2025-3000
docker compose config
docker build .
```

Integration tests require a running Elasticsearch instance:

```bash
TRIALMATCHAI_RUN_INTEGRATION=1 uv run pytest -m integration
```

## Support

- Email: abdallahmajd7@gmail.com
- DOI: https://doi.org/10.5281/zenodo.18329084
- arXiv: https://arxiv.org/abs/2505.08508
