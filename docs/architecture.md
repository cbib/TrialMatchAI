# TrialMatchAI Architecture

TrialMatchAI is an installable Python package exposed as `trialmatchai`. The supported runtime code lives under `src/trialmatchai`.

## Runtime Subsystems

- `trialmatchai.config`: Pydantic-backed settings, path resolution, and `TRIALMATCHAI_` environment overrides.
- `trialmatchai.entities`: schema-driven biomedical entity recognition plus LanceDB-backed concept linking.
- `trialmatchai.registry`: ClinicalTrials.gov v2 client, record normalization, manifest idempotency, preparation, and update orchestration.
- `trialmatchai.search`: embedded LanceDB search tables for trials and eligibility criteria.
- `trialmatchai.matching`: first-stage retrieval, criteria retrieval, ranking, and eligibility reasoning.
- `trialmatchai.models`: local embedding and LLM loader utilities. Install with `--extra llm` and `--extra gpu` for production model runs.
- `trialmatchai.cli`: public command entry points.

## Data Flow

1. `trialmatchai update-registry` fetches studies from ClinicalTrials.gov v2.
2. Raw source JSON is written to `data/registry/raw/<NCT_ID>.json`.
3. Normalized trial JSON is written to `data/trials_jsons/<NCT_ID>.json`.
4. Changed studies are embedded, criteria are optionally entity annotated, and LanceDB trial/criteria tables are upserted.
5. `trialmatchai run` matches already-imported patient profiles against LanceDB, reranks candidate criteria/trials, and writes results. (Ingest raw patient inputs first with `trialmatchai import-patient`, or use `trialmatchai e2e` to ingest and match in one step.)

## Storage

LanceDB is the only search database. It is embedded, file-backed, and stored under `data/search` by default. The concept linker uses a separate LanceDB database under `data/concepts`.

The registry manifest is append-only JSONL at `data/registry/manifest.jsonl`. The latest record per `nct_id` determines idempotency.

## Public API

New code imports from the canonical package namespace:

```python
from trialmatchai.config.config_loader import load_config
```

The old `Matcher` namespace is not shipped.
