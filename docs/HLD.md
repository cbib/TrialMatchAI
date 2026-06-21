# High Level Design - TrialMatchAI

## 1. Overview

TrialMatchAI is a batch clinical trial matching system. Given a GA4GH Phenopacket, it retrieves candidate trials from embedded LanceDB tables, reranks eligibility criteria with an LLM, performs criterion-level reasoning, and writes ranked recommendations.

### Goals

- Match de-identified research patient profiles to relevant clinical trials.
- Keep deployment local-first and service-light for a single GPU server.
- Provide criterion-level explainability.
- Use ontology-grounded entity extraction and normalization.

### Non-Goals

- Real-time clinical web application deployment.
- Direct EHR integration.
- Use with identifiable patient data without additional controls.
- Trial outcome prediction.

## 2. Architecture

```
ClinicalTrials.gov JSONs
        |
        v
prepare_trials.py / prepare_criteria.py
        |
        v
Schema entity annotation + LanceDB concept linking
        |
        v
BGE-M3 embeddings
        |
        v
data/search LanceDB tables
        |
        v
Phenopacket -> keywords -> Stage 1 retrieval -> Stage 2 criteria rerank
        |
        v
Stage 3 eligibility reasoning -> ranked_trials.json
```

The search layer is embedded in the Python worker. There is no separate database service to start, secure, or monitor for the default v1 deployment.

## 3. Persistent Stores

### `data/search` Trial Table

One row per trial.

| Field | Notes |
|---|---|
| `nct_id` | Primary trial identifier |
| `brief_title`, `brief_summary`, `condition`, `eligibility_criteria` | Source text fields |
| `*_vector` | Prepared BGE-M3 vectors from the existing preparation pipeline |
| `search_text` | Flattened text used for local full-text candidate generation |
| `search_vector` | Averaged vector used for vector candidate generation |
| `minimum_age`, `maximum_age`, `gender`, `overall_status` | Runtime filters |

### `data/search` Criteria Table

One row per eligibility criterion.

| Field | Notes |
|---|---|
| `criteria_id` | Stable criterion identifier |
| `nct_id` | Parent trial identifier |
| `criterion` | Criterion text |
| `criterion_vector` | BGE-M3 criterion vector |
| `entities` | Schema annotations and concept-link candidates |
| `entity_text`, `entity_synonyms_text` | Flattened entity search fields |
| `search_text` | Combined criterion/entity text |

### `data/concepts` Concept Table

The concept table is built from OMOP `CONCEPT.csv`, `CONCEPT_SYNONYM.csv`, and optional legacy dictionaries. It is used only for entity normalization and synonym expansion.

## 4. Retrieval

### Stage 1 - Trial Retrieval

`ClinicalTrialSearch` calls the configured search backend with:

- Primary disease/condition terms.
- Synonyms from accepted concept links.
- Other patient conditions.
- Age, sex, status, and optional NCT filters.
- Query embeddings when running in `vector` or `hybrid` mode.

The LanceDB backend generates candidates with local full-text/vector search and applies Python-side weighted scoring:

```
hybrid_score = 0.5 * text_score + 0.5 * vector_score
```

Text scoring weights trial fields by clinical relevance:

- `condition`: 6.0
- `eligibility_criteria`: 4.0
- `brief_title`: 3.0
- `brief_summary`: 2.0
- `detailed_description`: 1.5
- `official_title`: 1.0

Vector scoring combines condition, title, summary, eligibility, and other-condition similarities.

### Stage 2 - Criteria Retrieval

`SecondStageRetriever` searches the criteria table within the Stage 1 NCT subset. It searches criterion text and, when the entity annotator is enabled, flattened entity synonyms.

The retriever returns ES-like hit dictionaries internally (`_source`, `_score`) to keep reranking and aggregation code stable while the backend is LanceDB-native.

### Stage 3 - Eligibility Reasoning

The top trials are passed to `BatchTrialProcessor` or `BatchTrialProcessorVLLM`. The model evaluates inclusion and exclusion criteria and writes structured JSON outputs. The final ranker computes a normalized eligibility score from those outputs.

## 5. Entity Extraction

The BioNER path is Python-native:

- Schema-driven recognizer interface.
- GLiNER2-style backend as the target recognizer.
- GLiNER fallback and regex test backend behind the same interface.
- LanceDB concept linking with confidence bands.
- No runtime Java daemons, socket IPC, Sieve, or GNormPlus dependency.

Entity output preserves the fields downstream search expects:

- `entity_group`
- `text`
- `start`
- `end`
- `score`
- `normalized_id`
- `synonyms`
- `concept_candidates`
- `linker_score`
- `linker_status`

## 6. Deployment

Default deployment requires only the Python worker process and local mounted directories:

- `data/`
- `models/`
- `results/`

The application runs as a Python process and does not auto-start external services.

## 7. Runtime Configuration

Important settings:

| Env var | Default | Purpose |
|---|---|---|
| `TRIALMATCHAI_SEARCH_DB_PATH` | `data/search` | LanceDB search table directory |
| `TRIALMATCHAI_SEARCH_TRIALS_TABLE` | `trials` | Trial table |
| `TRIALMATCHAI_SEARCH_CRITERIA_TABLE` | `criteria` | Criteria table |
| `TRIALMATCHAI_SEARCH_MODE` | `hybrid` | `bm25`, `vector`, or `hybrid` |
| `TRIALMATCHAI_CONCEPT_DB_PATH` | `data/concepts` | Concept linker table directory |
| `TRIALMATCHAI_ENTITY_BACKEND` | `gliner2` | Entity recognition backend |
| `TRIALMATCHAI_COT_BACKEND` | `vllm` | Reasoning backend |

## 8. Operational Checks

Use:

```bash
trialmatchai-healthcheck
trialmatchai-healthcheck --require-tables
trialmatchai-index
trialmatchai-run
```

Preflight checks validate local paths, optional model artifacts, entity dependencies, GPU/vLLM readiness, and LanceDB table availability before expensive model startup.
