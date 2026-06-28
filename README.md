<div align="center">

<img src="img/logo.png" alt="TrialMatchAI" width="480"/>

<p><b>AI-driven clinical trial matching.</b> Import a patient — text, FHIR, Phenopacket, or OMOP — and get ranked, eligible trials with criterion-level eligibility explanations. Local LanceDB search + vLLM reasoning on a single GPU server; no Elasticsearch or hosted vector database to run.</p>

<p>
  <a href="#install">Install</a> ·
  <a href="#quickstart">Quickstart</a> ·
  <a href="#how-it-works">How it works</a> ·
  <a href="#configuration">Configuration</a> ·
  <a href="#cli-reference">CLI</a>
</p>

</div>

> **⚕️ For research and informational use only.** TrialMatchAI is not medical
> advice, not a medical device, and must not replace review by qualified
> healthcare professionals.

## TL;DR

TrialMatchAI runs in **two halves**: **build the system once**, then **match
patients many times**. Both commands are idempotent and resume after disruption.

```bash
uv sync --extra llm --extra gpu --extra entity   # GPU host + HuggingFace access
uv run trialmatchai bootstrap-data               # fetch prepared corpus + adapters
uv run trialmatchai build                        # 1) BUILD: prepare + index (once)
uv run trialmatchai e2e --input patient.txt      # 2) MATCH: ingest + match a patient
# -> results/<patient_id>/ranked_trials.json
```

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
| `entity` | GLiNER2 biomedical extraction |
| `llm` | local embedding and LLM dependencies |
| `gpu` | vLLM and bitsandbytes; intended for Linux CUDA hosts |
| `finetune` | training dependencies for `trialmatchai finetune` |

Installing the package only gives you the CLI. Real matching also needs the
trial corpus, model artifacts, and LanceDB search tables — all produced by the
**build** step below.

## Quickstart

The pipeline has two halves. **Build** is the heavy, one-time setup (GPU); it is
resumable and only does work that is not already done. **Match** is fast and
repeatable against the built system.

### 0. Set up the runtime (GPU host)

```bash
uv sync --extra llm --extra gpu --extra entity   # model-backed runtime
cp .env.example .env                             # optional local overrides
export HF_TOKEN=<token>                           # required for gated models (phi-4, gemma-2)
```

### 1. Build the system — once

```bash
uv run trialmatchai bootstrap-data   # download the prepared corpus + LoRA adapters
uv run trialmatchai build            # prepare embeddings/entities + build the index
uv run trialmatchai build --status   # see exactly what is built (and what isn't)
```

`build` fails fast if a GPU, an extra, or model access is missing — and resumes
from where it left off if interrupted. Bringing your **own** trials instead of
bootstrapping? Put normalized JSON in `data/trials_jsons/` and `build` will
prepare them. To enable entity→concept linking, add `--concepts` (open
vocabularies, **auto-downloaded**) — and optionally an OMOP `CONCEPT.csv` for
SNOMED/LOINC/RxNorm on top:

```bash
uv run trialmatchai build --concepts                          # genes, diseases, chemicals, cells, phenotypes
uv run trialmatchai build --concepts --concepts-csv data/omop/CONCEPT.csv --synonym-csv data/omop/CONCEPT_SYNONYM.csv
```

#### What gets fetched, and how

| Resource | How you get it | Automatic? |
| --- | --- | --- |
| Trial corpus (`processed_trials` + criteria) | `trialmatchai bootstrap-data` (Zenodo) | ✅ automatic |
| Fine-tuned LoRA adapters (CoT + reranker) | `trialmatchai bootstrap-data` (Zenodo) | ✅ automatic |
| Fine-tuning datasets (only if you re-train) | `trialmatchai bootstrap-data --finetune-data` (Zenodo) | ✅ automatic (opt-in) |
| Embedder (`BAAI/bge-m3`) | downloaded from HuggingFace on first use | ✅ automatic |
| Concept-linking vocabularies (genes, diseases, …) | `trialmatchai build --concepts` | ✅ automatic |
| Base LLMs (`microsoft/phi-4`, `google/gemma-2-2b-it`) | HuggingFace on first use | ⚠️ automatic, but **gated** models need a **one-time** `hf auth login` + accepting the model licence |
| OMOP clinical vocab (SNOMED/LOINC/RxNorm) | download `CONCEPT.csv` from [OHDSI Athena](https://athena.ohdsi.org/) | ❌ manual (licensed); linking works without it |

So a from-scratch user runs **two commands** (`bootstrap-data`, then `build --concepts`) after a one-time `hf auth login`. Everything else is pulled on demand.

### 2. Match patients — repeatably

`e2e` ingests the patient (format auto-detected) and matches in one command:

```bash
uv run trialmatchai e2e --input data/patients/raw/patient-1.txt
uv run trialmatchai e2e --input data/patients/raw/patient-1.fhir.json
uv run trialmatchai e2e --input data/patients/omop_extract
```

Results land in `results/<patient_id>/` (ranked trials + eligibility
explanations). Re-running skips patients already matched.

### Health and keeping trials current

```bash
uv run trialmatchai healthcheck                          # validate config/paths/deps
```

Fold new/changed ClinicalTrials.gov studies into the **live index** — fetch →
embed + entity-annotate → upsert, incremental and idempotent (unchanged studies
are skipped via a manifest, so it is safe to re-run):

```bash
uv run trialmatchai update-registry --since 2026-06-01   # one-shot
uv run trialmatchai update-registry --watch --interval 86400   # server: update daily
```

For a one-shot cadence you can also drive `update-registry` from cron, a systemd
timer, or GitHub Actions — see [docs/registry-updater.md](docs/registry-updater.md).

<details>
<summary>Manual / advanced control (the steps <code>build</code> and <code>e2e</code> wrap)</summary>

```bash
uv run trialmatchai index --prepare                          # prepare + index from trials_jsons (what `build` runs)
uv run trialmatchai import-patient --input patient.txt       # stage a profile only
uv run trialmatchai run                                      # match already-staged profiles
uv run trialmatchai trec --tracks "21 22"                    # benchmark: official TREC CT eval
```

</details>

## How It Works

The diagram below is the **match** path. The one-time **build** step produces the
LanceDB index it queries — trial and criterion embeddings, entity annotations,
and parsed eligibility constraints.

```text
Patient data (text / FHIR / Phenopacket / OMOP)
      |
      v
Interop importers -> canonical PatientProfile
      |
      v
GLiNER2 entity extraction + deterministic variant patterns -> concept linking
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
`trialmatchai run`.

See [docs/interoperability.md](docs/interoperability.md) for format details.

## First-Level Retrieval

First-level retrieval is recall-oriented. It builds a multi-channel query plan
from the canonical `PatientProfile` and matching summary, then searches each
channel separately and fuses candidates with reciprocal rank fusion.

Channels include primary conditions, linked concept synonyms, broader disease
terms, patient narrative text, biomarkers, prior therapy or procedures, and
optional LLM-generated expansions. LLM expansion is off by default; deterministic
concept and patient-profile expansion are the default path.

By default the first level hard-filters by age, sex/gender, and recruitment
status (`search.first_level.hard_filters`). Geographic **location** is an opt-in
hard filter: add `"location"` to `hard_filters` to keep only trials with a
recruiting site in the patient's country (country-level, site-aware, and
recall-safe — trials with unknown site countries are never dropped; patient
location is populated by the FHIR and OMOP importers). Biomarkers, phase, prior
therapy, and eligibility constraints remain soft signals for later stages.

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
| Biomedical extraction | `fastino/gliner2-base-v1` | `entity_extraction.model_name` |
| Reranker | `google/gemma-2-2b-it` | `model.reranker_adapter_path` |
| CoT eligibility | configured CoT model | `model.cot_adapter_path` |

Fine-tune model components with:

```bash
uv sync --extra finetune
uv run trialmatchai finetune cot \
  --base-model microsoft/phi-4 \
  --train-data data/finetune/cot.jsonl \
  --output-dir models/cot-adapter
uv run trialmatchai finetune reranker \
  --base-model google/gemma-2-2b-it \
  --train-data data/finetune/reranker.jsonl \
  --output-dir models/reranker-adapter
uv run trialmatchai finetune ner \
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

There is a single entry point — `trialmatchai` — and every capability is a
subcommand. Under the hood they are all slices of **one idempotent pipeline**.

**The unified pipeline (run any subset)**

| Command | Purpose |
| --- | --- |
| `trialmatchai pipeline` | Run the whole pipeline, or any slice: `--only` / `--from` / `--to` / `--skip` / `--force` over the stages `prepare → concepts → index → ingest → expand → match → eval`. Idempotent — finished work is skipped. See [docs](https://cbib.github.io/TrialMatchAI/pipeline/). |

The commands below are convenience presets over that pipeline.

**Build the system (setup half)**

| Command | Purpose |
| --- | --- |
| `trialmatchai build` | Prepare the corpus (embeddings + entities) and build the search index — resumable, with `--status` |
| `trialmatchai bootstrap-data` | Download and extract the prepared corpus + model adapters |
| `trialmatchai build-concepts` | Build the LanceDB concept table for entity normalization (optional, OMOP) |
| `trialmatchai update-registry` | Fetch changed ClinicalTrials.gov studies and upsert LanceDB |

**Match patients (run half)**

| Command | Purpose |
| --- | --- |
| `trialmatchai e2e` | Ingest a patient and match end-to-end (idempotent, per-patient resume) |
| `trialmatchai import-patient` | Import text, FHIR, Phenopacket, or OMOP patient data into a profile |
| `trialmatchai run` | Match already-staged patient profiles |
| `trialmatchai trec` | Benchmark: end-to-end evaluation on the official TREC Clinical Trials tracks |

**Utility**

| Command | Purpose |
| --- | --- |
| `trialmatchai healthcheck` | Validate config, paths, optional model deps, and LanceDB tables |
| `trialmatchai index` | Lower-level prepare/index of trial and criteria search tables |
| `trialmatchai finetune` | Fine-tune NER, reranker, or eligibility reasoning models |

```bash
uv run trialmatchai build --status      # what is built
uv run python -m trialmatchai e2e --input patient.txt
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
uv run pre-commit run --all-files   # ruff + gitleaks secret scan + hygiene
uv run pip-audit --progress-spinner off --ignore-vuln CVE-2025-3000
```

Install the git hooks once so secret scanning and linting run on every commit:

```bash
uv run pre-commit install
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

> Abdallah, M. _et al._ TrialMatchAI: an end-to-end AI-powered clinical trial
> recommendation system to streamline patient-to-trial matching. _Nature
> Communications_ **17**, 4472 (2026). <https://doi.org/10.1038/s41467-026-70509-w>

```bibtex
@article{abdallah2026trialmatchai,
  title   = {TrialMatchAI: an end-to-end AI-powered clinical trial recommendation system to streamline patient-to-trial matching},
  author  = {Abdallah, Majd and Nakken, Sigve and Georges, Mikael and Bierkens, Mariska and Galvis, Johanna and Groppi, Alexis and Karkar, Slim and Meiqari, Lana and Rujano, Maria Alexandra and Canham, Steve and Dienstmann, Rodrigo and Fijneman, Remond and Hovig, Eivind and Meijer, Gerrit and Nikolski, Macha},
  journal = {Nature Communications},
  volume  = {17},
  pages   = {4472},
  year    = {2026},
  doi     = {10.1038/s41467-026-70509-w},
  url     = {https://doi.org/10.1038/s41467-026-70509-w}
}
```

## Support

- Email: abdallahmajd7@gmail.com
- Software archive (DOI): <https://doi.org/10.5281/zenodo.18329084>
