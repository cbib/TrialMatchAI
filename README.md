<div align="center">

<img src="img/logo.png" alt="TrialMatchAI" width="480"/>

<p><b>TrialMatchAI matches patients to the clinical trials they're eligible for.</b> Give it a patient — clinical notes, FHIR, Phenopacket, or OMOP — and it returns a ranked shortlist of trials, each with a transparent, criterion-by-criterion explanation of why the patient does or doesn't qualify. Everything runs on your own infrastructure: hybrid retrieval over a local LanceDB index paired with chain-of-thought LLM reasoning served by vLLM on a single GPU, so sensitive patient data never leaves your environment.</p>

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
pip install "trialmatchai[llm,gpu,entity]"   # GPU host + HuggingFace access
trialmatchai bootstrap-data                  # fetch prepared corpus + adapters
trialmatchai build                           # 1) BUILD: prepare + index (once)
trialmatchai e2e --input patient.txt         # 2) MATCH: ingest + match a patient
# -> results/<patient_id>/ranked_trials.json
```

## Requirements

- Python 3.11 (`pyproject.toml` requires `>=3.11,<3.12`)
- `pip` or `uv` — install from PyPI, or a source checkout for development
- NVIDIA GPU for vLLM-backed matching and fine-tuning
- Around 100 GB disk for datasets, model artifacts, LanceDB tables, manifests,
  and run outputs
- OMOP vocabulary files if you want to build the concept-linking table locally

## Install

### From PyPI (recommended)

```bash
pip install trialmatchai          # or: uv pip install trialmatchai
trialmatchai --help
```

That installs the package and the `trialmatchai` CLI with its base dependencies.
For model-backed matching, add the runtime extras:

```bash
pip install "trialmatchai[llm,gpu,entity]"   # full GPU runtime (Linux CUDA host)
pip install "trialmatchai[finetune]"         # fine-tuning stack
```

| Extra | Adds |
| --- | --- |
| `entity` | GLiNER2 biomedical extraction |
| `llm` | local embedding and LLM dependencies |
| `gpu` | vLLM and bitsandbytes; intended for Linux CUDA hosts |
| `finetune` | training dependencies for `trialmatchai finetune` |

### From source (for development)

```bash
git clone https://github.com/cbib/TrialMatchAI.git
cd TrialMatchAI
uv sync                                       # base; add --extra llm --extra gpu --extra entity
uv run trialmatchai --help
```

> **Calling the CLI:** installed from PyPI, run it directly — `trialmatchai ...`.
> From a `uv` source checkout, prefix with `uv run` — `uv run trialmatchai ...`.
> The examples below use the `uv run` form; drop the prefix if you installed from PyPI.

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

## Data and storage

Everything is **embedded LanceDB** — no external services. A search DB
(`data/search`, with `trials` + `criteria` tables) and a concept-linking DB
(`data/concepts`). ClinicalTrials.gov records are normalized to
`data/trials_jsons/<NCT_ID>.json`, then prepared into one trial row and one
criteria row per eligibility criterion (text + embeddings + entity annotations +
parsed constraints). Both tables carry full-text and vector columns, so retrieval
runs in `bm25`, `vector`, or `hybrid` mode. Imported patients live under
`data/patients/{profiles,summaries}/`.

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

## Learn more

Deeper guides live in the **[documentation site](https://cbib.github.io/TrialMatchAI/)**:

- **[Pipeline &amp; CLI](https://cbib.github.io/TrialMatchAI/pipeline/)** — the stage registry, `--only/--skip/--from/--to/--force`, ablation, and presets.
- **[Architecture](https://cbib.github.io/TrialMatchAI/architecture/)** — multi-channel first-level retrieval, constraint-aware ranking, and the LanceDB tables.
- **[Patient interoperability](https://cbib.github.io/TrialMatchAI/interoperability/)** — text / FHIR / Phenopacket / OMOP importers.
- **[Fine-tuning &amp; custom models](https://cbib.github.io/TrialMatchAI/finetuning/)** — swap the NER, reranker, and CoT models; training-data formats.
- **[Registry updater](https://cbib.github.io/TrialMatchAI/registry-updater/)** — keep trials current from ClinicalTrials.gov.
- **[API reference](https://cbib.github.io/TrialMatchAI/api/)** — the Python API.

To bring your own models, point `entity_extraction.model_name`,
`model.reranker_adapter_path`, and `model.cot_adapter_path` at your checkpoints /
adapters, or train them with `trialmatchai finetune {cot,reranker,ner}` — see the
[fine-tuning guide](https://cbib.github.io/TrialMatchAI/finetuning/).

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
