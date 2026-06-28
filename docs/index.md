# TrialMatchAI

**AI-driven patient-to-clinical-trial matching.** Import a patient — free text,
FHIR, Phenopacket, or OMOP — and get ranked, eligible trials with criterion-level
eligibility explanations. Local LanceDB hybrid search + vLLM reasoning on a single
GPU server; no Elasticsearch or hosted vector database to run.

!!! warning "Research and informational use only"
    TrialMatchAI is not medical advice, not a medical device, and must not replace
    review by qualified healthcare professionals.

## Install

```bash
uv pip install "trialmatchai[llm,gpu,entity]"   # full model-backed runtime (GPU host)
# or, lightweight (CLI + base deps only):
uv pip install trialmatchai
```

## The two halves

TrialMatchAI runs in two halves — **build the system once**, then **match patients
many times** — and both are idempotent: finished work is never redone.

```bash
trialmatchai bootstrap-data            # fetch the prepared corpus + adapters (Zenodo)
trialmatchai build --concepts          # prepare + index + concept store (resumable)
trialmatchai e2e --input patient.txt   # ingest + match one patient
# -> results/<patient_id>/ranked_trials.json
```

## One pipeline, maximally modular

Under the hood everything is a slice of a single, ordered pipeline of idempotent
stages. Run the whole thing, or any subset:

```bash
trialmatchai pipeline                       # run every stage (skipping what's done)
trialmatchai pipeline --only match          # just (re)match
trialmatchai pipeline --to index            # the build half
trialmatchai pipeline --skip expand         # ablation: no query expansion
trialmatchai pipeline --force match         # redo a stage even if done
```

See **[Pipeline &amp; CLI](pipeline.md)** for the stage list and flag scheme,
**[Architecture](architecture.md)** for how it fits together, and the
**[API reference](api.md)** for the Python API.

## Cite

> Abdallah, M. *et al.* TrialMatchAI: an end-to-end AI-powered clinical trial
> recommendation system to streamline patient-to-trial matching. *Nature
> Communications* **17**, 4472 (2026).
> <https://doi.org/10.1038/s41467-026-70509-w>
