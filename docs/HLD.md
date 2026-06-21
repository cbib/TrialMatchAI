# High Level Design — TrialMatchAI

## 1. Overview

TrialMatchAI is an AI-driven clinical trial matching system. Given a patient record encoded as a GA4GH Phenopacket, it retrieves and ranks the most relevant clinical trials from a large Elasticsearch-backed corpus using a three-stage pipeline: hybrid retrieval, LLM reranking, and chain-of-thought eligibility reasoning.

### Goals
- Match individual patients to relevant open clinical trials at scale
- Provide criterion-level explainability (why a trial does or does not match)
- Support reproducible, ontology-grounded patient representations

### Non-Goals
- Real-time / online inference (designed for batch processing)
- EHR integration or direct clinical deployment
- Trial outcome prediction

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          TrialMatchAI                               │
│                                                                     │
│  ┌──────────────┐    ┌────────────────────────────────────────────┐ │
│  │   Indexing   │    │              Matching Pipeline             │ │
│  │   Pipeline   │    │                                            │ │
│  │              │    │  Phenopacket                               │ │
│  │ ClinicalTrials    │      │                                     │ │
│  │ .gov JSONs   │    │      ▼                                     │ │
│  │      │       │    │  ┌─────────────────────┐                  │ │
│  │      ▼       │    │  │  Phenopacket         │                  │ │
│  │  Schema NER +│    │  │  Processor + LLM     │                  │ │
│  │  concept link│    │  │  (Phi-4 summariser)  │                  │ │
│  │      │       │    │  └──────────┬──────────┘                  │ │
│  │      ▼       │    │             │ keywords.json                │ │
│  │  BGE-M3      │    │             ▼                              │ │
│  │  embeddings  │    │  ┌─────────────────────┐                  │ │
│  │      │       │    │  │  Stage 1: Hybrid     │                  │ │
│  │      ▼       │    │  │  BM25 + Vector Search│                  │ │
│  │ Elasticsearch│◄───┼──│  (clinical_trials)   │                  │ │
│  │  clinical_   │    │  └──────────┬──────────┘                  │ │
│  │  trials      │    │             │ ~300 candidates              │ │
│  │  trials_     │    │             ▼                              │ │
│  │  eligibility │◄───┼──│  Stage 2: Criteria    │                 │ │
│  └──────────────┘    │  │  Search + LLM Rerank  │                 │ │
│                      │  │  (Gemma-2-2B)         │                 │ │
│                      │  └──────────┬──────────┘                  │ │
│                      │             │ ~33 trials                   │ │
│                      │             ▼                              │ │
│                      │  ┌─────────────────────┐                  │ │
│                      │  │  Stage 3: CoT        │                  │ │
│                      │  │  Reasoning (Phi-4 +  │                  │ │
│                      │  │  finetuned adapter)  │                  │ │
│                      │  └──────────┬──────────┘                  │ │
│                      │             │                              │ │
│                      │             ▼                              │ │
│                      │  ┌─────────────────────┐                  │ │
│                      │  │  Ranker + Output     │                  │ │
│                      │  │  ranked_trials.json  │                  │ │
│                      │  └─────────────────────┘                  │ │
│                      └────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Components

### 3.1 Indexing Pipeline

Run once to prepare the Elasticsearch indices from raw ClinicalTrials.gov data.

```
Raw Trial JSONs
      │
      ├──► Schema entity annotation + LanceDB concept linking
      │         │
      │         ▼
      │    Annotated criteria with entity synonyms
      │
      ├──► prepare_trials.py  ──► BGE-M3 embed ──► clinical_trials index
      │
      └──► prepare_criteria.py ──► BGE-M3 embed ──► trials_eligibility index
```

**`clinical_trials` index** — one document per trial:

| Field | Type | Notes |
|---|---|---|
| nct_id | keyword | Primary key |
| brief_title, brief_summary, condition, eligibility_criteria | text | BM25 searchable |
| *_vector (×4) | dense_vector (1024d) | BGE-M3 embeddings |
| minimum_age, maximum_age | float | Years, for age filtering |
| overall_status, phase, gender | keyword | Facet filters |
| interventions, locations | nested | Structured metadata |

**`trials_eligibility` index** — one document per eligibility criterion:

| Field | Type | Notes |
|---|---|---|
| criteria_id, nct_id | keyword | Stable hash-derived ID |
| eligibility_type | keyword | `inclusion` or `exclusion` |
| criterion | text | BM25 searchable |
| criterion_vector | dense_vector (1024d, HNSW) | BGE-M3 embedding |
| entities | nested | Schema entity annotations, linked concept candidates, and synonyms |

---

### 3.2 Phenopacket Processor

Converts a GA4GH Phenopacket JSON into search-ready keywords.

**Input:** `example/<patient_id>.json` (Phenopacket v2.0)

**Steps:**
1. Validate schema with Pydantic
2. Extract structured sections: demographics, phenotypic features (HPO), diagnoses (MONDO), biosamples (UBERON), treatments (CHEBI), procedures (NCIT), genomic interpretations, family history
3. Build medical narrative sentences per section
4. Feed narrative to **Phi-4** (4-bit quantized) with a structured extraction prompt

**Output:** `keywords.json`
```json
{
  "main_conditions": ["coronary artery disease"],
  "other_conditions": ["type 2 diabetes mellitus", "hypercholesterolemia"],
  "expanded_sentences": ["58-year-old male with advanced CAD..."]
}
```

---

### 3.3 Stage 1 — Hybrid Retrieval

**Component:** `ClinicalTrialSearch`  
**Index:** `clinical_trials`  
**Goal:** Cast a wide net; recall over precision.

**Query construction:**
```
hybrid_score = α × normalized_text_score + β × normalized_vector_score
               (α = 0.5, β = 0.5 by default)
```

- **BM25 side:** multi-match across `condition`, `eligibility_criteria`, `brief_title`, `brief_summary` with field-specific boosts
- **Vector side:** cosine similarity against 4 embedded fields using script-score queries
- **Synonym expansion:** the schema entity annotator links disease mentions to the LanceDB concept table and expands accepted concepts with synonyms
- **Filters:** age range, gender, `overall_status = Recruiting`

**Output:** Up to 300 trial IDs with relevance scores → `nct_ids.txt`, `first_level_scores.json`

---

### 3.4 Stage 2 — Criteria Matching + LLM Reranking

**Component:** `SecondStageRetriever`  
**Index:** `trials_eligibility`  
**Goal:** Narrow to trials with genuinely matching eligibility criteria.

**Steps:**
1. For each query term (up to 150), search `trials_eligibility` within the Stage 1 trial subset
2. Retrieve up to 250 matching criteria per query (hybrid search on criterion text + entity synonyms)
3. **Gemma-2-2B reranker** scores each (query, criterion) pair by probability of "Yes" token
4. Apply type weighting: inclusion criteria score × 1.0, exclusion criteria score × 0.25
5. Aggregate per-trial score:
   ```
   trial_score = 0.7 × (Σ scores / √count) + 0.3 × max_score
   ```
6. Filter trials below threshold (default 0.5); select top ~33

**Output:** Ranked list of ~33 trial IDs → `top_trials.txt`

---

### 3.5 Stage 3 — Chain-of-Thought Reasoning

**Component:** `BatchTrialProcessor` (HuggingFace) or `BatchTrialProcessorVLLM` (vLLM)  
**Model:** Phi-4 + fine-tuned LoRA adapter (`models/finetuned_phi_reasoning`)  
**Goal:** Criterion-level eligibility assessment with explicit reasoning.

**Prompt structure:**
```
System: You are a medical expert with advanced knowledge in clinical reasoning...
User:   Patient Profile: <expanded_sentences>
        Trial Criteria: <inclusion + exclusion criteria>
        For each criterion, assess: Met / Not Met / Not Violated / Violated / Unclear
        Output JSON with criterion-level evaluation and Final Decision.
```

**Optimizations:**
- Length bucketing to minimise padding waste across batches
- Temperature 0.0 for deterministic outputs
- Idempotent: skips already-processed trials on resume
- vLLM backend for high-throughput GPU inference

**Output per trial:** `<nct_id>.txt` (raw), `<nct_id>.json` (parsed)
```json
{
  "Inclusion_Criteria_Evaluation": [
    {"criterion": "Age ≥ 18", "status": "Met", "reasoning": "Patient is 58."}
  ],
  "Exclusion_Criteria_Evaluation": [...],
  "Final Decision": "Eligible"
}
```

---

### 3.6 Ranker

**Component:** `trial_ranker.py`  
**Goal:** Produce a single ranked list from CoT outputs.

**Scoring formula per trial:**
```
inclusion_ratio = (met + not_violated) / (met + not_violated + not_met + violated)
exclusion_ratio = (not_violated + met) / (not_violated + met + violated)
final_score     = (inclusion_ratio + exclusion_ratio) / 2
```

Irrelevant and unclear criteria are excluded from both numerator and denominator.

**Output:** `ranked_trials.json` — sorted list of `{nct_id, score}`

---

## 4. Models

| Model | Role | Quantization | Backend |
|---|---|---|---|
| BAAI/bge-m3 | Text embeddings (1024d) | FP16 | HuggingFace |
| microsoft/phi-4 | Keyword extraction + CoT reasoning | 4-bit NF4 | HF / vLLM |
| finetuned_phi_reasoning | LoRA adapter for eligibility reasoning | — | PEFT / vLLM LoRA |
| google/gemma-2-2b-it | Pairwise criterion reranker | 4-bit NF4 | HuggingFace |

---

## 5. External Services

### Elasticsearch (3-node cluster)
- **Deployment:** Docker Compose (or Apptainer on HPC)
- **Version:** 8.13.4
- **Security:** HTTPS + X-Pack, TLS certificates
- **Memory:** 2 GB per node (6 GB total)
- **Ports:** 9200 (API), 5601 (Kibana)

### Schema Entity Annotator + LanceDB Concept Linker
- **Purpose:** Biomedical entity recognition and normalization without external Java daemons
- **Recognizer:** GLiNER2-style schema-driven extraction, with GLiNER/biomedical fallback support behind the same interface
- **Concept store:** LanceDB table built from OMOP vocabularies and legacy dictionaries
- **Entities:** diseases, genes, medications, procedures, labs, radiology, signs/symptoms, cell types, and species
- **Usage:** synonym expansion (Stage 1) and entity annotation (indexing)

---

## 6. Data Flow Summary

```
[One-time setup]
ClinicalTrials.gov JSONs
    → schema entity annotation + LanceDB concept linking
    → BGE-M3 embedding
    → Elasticsearch (clinical_trials + trials_eligibility indices)

[Per-patient inference]
phenopacket.json
    → PhenopacketProcessor → Phi-4 → keywords.json
    → ClinicalTrialSearch (BM25 + vector, ES) → ~300 trial IDs
    → SecondStageRetriever (criteria search + Gemma rerank) → ~33 trial IDs
    → BatchTrialProcessor (Phi-4 CoT reasoning) → per-trial JSON assessments
    → Ranker → ranked_trials.json
```

---

## 7. Configuration

All settings are driven by `source/Matcher/config/config.json` and can be overridden via environment variables:

| Env Var | Default | Description |
|---|---|---|
| `TRIALMATCHAI_ES_HOST` | `https://localhost:9200` | Elasticsearch URL |
| `TRIALMATCHAI_ES_USERNAME` | `elastic` | ES user |
| `TRIALMATCHAI_ES_PASSWORD` | *(required)* | ES password |
| `TRIALMATCHAI_EMBEDDER_MODEL_NAME` | `BAAI/bge-m3` | Embedding model |
| `TRIALMATCHAI_ES_AUTO_START` | `true` | Auto-launch ES if not running |
| `TRIALMATCHAI_LOG_LEVEL` | `INFO` | Logging verbosity |

---

## 8. Output Structure

```
results/
└── <patient_id>/
    ├── keywords.json           # Extracted conditions and expanded narrative
    ├── nct_ids.txt             # Stage 1 candidate trial IDs
    ├── first_level_scores.json # Stage 1 relevance scores
    ├── top_trials.txt          # Stage 2 shortlisted trial IDs
    ├── <NCT_ID>.txt            # Raw CoT reasoning text
    ├── <NCT_ID>.json           # Parsed criterion-level evaluation
    └── ranked_trials.json      # Final ranked list with scores
```

---

## 9. Infrastructure Requirements

| Resource | Minimum | Recommended |
|---|---|---|
| GPU VRAM | 24 GB | 40 GB+ (for Phi-4 unquantized) |
| RAM | 32 GB | 64 GB |
| Disk | 100 GB | 200 GB+ |
| OS | Linux / macOS | Linux (CUDA support) |
| Python | 3.10–3.11 | 3.11 |
| Elasticsearch | 3 × 2 GB nodes | 3 × 6 GB nodes |

---

## 10. Key Design Decisions

**Two-stage retrieval before reasoning** — LLM reasoning is expensive (5000 token outputs per trial). Stages 1 and 2 reduce the candidate set from hundreds of thousands of trials to ~33 before the LLM is invoked.

**Ontology-grounded inputs** — Using HPO, MONDO, CHEBI codes rather than free text ensures consistent entity matching across patient records and trial criteria regardless of phrasing variation.

**Criterion-level scoring** — The final ranking score is computed from individual criterion assessments rather than a holistic match score, enabling explainable recommendations.

**Resumable processing** — Both indexing and CoT reasoning are idempotent; partial runs can be continued without reprocessing completed items.

**Pluggable LLM backend** — The reasoning stage supports both HuggingFace (single GPU, lower overhead) and vLLM (multi-GPU, higher throughput) backends, selectable via config.

---

---

## HLD — LLM & Agent Harness View

A simpler view of the system focused on how LLMs are used and how the pipeline harness orchestrates them.

---

## 1. The Three LLM Agents

TrialMatchAI uses three LLMs, each with a distinct, narrow role. They do not share state or communicate with each other directly — the harness (`main.py`) wires their inputs and outputs together.

```
┌─────────────────────────────────────────────────────────────────┐
│                        Pipeline Harness                         │
│                          (main.py)                              │
│                                                                 │
│   Patient        ┌──────────────┐   keywords.json              │
│   Phenopacket ──►│  LLM Agent 1 ├──────────────────────┐       │
│                  │  Phi-4       │                       │       │
│                  │  Summariser  │                       ▼       │
│                  └──────────────┘             Elasticsearch     │
│                                               (BM25 + vector)   │
│                                                       │         │
│                                               ~300 candidates   │
│                                                       │         │
│                                                       ▼         │
│                  ┌──────────────┐                             │
│                  │  LLM Agent 2 │◄──────────────────────────┘  │
│                  │  Gemma-2-2B  │  (query, criterion) pairs     │
│                  │  Reranker    ├──────────────────────┐       │
│                  └──────────────┘                      │       │
│                                               ~33 shortlisted   │
│                                               trial IDs         │
│                                                       │         │
│                                                       ▼         │
│                  ┌──────────────┐                             │
│                  │  LLM Agent 3 │◄──────────────────────────┘  │
│                  │  Phi-4 +     │  (patient profile,            │
│                  │  LoRA CoT    │   trial criteria)             │
│                  │  Reasoner    ├──────────────────────┐       │
│                  └──────────────┘                      │       │
│                                               per-trial JSON    │
│                                                       │         │
│                                                       ▼         │
│                                               Rule-based Ranker │
│                                               ranked_trials.json│
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Agent 1 — Summariser (Phi-4)

**Job:** Translate a structured Phenopacket into free-text medical keywords suitable for search.

**Trigger:** Once per patient at the start of the pipeline.

**Input:**
```
Medical narrative sentences built from the Phenopacket sections:
  DEMOGRAPHICS: Sex: MALE; Age: 58Y...
  PHENOTYPE: Myocardial infarction (Present, Severe, Recurrent)...
  DIAGNOSIS: Coronary artery disease, Stage III...
  TREATMENT: Atorvastatin 40mg oral daily...
  INTERPRETATION: LDLR c.1444G>A variant...
```

**Prompt pattern:**
```
System: You are a specialized medical assistant for clinical trial matching.
        Extract primary conditions, secondary conditions, and expanded
        medical notes. Return strict JSON — no commentary.

User:   <joined medical narrative sentences>
```

**Output (JSON, max 2048 tokens):**
```json
{
  "main_conditions": ["coronary artery disease", "CAD", "ischemic heart disease", ...],
  "other_conditions": ["type 2 diabetes", "hypercholesterolemia", "LDLR mutation", ...],
  "expanded_sentences": ["58-year-old male with advanced multivessel CAD...", ...]
}
```

**Why this agent exists:** Phenopackets use ontology codes (HP:0001627), not plain text. The LLM bridges structured clinical data into natural language terms that map to how trials are described in ClinicalTrials.gov.

---

## 3. Agent 2 — Reranker (Gemma-2-2B)

**Job:** Score how relevant a specific eligibility criterion is to a patient query term.

**Trigger:** Called in bulk during Stage 2, once per (query term, criterion) pair. Runs in a thread pool with 4 parallel workers.

**Input (per pair):**
```
System: You are a clinical assistant determining if patient data supports
        trial criterion evaluation.

User:   Statement A: <one search query term, e.g. "coronary artery disease">
        Statement B: <one trial eligibility criterion, e.g. "Patients must have
                      documented CAD with ≥50% stenosis in a major vessel">
        Respond with 'Yes' or 'No'.
```

**Output:** Not free text — the model never generates a full response. The harness reads the **logit scores** for the "Yes" and "No" tokens directly from the last position of the prompt and applies softmax:

```python
score = softmax([logit_Yes, logit_No])[0]   # probability of "Yes"
```

Inclusion criteria scores are kept as-is; exclusion criteria scores are multiplied by 0.25 (a trial that excludes the patient's condition is less bad than one that requires something the patient doesn't have).

**Why this agent exists:** ES keyword and vector search can retrieve a criterion that mentions the right disease but actually *excludes* patients with that disease. The LLM reranker catches this semantic mismatch cheaply — a single forward pass with no generation overhead.

---

## 4. Agent 3 — CoT Reasoner (Phi-4 + LoRA)

**Job:** Assess each eligibility criterion one-by-one and produce an explainable eligibility decision.

**Trigger:** Once per shortlisted trial (~33 trials), batched by prompt length.

**Input:**

```
System: You are a medical expert with advanced knowledge in clinical reasoning.
        Answer the following question. Before answering, create a concise
        chain of thoughts reasoning to ensure a logical and accurate response.

User:   Assess the patient's eligibility for this clinical trial by evaluating
        each criterion individually.

        For inclusion criteria classify as: Met | Not Met | Unclear | Irrelevant
        For exclusion criteria classify as: Violated | Not Violated | Unclear | Irrelevant

        --- Trial Criteria ---
        Inclusion:
          1. Age ≥ 18 years
          2. Documented CAD with ≥50% stenosis
          3. Stable angina or prior MI
        Exclusion:
          1. Recent CABG within 6 months
          2. Active malignancy
        --- Patient Description ---
        <expanded_sentences from keywords.json>
```

**Output (JSON, max 5000 tokens):**

```json
{
  "Inclusion_Criteria_Evaluation": [
    {
      "Criterion": "Age ≥ 18 years",
      "Classification": "Met",
      "Justification": "Patient is 58 years old."
    },
    {
      "Criterion": "Documented CAD with ≥50% stenosis",
      "Classification": "Met",
      "Justification": "Diagnosis confirms advanced CAD with multivessel involvement."
    }
  ],
  "Exclusion_Criteria_Evaluation": [
    {
      "Criterion": "Recent CABG within 6 months",
      "Classification": "Violated",
      "Justification": "Patient underwent CABG on 2022-11-15, which is within 6 months of enrollment."
    }
  ],
  "Recap": "Patient meets most inclusion criteria but was disqualified by recent CABG.",
  "Final Decision": "Ineligible"
}
```

**Why the LoRA adapter:** The base Phi-4 model is general-purpose. The fine-tuned LoRA adapter (`models/finetuned_phi_reasoning`) was trained specifically on clinical trial eligibility assessment tasks, improving structured JSON output and medical reasoning precision.

---

## 5. The Harness (main.py)

The harness is a **fixed sequential pipeline** — not a dynamic agent loop. There is no tool-calling, no self-reflection, and no LLM deciding what to do next. The harness:

1. Loads all three models once at startup (cold start is expensive)
2. Iterates over patients in `patients_dir/`
3. Calls agents in a fixed order, passing outputs as inputs to the next stage
4. Writes intermediate results to disk at each step (enabling resume on failure)
5. Handles per-patient exceptions without stopping the full batch

```
startup:
    load Phi-4 (4-bit) + LoRA adapter
    load Gemma-2-2B (4-bit)
    load BGE-M3 embedder
    load schema entity annotator + LanceDB concept linker
    connect to Elasticsearch

for each patient:
    [Agent 1]  phenopacket → keywords.json
    [ES]       BM25 + vector search → nct_ids.txt
    [Agent 2]  criteria reranking → top_trials.txt
    [Agent 3]  CoT reasoning → <NCT_ID>.json (×33 trials)
    [Ranker]   score aggregation → ranked_trials.json
```

**The ranker is not an LLM** — it is a deterministic formula over the structured JSON that Agent 3 produces. This is intentional: keeping the final scoring rule-based makes it auditable and consistent.

---

## 6. Data Handed Between Agents

| From | To | Artifact | Content |
| --- | --- | --- | --- |
| Phenopacket (raw) | Agent 1 | medical narrative sentences | Plain-text sentences per clinical category |
| Agent 1 | ES Stage 1 | `keywords.json` | `main_conditions`, `other_conditions`, `expanded_sentences` |
| ES Stage 1 | Agent 2 | `nct_ids.txt` | ~300 trial IDs + scores |
| Agent 2 | Agent 3 | `top_trials.txt` | ~33 trial IDs ranked by criterion match |
| Agent 3 | Ranker | `<NCT_ID>.json` (×33) | Per-criterion classification + final decision |
| Ranker | User | `ranked_trials.json` | Scored, sorted list of trial IDs |

---

## 7. What This Is Not

- **Not a ReAct / tool-use agent** — no LLM decides what action to take next
- **Not a multi-agent conversation** — agents never see each other's outputs or converse
- **Not RAG in the classic sense** — retrieved documents (criteria) are fed as structured context to Agent 3, not as retrieved passages for general Q&A
- **Not streaming / real-time** — designed for offline batch processing of patient cohorts

---

---

## HLD — Strategy Summary (AI Engineering Learners)

---

### The Problem

Clinical trials fail to recruit because matching patients to trials manually is slow, expensive, and inconsistent. A trial might have 50 eligibility criteria. A hospital might have thousands of patients. No human team can cross-reference all of that at scale.

The goal: automate that matching with AI, and make the reasoning **explainable** — not just a black-box score.

---

### The Core Strategy: Progressive Filtering

The system never throws the most expensive AI at the full problem. Instead it uses a **funnel** — each stage is smarter but slower than the last, and only operates on what survived the previous stage.

```
Hundreds of thousands of trials
        ↓  cheap keyword + semantic search
        300 candidates
        ↓  fast LLM reranking
        33 shortlisted
        ↓  deep LLM reasoning
        Ranked results with explanations
```

This is the central engineering insight: **don't use a sledgehammer where a sieve will do.**

---

### The Four AI Techniques Used

#### 1. Embeddings + Vector Search

Convert text into numbers (vectors) that capture semantic meaning. Two pieces of text that mean the same thing — even with different words — land close together in vector space. Used in Stage 1 to find trials that are *semantically* related to the patient's conditions, not just keyword-matched.

> *Key model: BAAI/bge-m3 (1024-dimensional embeddings)*

#### 2. RAG — Retrieval-Augmented Generation

Rather than asking an LLM to memorise every clinical trial, retrieve the relevant trial's criteria at query time and inject them into the prompt as context. The LLM reasons over what it's given, not what it was trained on.

> *"Give the model the patient chart and the trial rulebook, then ask it to decide."*

#### 3. Fine-Tuning with LoRA

The base Phi-4 model is a general-purpose LLM — it can write poetry as easily as it can read medical text. Fine-tuning with a LoRA adapter (a small, efficient set of weight adjustments) specialises it for one task: structured clinical eligibility reasoning. It learns **how** to think about the problem, not specific trial content.

> *LoRA = teach the model the job, RAG = hand it the case file.*

#### 4. LLM-as-Scorer (not generator)

Gemma-2-2B is never asked to generate text. Instead it's used as a **scoring function** — the reranker reads the logit (raw probability) for just two tokens (Yes/No) to score relevance. One forward pass, no generation, extremely fast. This is an underused pattern in production AI systems.

---

### The Data Strategy

Patient data is structured using the **GA4GH Phenopacket standard** — ontology codes (HPO, MONDO, CHEBI) rather than free text. This solves the vocabulary mismatch problem: a trial saying "coronary artery disease" and a patient record saying "CAD" both resolve to `MONDO:0005066`, ensuring consistent matching regardless of phrasing.

The schema entity annotator bridges the gap between ontology codes and natural language by extracting mentions, linking them to LanceDB concept candidates, and expanding accepted disease concepts with synonyms.

---

### Key Engineering Decisions

| Decision | Why |
| --- | --- |
| Three-stage funnel | LLM inference is expensive — only invoke it on a small, pre-filtered set |
| RAG over full fine-tuning for trial knowledge | Trials change constantly; RAG stays current without retraining |
| LoRA over full fine-tuning | Trains faster, uses less memory, can be swapped per use case |
| LLM-as-scorer for reranking | Orders of magnitude faster than generating full responses |
| Criterion-level output | Explainability — doctors need to know *why*, not just a score |
| Deterministic ranker at the end | Keeps the final decision auditable and consistent |

---

### The Takeaway

TrialMatchAI is a masterclass in **composing AI primitives** rather than reaching for one big model:

- Use **search** (fast, cheap) to filter
- Use **a small LLM as a scorer** (fast, precise) to refine
- Use **RAG + fine-tuned LLM** (slow, thorough) only on the shortlist
- Use **rules** (deterministic, auditable) for the final decision

Each component does what it's best at. The system as a whole does what none of them could do alone.
