# TrialMatchAI — Michigan Health Integration Notes

## Pipeline Modes

### Full Pipeline (Linux GPU server)

Requires CUDA GPU. Loads two large models:

| Model | Size | Role |
|-------|------|------|
| `microsoft/phi-4` | ~14GB | Keyword extraction from phenopacket + CoT eligibility reasoning |
| `google/gemma-2-2b-it` | ~2GB | LLM reranker (binary inclusion/exclusion classifier) |

**Stages:**
1. Phenopacket → LLM keyword extraction (`main_conditions`, `other_conditions`, `expanded_sentences`)
2. First-level BM25 + vector search against `clinical_trials` ES index
3. Second-level hybrid retrieval + Gemma-2 reranking against `eligibility_criteria` ES index
4. CoT reasoning (phi-4 via vLLM) evaluating each eligibility criterion
5. Final scoring: `(met_criteria - violated_criteria) / total_criteria`

**Note:** Gemma-2 requires accepting Google's license on HuggingFace before the model weights can be downloaded.

---

### `--skip-llm` Mode (local Mac / no GPU)

Skips all model loading. Runs retrieval only.

```bash
python -m Matcher.main --config source/Matcher/config/config.json --skip-llm
```

**What changes:**
- Keyword extraction uses phenopacket structured fields directly (no LLM):
  - `main_conditions` ← `diseases[*].term.label`
  - `other_conditions` ← `phenotypicFeatures[*].type.label` (present features only)
  - `expanded_sentences` ← `PhenopacketProcessor.generate_medical_narrative()`
- Gemma-2 reranker is skipped; second-level search falls back to ES scores only
- CoT reasoning and final LLM ranking are skipped
- Output is `ranked_trials.json` with retrieval scores (not criterion-level scores)

**What still runs:**
- BGE-M3 embedding model (570MB, runs on MPS on Apple Silicon)
- First-level BM25 + vector search
- Second-level hybrid retrieval

**Apple Silicon (M4 Pro):** BGE-M3 embedder auto-detects MPS and runs on the GPU. The full pipeline (phi-4 + vLLM) requires CUDA and won't run on MPS — that stays Linux-only for now.

---

## Data Architecture

### Reference Data (Clinical Trials)

Trials are stored as **flat JSON** — no standard format required. The indexer expects these fields per trial:

| Field | Source |
|-------|--------|
| `nct_id` | Trial identifier |
| `brief_title` | Short title |
| `brief_summary` | Plain-language description |
| `condition` | Target condition(s) |
| `eligibility_criteria` | Inclusion/exclusion text blob |
| `overall_status` | Recruiting, completed, etc. |
| `phase` | I / II / III / IV |
| `gender` | All / Male / Female |
| `minimum_age` / `maximum_age` | Age eligibility |
| `intervention` | Drug/device/procedure |
| `location` | Site(s) |

Current data source: ClinicalTrials.gov XML (109k trials in `data/processed_docs/`, 2.1M eligibility criteria in `data/processed_criteria/`). Pre-computed BGE-M3 vectors are embedded in these files.

### Query Data (Patients)

Patients must be represented as **GA4GH Phenopackets** (JSON). Key fields the pipeline reads:

| Field | Used for |
|-------|----------|
| `diseases[*].term.label` | Primary condition matching |
| `phenotypicFeatures[*].type.label` | Secondary condition/symptom matching |
| `interpretations[*].diagnosis.genomicInterpretations` | Genomic variant context (gene symbol, variant label) |
| `medicalActions[*].treatment.agent.label` | Prior/current treatments |
| `subject.sex`, `subject.dateOfBirth` | Demographic eligibility filtering |
| `biosamples[*].histologicalDiagnosis` | Tissue/histology context |

---

## Michigan Health Integration Plan

### Clinical Trials: OnCore / REDCap → TrialMatchAI

ClinicalTrials.gov is often out of date. Michigan Health manages active trials in **OnCore** (built on REDCap). OnCore is the authoritative source.

**To-do:** Build a translation layer in `utils/DataLoader/` that:
1. Calls OnCore/REDCap API to pull active trials
2. Maps OnCore field names → the flat dict schema above
3. Feeds translated records into the existing indexer (`utils/Indexer/index_trials.py` + `index_criteria.py`)

**Key mapping challenge:** OnCore's eligibility criteria fields may use different names and structure than the indexer expects. The translation layer is a one-time mapping exercise — once mapped, re-indexing on schedule keeps ES current.

**Benefit:** Trials appear in the matcher the moment they're entered into OnCore, without waiting for ClinicalTrials.gov sync.

---

### Patient Data: EHR → Phenopacket

Michigan Health uses Epic. Patient records need to be converted to phenopacket format before the pipeline can process them.

**Options:**
1. **Epic FHIR API → Phenopacket converter** — Epic exposes FHIR R4 endpoints; write a converter that maps FHIR Condition/Observation/MedicationStatement resources to phenopacket fields
2. **Manual phenopacket authoring** — for clinical trial coordinators entering patients manually
3. **REDCap instrument → Phenopacket** — if patient data is already captured in REDCap, map REDCap fields to phenopacket schema

**HIPAA note:** Phenopackets used as pipeline input will contain PHI. All processing must stay within Michigan Health infrastructure. UMGPT (Azure OpenAI) is approved for PHI and should replace any OpenAI API calls in the pipeline (see Task #1).

---

### Fine-tuning on Michigan Oncology Cases

The current phi-4 and Gemma-2 adapters were fine-tuned on general ClinicalTrials.gov data. Michigan-specific fine-tuning via UMGPT would improve matching quality for:

- Michigan's specific trial vocabulary and eligibility language (OnCore format)
- Michigan Medicine oncology protocols
- Local standard-of-care assumptions baked into eligibility criteria

**Approach:** Use `utils/finetuning/` scripts but point them at UMGPT instead of OpenAI. The data generation scripts (`gpt_generate_reranking_data.py`, `gpt_generate_summaries.py`, `gpt_generate_ideal_candidates.py`) generate training pairs — swap in UMGPT credentials there (Task #1).

---

### Oncology-Specific Enhancements (from MatchMiner)

TrialMatchAI handles general matching. For precision oncology, consider borrowing MatchMiner's strict genomic pre-filter logic (Task #3):

- Exact genomic variant matching before LLM stages (HGVS notation, gene symbol + variant type)
- Reduces LLM hallucination risk on critical genomic eligibility criteria
- Can be inserted as a pre-filter step between first-level search and second-level retrieval

This keeps TrialMatchAI as the foundation (no oncology-only technical debt) while adding the precision that oncology matching requires.

---

## Open Tasks

| # | Task | Notes |
|---|------|-------|
| 1 | Update `utils/gpt/` scripts to use UMGPT | Change `base_url` and `api_key` in `ChatOpenAI` constructor; required before fine-tuning on Michigan data |
| 3 | Genomic variant pre-filter | Borrow MatchMiner exact-match logic; insert before second-level search |
| 4 | Replace ClinicalTrials.gov with OnCore/REDCap | Implement in `utils/DataLoader/` |
| 5 | OnCore → indexer field translation layer | Map OnCore field names to `nct_id`, `eligibility_criteria`, etc. |
| 6 | Harden `bootstrap_data.sh` | Checksum validation, resume on partial download, extraction validation |
