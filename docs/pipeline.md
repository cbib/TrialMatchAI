# Pipeline &amp; CLI

TrialMatchAI is **one end-to-end pipeline** built from an ordered registry of
**idempotent stages**. Every command is a *slice* of this pipeline. Because each
stage detects and skips work that is already done, a run "just works" from any
starting state — finished stages are cheap no-ops, unfinished ones run.

## The stages

| # | Stage | What it does | Idempotency check |
|---|-------|--------------|-------------------|
| 1 | `prepare` | embed + entity-annotate the trial corpus | per-trial prepared file |
| 2 | `concepts` | build the entity-linking concept store | concept table present |
| 3 | `index` | build the LanceDB search tables | both tables present |
| 4 | `ingest` | import patient inputs into canonical profiles | per-patient profile |
| 5 | `expand` | CoT query expansion of patient summaries | `query_expanded` marker |
| 6 | `match` | retrieval + reranking + CoT eligibility + ranking | per-patient `ranked_trials.json` |
| 7 | `eval` | score results against qrels (benchmark runs) | benchmark-only |

## The single command

```bash
trialmatchai pipeline [selection] [options]
```

**Selection — run any subset** (the unit of modularity is the stage):

| Flag | Meaning | Example |
|------|---------|---------|
| *(none)* | run every stage, skipping what's done | `trialmatchai pipeline` |
| `--only` | run exactly these stages | `--only match,eval` |
| `--from` / `--to` | run a contiguous slice | `--from index --to match` |
| `--skip` | omit stages (great for ablation) | `--skip expand` |
| `--force` | redo stages even if done (`all` = everything) | `--force match` |

**Options:** `--input` (repeatable patient files/dirs), `--format`,
`--trials-json-folder`, `--processed-trials-folder`, `--processed-criteria-folder`,
`--concepts` / `--concepts-csv` / `--synonym-csv`, `--config`.

```bash
trialmatchai pipeline --only prepare,index             # build the search index
trialmatchai pipeline --input patient.fhir.json        # ingest + match one patient
trialmatchai pipeline --skip concepts,expand           # leaner run for an ablation
trialmatchai pipeline --force all                       # rebuild everything from scratch
```

## Ablation

Stage flags double as ablation knobs — toggle a component and compare:

```bash
trialmatchai pipeline --skip expand     # matching without LLM query expansion
trialmatchai pipeline --skip concepts   # without entity-concept linking
```

Component backends (reranker, CoT, search mode `bm25`/`vector`/`hybrid`) are set in
the config; see [Architecture](architecture.md).

## Presets (the same pipeline, named)

These are thin wrappers over the pipeline that add their own setup:

| Command | Equivalent slice | Adds |
|---------|------------------|------|
| `trialmatchai build` | `--to index` | build manifest; bootstrap-aware prepare |
| `trialmatchai e2e` | `--from index --to match` | patient ingestion convenience |
| `trialmatchai trec` | `--from index --to eval` (per track) | official topics + qrels + corpus restriction |
| `trialmatchai run` | `--only match` | match already-staged profiles |
| `trialmatchai index` | `--only prepare,index` | — |

Every command is idempotent and resumable: re-running continues from the last
completed work.

## Python API

```python
from trialmatchai.config.config_loader import load_config
from trialmatchai.pipeline import StageContext, run_pipeline

ctx = StageContext(config=load_config(), inputs=["patient.txt"])
run_pipeline(ctx, from_stage="index", to_stage="match")
```

See the [API reference](api.md) for `StageContext`, `Stage`, `select_stages`, and
`run_pipeline`.
