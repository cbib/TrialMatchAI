# Evaluating completed TREC runs

TrialMatchAI can recompute ranking metrics from completed result directories
without rebuilding an index or loading an LLM:

```bash
trialmatchai trec-evaluate \
  --track 21 \
  --results-dir h100_rrf/results_trec21 \
  --results-dir qwen36_medcpt/results_trec21 \
  --qrels data/trec/qrels/qrels_21.txt \
  --include-per-query \
  --output trec21-evaluation.json
```

The command evaluates both supported unjudged-document policies by default:

- `exclude` removes unjudged trials before applying each rank cutoff. This is
  the condensed-list convention previously used by the current evaluator.
- `include_as_zero` preserves every ranked position and assigns unjudged trials
  relevance grade zero. This is the conventional trec_eval interpretation.

Both policies use the same tie-aware nDCG calculation. The report includes
macro means and medians, every topic when `--include-per-query` is supplied,
the qrels SHA-256, and a SHA-256 over every evaluated `ranked_trials.json` and
`nct_ids.txt`. The two policies must never be compared without their names.

## Existing completed analysis

Six local configurations have complete TREC 2021 and 2022 results: 75 and 50
topic directories respectively, each with first-stage candidates, criterion
assessments, final rankings, and evaluation output. Re-running model inference
would duplicate these expensive runs. The closest current-code analogue to the
paper stack is `h100_rrf`: BGE-M3 retrieval, Gemma reranking, and Phi-4
eligibility assessment. Its original effective configuration was not stored
beside its outputs, so it is evidence from a completed experiment rather than a
fully provenance-bound replication.

The matching pipeline uses vLLM by default for the reranker and eligibility
reasoner. vLLM batches many criterion pairs and trial prompts within a patient.
Independent tracks and experiment arms can run as Slurm arrays on separate
GPUs; tensor parallelism is available for models that need multiple GPUs. For
these completed runs, metric evaluation is CPU-only and finishes faster than
queueing a GPU job.

## Measured policy sensitivity

These values were regenerated from the complete local rankings on 19 September
2026. `P@10` below is graded precision, `sum(relevance grade)/(2*10)`.

| Track | Run | nDCG@10 exclude | nDCG@10 include-zero | P@10 exclude | P@10 include-zero | Unjudged top 10 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| 2021 | h100_rrf | 0.790937 | 0.608344 | 0.775333 | 0.586667 | 26.3% |
| 2021 | h100_medcpt | 0.798727 | 0.585323 | 0.782667 | 0.572000 | 28.3% |
| 2021 | h100_baichuan | 0.822496 | 0.621061 | 0.802667 | 0.601333 | 26.1% |
| 2021 | qwen36_medcpt | 0.821458 | 0.613718 | 0.812667 | 0.601333 | 27.9% |
| 2021 | huatuo_medcpt | 0.681722 | 0.430457 | 0.676000 | 0.427333 | 39.1% |
| 2021 | iimedical_medcpt | 0.753910 | 0.545734 | 0.737333 | 0.534667 | 29.6% |
| 2022 | h100_rrf | 0.752821 | 0.653702 | 0.700000 | 0.603000 | 15.2% |
| 2022 | h100_medcpt | 0.714433 | 0.625418 | 0.671000 | 0.584000 | 17.8% |
| 2022 | h100_baichuan | 0.727787 | 0.622698 | 0.695000 | 0.589000 | 17.6% |
| 2022 | qwen36_medcpt | 0.783205 | 0.691356 | 0.731000 | 0.639000 | 14.2% |
| 2022 | huatuo_medcpt | 0.610370 | 0.490513 | 0.598000 | 0.486000 | 22.8% |
| 2022 | iimedical_medcpt | 0.690797 | 0.576935 | 0.659000 | 0.547000 | 18.8% |

The inclusive policy lowers nDCG@10 by 0.089–0.251 across these runs. This is
expected because the systems retrieve trials outside the original judgment pool;
the inclusive policy treats those trials as incorrect even though they were
never assessed. The condensed policy avoids that assumption but measures
ordering only among judged retrieved trials. Report both, and use recall-aware
`ndcg_full@k` alongside them when retrieval coverage matters.

The complete per-topic reports are tracked as
`benchmarks/trec/unjudged-policy-trec21.json` and
`benchmarks/trec/unjudged-policy-trec22.json`.

## When another GPU run is warranted

Launch a new Slurm inference job only for a configuration whose effective model,
adapter, corpus, index, prompts, candidate budgets, or ranking code differs from
an existing run. A production experiment should write that effective
configuration, Git revision, model revisions, index fingerprint, Slurm job ID,
GPU type, and output manifest into the result directory before processing the
first topic. Use one array arm per independent track/configuration and vLLM
batching within each arm. Resume incomplete patient and trial outputs rather
than passing `--rematch` unless invalidation is intentional.
