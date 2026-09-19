# Reproducing the published results

The Nature Communications study evaluated TrialMatchAI on 75 TREC 2021 topics,
50 TREC 2022 topics, and a separate synthetic “ideal candidate” experiment. The
current repository can verify the immutable result artifact and recalculate its
TREC metrics without a GPU. It cannot yet perform an exact fresh inference rerun
of the historical pipeline.

## Run the artifact audit

From an installed release containing this command:

```bash
trialmatchai reproduce-paper --workdir ./paper-reproduction
```

The command downloads the 124.7 MB `matching_results.zip` from the paper's
[Zenodo data record](https://zenodo.org/records/15045515), verifies its pinned
SHA-256 digest, extracts only the files needed for the audit, downloads the
official TREC qrels, and writes `paper-reproduction/reproduction-report.json`.
It reports three separate views:

1. **Stored per-topic aggregation** checks whether the archived topic metrics
   aggregate to the archived `average_metrics.json`.
2. **Recalculated from rankings** applies the paper artifact's metric convention
   to `ranked_trials.json`: remove unjudged trials, use linear relevance gains,
   retain file order within score ties, and compute graded precision as
   `sum(grade) / (2*k)`.
3. **Current evaluator** applies the repository's present tie-aware, condensed
   evaluator to the same rankings. These values answer a different metric
   question and should not be substituted for the published values.

Use [`trialmatchai trec-evaluate`](trec-evaluation.md) to compare the current
evaluator with unjudged trials excluded versus retained as zero-gain results.
That comparison can reuse any completed result directory and performs no model
inference.

For an offline or CI run, supply the already-downloaded official archive and
qrels:

```bash
trialmatchai reproduce-paper \
  --workdir ./paper-reproduction \
  --archive /artifacts/matching_results.zip \
  --qrels-dir /artifacts/qrels \
  --no-download --json
```

The official archive accepted by the command has SHA-256
`dbfd11f19ffbfd78acfeaeee634c3e7e7ca8cb4d72b526ff8d191f9e880312c8`.
The TREC 2021 and 2022 qrels are also pinned by SHA-256. Changing any input byte
causes the audit to stop.

## What the published artifacts reproduce

The following values were obtained with the command above on 19 September 2026.
Means and medians are macro-averages across all topics in each track.

| Track and metric | Stored mean | Stored median | Recalculated mean | Current-evaluator mean |
| :--- | ---: | ---: | ---: | ---: |
| TREC 2021 nDCG@5 | 0.729721 | 0.746939 | 0.727659 | 0.669658 |
| TREC 2021 nDCG@10 | 0.716471 | 0.735478 | 0.706848 | 0.669746 |
| TREC 2021 nDCG@20 | 0.724413 | 0.733114 | 0.692088 | 0.688335 |
| TREC 2021 graded P@10 | 0.705333 | 0.750000 | 0.692667 | 0.692667 |
| TREC 2022 nDCG@5 | 0.722144 | 0.819318 | 0.717099 | 0.587426 |
| TREC 2022 nDCG@10 | 0.708484 | 0.762500 | 0.683137 | 0.598333 |
| TREC 2022 nDCG@20 | 0.719456 | 0.766262 | 0.656213 | 0.632342 |
| TREC 2022 graded P@10 | 0.663000 | 0.725000 | 0.661000 | 0.661000 |

For the same archived rankings, retaining unjudged trials as zero-gain results
instead of excluding them changes current tie-aware mean nDCG@10 from 0.669746
to 0.435800 on TREC 2021 and from 0.598333 to 0.458934 on TREC 2022. The JSON
report stores both policies explicitly.

Across all 125 topics, weighting both tracks by topic count gives a stored mean
nDCG@10 of 0.713276 and stored mean graded P@10 of 0.688400. These do not exactly
regenerate the paper text's pooled values of 0.7232 and 0.6865 from the published
per-topic metric files.

The stored per-topic files aggregate to the stored summaries. Recalculation from
the archived rankings finds at least one metric difference in 20 of 75 TREC 2021
topics and 19 of 50 TREC 2022 topics. The report records every affected topic and
value. This means the aggregate tables are verifiable as an aggregation of stored
metrics, but some stored topic metrics cannot be regenerated from the ranking
files in the same archive.

The archived first-stage cutoff files produce the following grade-1-or-higher
retrieval recall:

| Track | Recall@500 mean | Recall@500 median | Recall@1000 mean | Recall@1000 median |
| :--- | ---: | ---: | ---: | ---: |
| TREC 2021 | 0.693768 | 0.738095 | 0.757988 | 0.817518 |
| TREC 2022 | 0.676977 | 0.719643 | 0.716526 | 0.769697 |

These archived values do not establish the paper text's statement of greater
than 90% retrieval recall at roughly the top 3% of trials under this relevance
definition and macro-averaging. That statement may use a different run,
denominator, or aggregation, but the distinction is not recoverable from the
published artifact.

The paper's source-data workbook supports the reported ideal-candidate result:
92 of 100 ground-truth trials are ranked first, 95 are in the top two, and all
are in the top nine. The matching archive's `IDEAL_CANDIDATES` directory contains
102 runs and does not regenerate that workbook table from its final ranking
files, so the current command does not present it as a ranking-file
recalculation.

## Why a fresh exact rerun is not yet available

The [paper](https://doi.org/10.1038/s41467-026-70509-w) describes Elasticsearch,
BGE-M3 retrieval, Gemma-2-2B reranking, and Phi-4 eligibility reasoning. The
current pipeline uses LanceDB and has changed ranking, tie, configuration, and
resume behavior. The [archived v0.01 code](https://doi.org/10.5281/zenodo.18329084)
does not include a single executable workflow that reconstructs all figures, and
its checked-in defaults do not fully match the methods text. The published
bundle also lacks enough lineage to bind every output to exact registry records,
model revisions, container images, prompts, random seeds, and hardware/runtime
versions.

An exact inference reproduction therefore needs a new immutable experiment
manifest containing:

- the historical full trial corpus and a content manifest;
- exact topics, qrels, preprocessing outputs, and candidate pools;
- model and adapter revisions with checksums;
- the Elasticsearch version, mappings, analyzers, and index snapshot;
- effective configuration and prompt templates for every stage;
- deterministic seeds, runtime/container lock, and GPU details;
- all per-topic run files, failures, timings, and figure-generation code.

Until those inputs are recovered or rebuilt, call the command an **artifact
audit**, not an exact end-to-end replication. New experiments should use the
current evaluator and publish a complete run manifest; compare them with the
paper only after labeling the changed corpus, engine, models, and metrics.
