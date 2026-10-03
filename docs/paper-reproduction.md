# TrialMatchAI paper results and reproduction

This report summarizes the TrialMatchAI evaluation on 125 patient topics from
TREC 2021 and TREC 2022 and on 100 synthetic ideal candidates. It accompanies the
[Nature Communications study](https://doi.org/10.1038/s41467-026-70509-w)
and the [deposited result files](https://zenodo.org/records/15045515).

The tables use the deposited rankings, official TREC judgments, and accompanying
source data. Metric definitions and aggregation methods are described below,
along with commands for recalculating the TREC results. Calculation date:
3 October 2026.

## Trial ranking

Each patient is one ranking query: 75 topics in TREC 2021 and 50 in TREC 2022.
The tables show arithmetic means and medians over the topics in each track.
Values are rounded to four decimal places; downloadable files retain full
precision.

### nDCG

| Cutoff | TREC 2021 mean | TREC 2021 median | TREC 2022 mean | TREC 2022 median |
| :--- | ---: | ---: | ---: | ---: |
| nDCG@5 | 0.7297 | 0.7469 | 0.7221 | 0.8193 |
| nDCG@10 | 0.7165 | 0.7355 | 0.7085 | 0.7625 |
| nDCG@20 | 0.7244 | 0.7331 | 0.7195 | 0.7663 |

These calculations use linear TREC relevance grades and a logarithmic position
discount. Unjudged trials are removed before applying the cutoff. The ideal
ranking consists of the judged trials present in the returned ranking. Score
ties retain their deposited file order.

### P@k in the deposited results

| Cutoff | TREC 2021 mean | TREC 2021 median | TREC 2022 mean | TREC 2022 median |
| :--- | ---: | ---: | ---: | ---: |
| P@5 | 0.7507 | 0.8000 | 0.7060 | 0.8000 |
| P@10 | 0.7053 | 0.7500 | 0.6630 | 0.7250 |
| P@20 | 0.6740 | 0.7000 | 0.6200 | 0.6375 |

P@k in the deposited results is calculated by dividing the sum of the top-k
relevance grades by `k × max(top-k grades)`, with zero assigned when the maximum
grade is zero. Grades are 2 for eligible, 1 for excluded, and 0 for not relevant.
Eligible-only precision and graded precision with a fixed denominator of
`2 × k` are reported separately in the downloadable results.

Across the 125 topics, the topic-weighted means are 0.7133 for nDCG@10 and
0.6884 for P@10 under this convention.

## Retrieval recall across TREC 2021 and 2022

The combined judged-trial corpus contains 48,714 unique trials: 26,162 from
TREC 2021 and 26,585 from TREC 2022, with 4,033 shared IDs. This corpus count is
the union of the official judgment pools.

At a cutoff of 1,000 candidates, the pooled median eligible-trial recall across
the 125 topics is 90.91%. This candidate budget represents 2.05% of the combined
judged-trial corpus and falls within a 3% budget, equivalent to approximately
1,462 candidates.

The table reports recall at 1,000 candidates, using each topic's own judgments
as the recall denominator:

| Topics | Eligible mean | Eligible median | Grade ≥1 mean | Grade ≥1 median |
| :--- | ---: | ---: | ---: | ---: |
| TREC 2021 · 75 | 83.58% | 90.91% | 75.80% | 81.75% |
| TREC 2022 · 50 | 81.19% | 89.78% | 71.65% | 76.97% |
| Combined · 125 | 82.62% | 90.91% | 74.14% | 80.32% |

Eligible recall counts grade-2 trials. Grade ≥1 recall, used in the
Figure 3A source data, counts both grade-1 and grade-2 trials. For either
definition, recall is the number of relevant IDs in `nct_ids_K.txt` divided by
the number of relevant IDs in that topic's qrels. Combined means and medians
are calculated directly over the 125 individual topic values.

The [per-topic recall data](assets/paper-results/retrieval-recall.csv) include
both definitions at 12 candidate cutoffs, from 10 to 1,000.

## Synthetic ideal candidates

For each synthetic patient, the evaluation records the rank of the trial used
to generate that patient's profile. The table summarizes these positions from
the Figure 2 source data and supplementary ideal-candidate rankings:

| Ground-truth trial position | Patients | Percentage |
| :--- | ---: | ---: |
| Rank 1 | 92 / 100 | 92% |
| Within top 2 | 95 / 100 | 95% |
| Within top 10 | 100 / 100 | 100% |

The maximum ground-truth rank is 9.

## Reproduce the TREC calculations

From a checkout containing the reproduction command and calculation conventions
described above, install TrialMatchAI and run:

```bash
python -m pip install -e .
trialmatchai reproduce-paper --workdir ./paper-reproduction
```

The command downloads the deposited result archive and official TREC qrels,
verifies their pinned SHA-256 checksums, and writes
`paper-reproduction/reproduction-report.json`. Calculations from saved outputs
run on CPU. The report includes the saved metrics, recalculated ranking metrics,
track summaries, and Figure 3A recall definition. It also records the current
evaluator's results separately, with its own treatment of ties and unjudged
trials, so each set of values retains its evaluation settings.

The ranking recalculation agrees with the saved topic metrics and track
summaries. Numerical checks and input checksums are recorded in the JSON data.

For an offline calculation with downloaded inputs:

```bash
trialmatchai reproduce-paper \
  --workdir ./paper-reproduction \
  --archive /artifacts/matching_results.zip \
  --qrels-dir /artifacts/qrels \
  --no-download --json
```

Use a fresh work directory for an independent calculation. The accepted result
archive has SHA-256
`dbfd11f19ffbfd78acfeaeee634c3e7e7ca8cb4d72b526ff8d191f9e880312c8`.
The [machine-readable results](assets/paper-results/results.json) record the
qrels and source-workbook checksums, the combined-corpus calculation, and the
eligible-only recall values reported here.

## Data and sources

| Download | Contents |
| :--- | :--- |
| [Results JSON](assets/paper-results/results.json) | Definitions, checksums, calculation checks, and full-precision summaries |
| [Ranking metrics CSV](assets/paper-results/ranking-metrics.csv) | Per-topic saved and recalculated nDCG and P@k values |
| [Retrieval recall CSV](assets/paper-results/retrieval-recall.csv) | Per-topic recall at each candidate cutoff, hit counts, and both relevance definitions |

The ranking tables and downloadable ranking metrics use the deposited
`matching_results.zip` values.

The paper's
[Source Data workbook](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41467-026-70509-w/MediaObjects/41467_2026_70509_MOESM6_ESM.xlsx)
provides Figures 2 and 3; the
[Supplementary Data workbook](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41467-026-70509-w/MediaObjects/41467_2026_70509_MOESM4_ESM.xlsx)
also provides the ideal-candidate rankings. Official judgments are available
from NIST for
[TREC 2021](https://trec.nist.gov/data/trials/qrels2021.txt) and
[TREC 2022](https://trec.nist.gov/data/trials/qrels2022.txt).
