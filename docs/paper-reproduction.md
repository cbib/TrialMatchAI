# Published result files

The [TrialMatchAI study in Nature Communications](https://doi.org/10.1038/s41467-026-70509-w)
reports evaluations on TREC 2021, TREC 2022, and a synthetic ideal-candidate
dataset. The paper, its source data, and the [deposited result files](https://zenodo.org/records/15045515)
are the references for those findings. This guide explains how to inspect the
deposited TREC files with the current TrialMatchAI CLI.

## Check the deposited files

From an installed release containing `reproduce-paper`, run:

```bash
trialmatchai reproduce-paper --workdir ./paper-reproduction
```

The command downloads the deposited `matching_results.zip` archive and official
TREC judgments, verifies their pinned checksums, and writes
`paper-reproduction/reproduction-report.json`. It checks that the saved
per-topic metrics aggregate to the saved track summaries. It also calculates
metrics from the saved rankings using explicit evaluation settings, and reports
the results of the current evaluator separately. All checks use saved files and
run on CPU.

To use files you have already downloaded:

```bash
trialmatchai reproduce-paper \
  --workdir ./paper-reproduction \
  --archive /artifacts/matching_results.zip \
  --qrels-dir /artifacts/qrels \
  --no-download --json
```

The accepted archive has SHA-256
`dbfd11f19ffbfd78acfeaeee634c3e7e7ca8cb4d72b526ff8d191f9e880312c8`.
The TREC 2021 and 2022 judgment files are pinned by SHA-256 as well. Use a fresh
work directory when checking independently downloaded files.

## Read the report

The report separates saved metrics, calculations from saved rankings, and values
from the current evaluator. The calculations specify their treatment of
unjudged trials, score ties, relevance gains, and nDCG normalization. These
settings affect the resulting numbers, so comparisons should use the same
inputs and evaluation procedure. The current evaluator is provided to help
interpret results produced by this software version; its values are not a
replacement for the study's reported metrics.

The command checks the deposited TREC files. For the study's figures and
ideal-candidate analysis, consult the paper and its accompanying source data.
