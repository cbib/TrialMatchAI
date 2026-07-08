# Changelog

All notable changes to TrialMatchAI are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project adheres to
[Semantic Versioning](https://semver.org/).

## [0.5.0] — 2026-07-08

### Added
- **Asymmetric dual-encoder embedders.** Setting `embedder.query_model_name` selects an
  embedder whose document and query sides use different encoders that share one vector space
  (e.g. MedCPT's article encoder for trials and query encoder for patient queries). Trial
  documents are embedded with the document encoder and first-level query terms with the query
  encoder; symmetric embedders (the default `BAAI/bge-m3`) are unaffected. The two encoders are
  checked to emit the same dimension at load time.
- **Configurable first-level vector similarity** via `search_backend.vector_metric`: `cosine`
  (default) or `dot`. It drives both the LanceDB ANN index and the Python re-ranker, so a
  dual-encoder trained for inner-product retrieval (unnormalized embeddings, e.g. MedCPT) can be
  scored with its native metric instead of cosine.
- **Configurable hybrid blend** via `search_backend.vector_weight` (default `0.5`): the
  first-level score is `(1 - vector_weight) · text + vector_weight · vector`, so retrieval can
  lean further on the embedder or on BM25.
- **First-level embedder benchmark harness** (`scripts/benchmark_embedder.py` +
  `benchmarks/embedders/registry.json`). It builds an isolated trials-only index per embedder by
  re-embedding the TREC qrels corpus, then reports first-level recall@k for both grade-2
  (eligible) and grade-1+2 (relevant) definitions — non-destructively, each model writing its own
  index and `benchmarks/embedders/<name>.json`, with concept-linking held on a fixed reference
  embedder so only the retrieval embedder varies.

### Fixed
- The cosine re-ranker keeps its original single-pass scoring; only the `dot` path buffers
  candidates to min-max normalize the vector score before blending. An all-equal `dot` batch (no
  matching query vector) now maps to a neutral `0.0` instead of inflating every candidate.
- `vector_metric`/`vector_weight` are validated where the backend is constructed, and incremental
  `upsert_trials` now rebuilds the trial vector index with the configured metric instead of
  hard-coded cosine.

## [0.4.1] — 2026-07-06

### Added
- **`trialmatchai trec --rerank`** re-ranks a finished run from its cached chain-of-thought
  outputs and re-evaluates against the official qrels — no build, match, or model inference.
  It reuses the stored per-trial `NCT*.json` eligibility outputs plus the reranker/first-level
  scores already in `ranked_trials.json`, so a ranking-logic change (e.g. the Unclear-credit
  change below) can be re-applied to existing results in seconds instead of re-running the GPU
  pipeline; it rewrites each patient's `ranked_trials.json` and the track's
  `evaluation_metrics.json`.

### Changed
- **Eligibility scoring credits "Unclear" criteria instead of dropping them.** The coarse
  eligibility band scored only the Met/Not-Met inclusion criteria and discarded "Unclear"
  entirely. Unclear is the dominant classification, so discarding it collapsed the band — a
  mostly-Unclear trial shared a band with a genuinely all-Met one, and the applicability
  reranker then broke those ties, letting merely-relevant trials outrank eligible ones near
  the top. An Unclear inclusion criterion now counts at 0.5 (Met=1, Not Met=0), so the band
  separates all-Met trials from Unclear-heavy ones. "Irrelevant" (criterion does not apply to
  the patient) is still excluded, and a Violated exclusion still hard-disqualifies.

## [0.4.0] — 2026-07-05

### Added
- **Per-condition first-level query channels for comorbidities.** Each distinct
  `other_conditions` term now gets its own focused first-level retrieval channel (weight
  0.25, below the primary condition) instead of only diffusing into the shared narrative
  channel. A single blended channel over many distinct conditions dilutes to noise, which
  starved retrieval on multi-morbid patients; per-condition channels restore their coverage
  without displacing single-condition patients. The channels are de-duplicated against the
  primary terms and bounded by a fixed cap.
- **Recall-aware nDCG reported alongside the condensed variant.** TREC evaluation now emits
  both `ndcg@k` — normalized by the ideal over judged-**and-ranked** trials (pure ordering
  quality, independent of recall) — and `ndcg_full@k` — normalized by the ideal over the
  **full** judged pool (`trec_eval`-style), where a relevant trial that was never ranked
  lowers the score. Reporting both keeps ordering quality and retrieval coverage legible
  instead of conflating them in a single number. The DCG numerator is condensed (ignores
  unjudged trials) in both.

### Changed
- **Second-level shortlist selection fuses rankings instead of summing raw scores.**
  The pre-CoT shortlist previously ranked trials by `first_level_score + second_level_score`.
  Those two scores live on different scales (the first-level value is a reciprocal-rank
  fusion contribution; the second-level value is an aggregated reranker score), so the
  larger-scaled signal dominated the sum and a trial the retriever ranked highly could be
  dropped whenever the reranker scored its criteria low — or omitted entirely when no
  criterion cleared the aggregation threshold. Shortlist selection now combines the
  first-level and second-level *rankings* with reciprocal-rank fusion over the union of
  both, so a strong retrieval hit keeps a floor and rank fusion is scale-free. Behaviour is
  governed by `search.shortlist_fusion` (`rrf`, the new default, or `score_sum` for the
  previous behaviour), `search.shortlist_rrf_k`, and per-source weights
  `search.shortlist_first_level_weight` / `search.shortlist_second_level_weight`. The
  shortlist size is unchanged — it is still keyed to the number of reranked trials — so only
  which trials are selected changes, not how many.

### Fixed
- **TREC precision was computed on the raw ranked list while its docstring — and nDCG —
  used the condensed (judged-only) list.** `evaluate` now condenses the ranking to the
  judged pool before the P@k cutoff, so an unjudged trial the assessors never saw is no
  longer scored as a miss. This aligns precision with `condensed_ndcg` and with the
  function's own documented behaviour; the prior inconsistency understated P@k for a
  non-pooled system and made it incomparable to nDCG. Judged-but-non-relevant (grade 0)
  trials still count as negatives — only genuinely unjudged pairs are ignored. A regression
  test pins the condensing.

## [0.3.3] — 2026-07-02

### Fixed
- **Config knobs were silently dropped by settings validation.** `VllmSettings` and
  `LLMRerankerSettings` only preserve declared fields, so `LLM_reranker.tensor_parallel_size`
  (added in 0.3.2) never survived `load_config` and the reranker stayed single-GPU. It is now
  declared and reaches the engine. A regression test pins that these knobs round-trip.

### Added
- **fp8 KV cache + concurrency cap for vLLM.** New `vllm.kv_cache_dtype` (`fp8` / `fp8_e4m3` /
  `fp8_e5m2`) halves KV-cache memory so a large context window (e.g. 8192) fits on a single
  48 GB card, and `vllm.max_num_seqs` caps concurrency so that tight KV budget does not thrash.

## [0.3.2] — 2026-07-02

### Fixed
- **Healthcheck no longer aborts a cold-start build.** The 0.3.1 hardening made
  `LanceDBSearchBackend.health()` flag a never-built (table-less) index as
  unhealthy *by default*. Since a pre-build preflight (`trialmatchai healthcheck`)
  legitimately runs before the index exists — and job scripts run it under
  `set -e` — a from-scratch run aborted before it ever built. The never-built
  check is now gated on `require_tables=True`, so the lenient default preflight
  passes pre-build while match-readiness (which already passes `require_tables=True`)
  still catches an unbuilt index.

### Added
- **Multi-GPU reranker.** `LLM_reranker.tensor_parallel_size` lets the vLLM
  reranker shard across GPUs, matching the CoT engine, so both engines can run
  tensor-parallel on a multi-GPU node (e.g. 2× L40) instead of contending for a
  single device.

## [0.3.1] — 2026-07-02

A deep, line-by-line robustness audit. 82 verified defects were fixed across
retrieval, constraint evaluation, entity linking, FHIR/OMOP ingest, TREC
metrics, and the build/resume pipeline — each pinned by a regression test
(test suite grew to 377). No API changes; behaviour is more correct, not
different in shape.

### Fixed
- **Constraint evaluation & extraction.** Lab thresholds and patient values
  with thousands-separator commas (`10,000`) no longer truncate to a wrong
  magnitude with the unit dropped; a relative's disease in `family_history` no
  longer satisfies a patient-disease inclusion; bare gene names in drug/therapy
  phrases no longer emit spurious "present" biomarker constraints; ECOG parsing
  no longer grabs an unrelated distant number; cross-unit numeric comparisons
  abstain instead of mis-deciding; inclusion aggregation uses worst-case (not
  mean), mirroring exclusion; "greater than" is exclusive; whole-word matching
  replaces raw substring containment.
- **Retrieval & search.** A criterion retrieved by several query paraphrases is
  de-duplicated before trial aggregation (no more inflation by query overlap);
  short query tokens no longer match inside unrelated words for a spurious
  phrase score; the scan fallback pushes its limit into LanceDB instead of
  loading the whole table; a never-built search index is reported unhealthy
  instead of silently passing.
- **Entity linking.** The lexical accept-gate scores substring matches by length
  ratio instead of a flat high constant (no more linking "carcinoma" to
  "hepatocellular carcinoma"); the patient/registry annotation path now wires
  the lexical reranker so correct concepts ranked #2+ are not abstained; OBO
  concept files no longer drop their last term; an empty mention degrades to
  FTS-only instead of aborting linking.
- **Query expansion.** The CoT reasoning output is now stripped of `<think>`
  blocks before JSON extraction, so a reasoning preamble can no longer poison
  the primary retrieval query with placeholder text.
- **Patient ingest (FHIR/OMOP).** OMOP `NOTE_NLP` facts are no longer silently
  dropped (correct `person_id` resolution via `NOTE`); OMOP NaN dates,
  float-promoted negations, and the `concept_id 0` sentinel are handled;
  ISO-8601 ages with day/week components no longer lost; FHIR `valueRange`
  keeps negative signs, `valueRatio` no longer emits literal "None", genomic
  resources with non-`CodeableConcept` types are no longer dropped.
- **TREC & ranking.** The final ranking passes the pure second-level reranker
  score (not the first+second combined value) to `rank_trials`; the score blend
  breaks coarse eligibility ties within-band; trial dates with month/year
  precision are parsed deterministically instead of fabricated from today.
- **Build/resume integrity.** Atomic temp-then-rename writes for concept
  dictionaries, downloads, and qrels/topics caches; completion sentinels so a
  crashed bootstrap extract or partial topic import is not treated as done; the
  index fingerprint no longer chains off a stale link fingerprint; per-trial
  resume re-embeds a trial whose content changed; the concept-store fingerprint
  includes the embedder identity; atomic-write temp files no longer match
  `*.json` artifact globs; the reranker/CoT model identity invalidates cached
  matches.
- **Config & CLI.** Falsy-but-valid config values (`0`, empty collections) are
  respected; vLLM `top_p=0` is rejected at validation; `trialmatchai run` goes
  through corpus-fingerprint invalidation so it can't serve stale rankings after
  a reindex; failure accounting and exit codes no longer mask an all-failed
  resume run as success.
- **`__version__`** is realigned with the packaged version (had silently drifted
  to `0.2.0`), and a test now asserts the two sources agree.

### Changed
- Verbose explanatory comments and docstrings tightened to intent-only
  across the source tree (no behavioural change).

## [0.3.0] — 2026-06-30

Adds a clinician-facing results report and deepens retrieval, on top of bug
fixes and broader test coverage from the 0.2.0 deployment-readiness release.

### Added
- **Self-contained HTML match report.** `trialmatchai report --patient <id>`
  renders a portable, offline `report.html` (no server, no build step, no CDN)
  from a patient's existing results: a ranked, searchable/filterable trial list
  with per-criterion eligibility verdicts, a collapsible chain-of-thought panel,
  ClinicalTrials.gov links, and print/PDF styles. Emitted automatically after
  each match, gated by `reporting.emit_html` (off for TREC sweeps).
- **Unified multi-patient report.** `trialmatchai report --all` — and the
  end-of-run auto-emit — writes one self-contained `index.html`: a patient front
  page that drills into each patient's report client-side. New 13th subcommand:
  `report`.
- **BM25 + heuristic text-score fusion** in the LanceDB retrieval backend.

### Changed
- **Deeper TREC funnel** (1000 → 500 → 250 across first-level → rerank → CoT) via
  a configurable `search.second_level_keep_divisor`, with concept linking enabled.
- **`python-dotenv` is now a declared dependency**, so `.env` files (HF_TOKEN, API
  keys, path overrides) load on a core-only `pip install trialmatchai` — it was
  previously imported but undeclared.

### Fixed
- P0 and medium-severity bugs surfaced by the codebase audit (RRF dedup key,
  atomic `write_text_file`, and others).

### Tests
- Coverage for production retrieval/inference paths, crash-safe resume, and
  variant-recognizer / embedder / metrics hardening.

## [0.2.0] — 2026-06-28

The **deployment-readiness** release: TrialMatchAI becomes one installable,
idempotent, end-to-end pipeline behind a single CLI, with crash-safe resume,
faithful evaluation, professional docs, and PyPI-ready packaging.

### Added
- **Unified pipeline.** A single ordered registry of idempotent stages
  (`prepare → concepts → index → ingest → expand → match → eval`) driven by one
  command — `trialmatchai pipeline` — with `--only / --skip / --from / --to /
  --force`. Run the whole thing or any slice; finished work is skipped. Stage
  flags double as ablation knobs.
- **Single entry point.** One `trialmatchai` console script with 12 subcommands
  (`pipeline, healthcheck, bootstrap-data, index, build-concepts,
  update-registry, import-patient, build, run, e2e, trec, finetune`).
- **Entity-linking concept store** from openly-licensed vocabularies
  (genes, diseases, chemicals, cell lines, cell types, phenotypes), auto-downloaded
  via `trialmatchai build --concepts` / `build-concepts --sources open`; OMOP
  (SNOMED/LOINC/RxNorm) foldable via `--concepts-csv`.
- **Registry updater** (`update-registry`) — incremental fetch → prepare → upsert
  from the ClinicalTrials.gov v2 API, with a `--watch` server mode.
- **`bootstrap-data`** for the prepared corpus + LoRA adapters, plus
  `--finetune-data` for the published training sets (Zenodo).
- **Tie-aware nDCG** (McSherry–Najork) + `P@10`, with a deterministic ranking
  tie-break and a TREC reproduction regression guard.
- **Multi-format patient ingestion** — text, FHIR, Phenopacket, OMOP; patient
  location populated from FHIR/OMOP with an optional country-level trial filter.
- **PyPI-ready packaging** + a Trusted-Publishing (OIDC) release workflow.
- **Documentation site** (MkDocs Material + auto API reference) with a deploy
  workflow; `gitleaks` pre-commit and Dependabot.

### Changed
- **LanceDB** provides embedded hybrid search (no external search service); **vLLM**
  is the only LLM backend (LoRA adapters served natively) with a single shared engine.
- **GLiNER2** in-process NER replaces the BERN2 socket daemons; concept linking
  moves to a LanceDB FTS+vector store.
- **Crash-safe resume everywhere** — atomic writes (`tmp + fsync + os.replace`),
  completion-marker-last ordering, and parse-validated resume gates; every stage
  is idempotent and re-running continues from the last completed work.
- **Two-halves CLI** (`build` once, then `e2e`/`run`/`trec` many times) with
  fail-fast preflight; a Transformers CPU backend for non-GPU paths.
- **Official TREC sourcing** (NIST topics/qrels); the prepare corpus is the
  judged pool, matching the published methodology.
- Faithful domain-aware criteria chunking + a hybrid genetic-variant recognizer.

### Fixed
- Resume/ingest correctness: transiently-failed trials are retried (not locked
  in); stale criteria cleared on re-prepare; final ranking scoped to the current
  shortlist; eligibility-criteria source falls back `processed_trials →
  trials_jsons` (fail-loud on empty); `import-patient` writes atomically,
  summary-before-profile.
- The CoT/reranker vLLM engine is loaded **once** and shared (was reloaded
  per patient).
- FHIR importer hardened for real-world EHR exports.
- `concepts` / `run` / `prepare` are now idempotent (skip-if-done, `--force` to
  redo).

### Removed
- Legacy `source/` (BERN2 daemon + CRF parser) and `utils/` experiment trees.
- The homegrown secret scanner (replaced by `gitleaks`).
- The standalone `trialmatchai-*` console scripts (now `trialmatchai <subcommand>`).

[0.2.0]: https://github.com/cbib/TrialMatchAI/releases/tag/v0.2.0
