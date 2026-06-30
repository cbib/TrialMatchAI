# Changelog

All notable changes to TrialMatchAI are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project adheres to
[Semantic Versioning](https://semver.org/).

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
