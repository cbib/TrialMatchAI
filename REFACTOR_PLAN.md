# TrialMatchAI Refactor Plan

Derived from the comprehensive audit (170 findings → 151 confirmed, 8 uncertain, 11 refuted).
Each phase below is an independently shippable PR. Work proceeds on branch `refactor/audit-fixes`
(or per-PR branches off it).

## Principles
1. **Safety net before surgery** — CI must exercise the ML surface and characterization tests
   must lock current behavior before any behavior-changing PR merges.
2. **One PR = one theme**, independently reviewable and revertable.
3. **Behavior-preserving by default**; behavior-changing items are flagged and validated.
4. **Delete before dedup** — remove dead code before refactoring duplicates.
5. **Re-grep every deletion at edit time**, not just from the audit.

## Resolved decisions
- **Phenopacket pipeline:** `interop/importers/phenopacket.py` + `interop/narrative.py` are canonical.
  Delete `matching/phenopacket_processor.py` wholesale (only caller is its own test — verified).
- **`utils/evaluation.py`:** delete (orphaned, no entry point, no test).
- **Delivery:** this plan is tracked here; execution starts with PR0.

## PR sequence

| PR | Theme | Risk | Behavior change | Depends on |
|----|-------|------|-----------------|-----------|
| 0 | Safety net: CI ML-extras job + characterization tests + CI wheel-path fix | low | no | — |
| 1 | 🔴 Eligibility scoring contract (C1): exclusion = hard disqualifier | med | **yes** | 0 |
| 2 | 🔴 `models/llm/_common.py` + reranker hardening (C2 + padding/attn/dtype/device) | med | **yes (correctness)** | 0 |
| 3 | Dead-code sweep #1: whole modules (evaluation, embedders, regex tree, phenopacket) | low | no | 0 |
| 4 | Dead-code sweep #2: symbols/params/config/shims | low | no | 0 |
| 5 | OMOP importer: float/sanitized id join + groupby perf | med | **yes (recovers dropped records)** | 0 |
| 6 | Retrieval/indexing: restore fields, NCT sidecar filter, fallback WHERE, create_query | med | **yes (re-index required)** | 0 |
| 7 | Deduplication: BaseTrialProcessor, build_embedder, flatten_text, get_synonyms | med | no | 1,3,4 |
| 8 | Performance: lazy HF model under vLLM, precompute lancedb records | low | no | 2,5 |
| 9 | Hygiene: logging, broad excepts, packaging/deps, 8 uncertain findings | low | no | all |

Critical path: `0 → {1,2,5,6} → {3,4} → 7 → 8 → 9`.

## Per-PR detail

### PR0 — Safety net
- CI: add a Linux job that runs `uv sync --extra entity` and smoke-imports the ML modules
  (`entities.recognizers`, `models.embedding.text_embedder`, `models.llm.llm_loader`,
  `models.llm.llm_reranker`, `matching.eligibility_reasoning_vllm`).
- CI: fix the installed-smoke wheel path (`ls "$PWD"/dist/...`) so `$WHEEL` survives the `cd`.
- Tests: characterization test for `score_trial` (current behavior) + `xfail` tests encoding the
  desired post-PR1 contract (a Violated exclusion hard-disqualifies). PR1 flips xfail→pass.
- Note: OMOP (PR5) and indexing (PR6) characterization tests are added at the start of those PRs,
  immediately before their changes.

### PR1 — Eligibility scoring contract (C1)
- `trial_ranker.score_trial`: score inclusion and exclusion separately; **any `Violated` exclusion
  is a hard disqualifier** (ranks below all eligible trials). Remove impossible labels
  (`Not Violated`/`Violated` from inclusion, `Met` from exclusion) from both eligibility_reasoning
  prompt calls. Document the scoring contract.
- Update `test_score_trial_basic` (currently encodes the buggy averaging) and flip PR0's xfail tests.
- **Behavior change:** ranking order — validate against a labeled trial set.

### PR2 — LLM `_common.py` + reranker hardening
- New `models/llm/_common.py`: `load_llm_dependencies`, `resolve_cuda_device`,
  `build_4bit_quant_config`, `select_attn_impl` (flash-attn probe → `sdpa`).
- Rewrite `llm_reranker.py` on it: device pinning (C2), `padding_side='left'` + pad token,
  sdpa fallback, dtype-from-config, `device:int` coercion, honest concurrency.
- Refactor `llm_loader.py` to reuse the helpers.

### PR3 — Dead-code sweep #1 (whole modules)
Delete (re-grep first): `utils/evaluation.py`, `models/embedding/query_embedder.py`,
`models/embedding/sentence_embedder.py`, `preprocessing/regex/` tree (+ `pyproject` package-data +
`config_loader` whitelist), `matching/phenopacket_processor.py` (+ its test). Resolves the
phenopacket ontology bug, dead summarizer, always-true guards, `truncate` typo, and duplicate
pipeline by deletion.

### PR4 — Dead-code sweep #2 (symbols/params/config)
Remove `recognizers.with_schema_threshold`, `types.to_index_entity`, `retry.with_retries`,
`interop` `EvidenceSpan`/`Provenance.raw_text_span`/`all_facts`, `narrative` `style='audit'`,
`CompatibilityEntityAnnotator`, `annotator` `retries`/`delay`, dead settings (`cot`,
`LLM_reranker`, `TokenizerSettings`, `entity_extraction.threshold`, `max_text_score`,
`rerank_criteria.queries`).

### PR5 — OMOP importer
Join on raw `person_id`; normalize `'1.0'→'1'`; group child tables by `person_id` once (kills N+1);
replace `iterrows` with `itertuples`/indexed dict. Test: null `person_id` in a child table no longer
drops records.

### PR6 — Retrieval/indexing correctness
Restore `detailed_description` + `official_title` in `prepare_trial_document`; filter
`load_trial_data` to `NCT*` (or trials subdir); thread `where` into `_scan_rows` fallback; resolve
`create_query` dead keys + age semantics. **Requires LanceDB trial index rebuild after merge.**

### PR7 — Deduplication
`BaseTrialProcessor` with abstract `_generate_batch` (HF + vLLM override only that); `build_embedder(cfg)`
replacing 4 copy-pasted blocks; one canonical `flatten_text`/`clean_text`; shared `get_synonyms`.

### PR8 — Performance
Lazy-load HF CoT model only when `cot_backend != 'vllm'`; precompute
`build_trial_record`/`build_criteria_record` instead of per-row at query time.

### PR9 — Hygiene & uncertain triage
Centralize logging (no import-time `basicConfig`); replace broad bare-`except` in `import_patient` and
pass the embedder so semantic linking isn't silently disabled; drop 5 unused deps, dedupe
`torch`/`transformers` pins via shared base extra, gitignore `egg-info`/`__pycache__`/`dist`;
investigate & resolve the 8 uncertain findings (LanceDB status/age/sex push-down, vLLM sampling under
greedy, unused `query_vector`).

## Behavior-change register (validate, don't just unit-test)
PRs **1, 5, 6** change pipeline outputs. Diff before/after on a fixed patient+trial set:
ranking order (PR1), patient record counts (PR5), retrieval recall after re-index (PR6).
