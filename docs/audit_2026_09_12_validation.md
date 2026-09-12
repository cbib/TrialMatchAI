**Validation record — TrialMatchAI review, 12 September 2026**

Source revision: `508db33`, package version `0.8.2`. Application code was not changed. Review files and synthetic evidence were added under `docs/`.

| Check | Result | Limit |
|---|---|---|
| Repository Ruff check | Passed | Lint does not establish functional correctness |
| AST parse of tracked Python source/tests/scripts | Passed | Inventory contains file hashes and line counts |
| Focused existing test selection | **181 passed, 4 deselected in 6.29s** | 29 selected test modules; four tests requiring an actual LanceDB connection were explicitly excluded |
| Full pytest attempt | **Incomplete; timed out with exit 124** | Stalled in `lancedb.connect` during `test_health_reports_never_built_search_db`; no full-suite pass is claimed |
| Synthetic audit probes | **18 observations reproduced** | Uses fakes/pure functions and invented patient/trial data; no GPU inference or actual LanceDB search |
| Wheel build in a temporary source copy | Passed | Local setuptools 81.0.0; built wheel lacks `trialmatchai/config/catalog/embedders.json` |
| GPU quality, end-to-end latency, clinical validation | Not run | Recommendations are not measured score improvements |
| Live registry reconciliation, dependency vulnerability scan, browser interaction test | Not run | Registry/UI findings are source-level findings; this review makes no new security-advisory status claims |

The full run used offline model settings and a 240-second process timeout. A faulthandler dump identified the blocked call:

```text
tests/test_audit_fixes.py:424 test_health_reports_never_built_search_db
src/trialmatchai/search/lancedb_backend.py:223 __init__
lancedb/__init__.py:121 connect
lancedb/db.py:493 __init__
lancedb/background_loop.py:25 run
concurrent/futures/_base.py:451 result
threading.py:327 wait
```

This identifies the observed blocking location, not its root cause. It is not classified as a proven application defect. Intermediate narrowed runs encountered the same connection wait in catalog/backend tests. The final selection excludes all four known connection-dependent tests below and completes successfully.

Reproduce the focused existing tests from the repository root:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest \
  tests/test_constraints.py \
  tests/test_constraint_exclusion_aggregation.py \
  tests/test_criteria_chunking.py \
  tests/test_fhir_robustness.py \
  tests/test_patient_interop.py \
  tests/test_interop_fixtures.py \
  tests/test_metrics.py \
  tests/test_qrels_eval.py \
  tests/test_trial_ranker_pytest.py \
  tests/test_trial_ranker_scoring_contract.py \
  tests/test_trial_ranker_tiebreak.py \
  tests/test_first_level_planner.py \
  tests/test_first_level_search_pytest.py \
  tests/test_second_level_search_pytest.py \
  tests/test_search_queries_pytest.py \
  tests/test_settings.py \
  tests/test_settings_validators.py \
  tests/test_config_pytest.py \
  tests/test_model_catalog.py \
  tests/test_html_report.py \
  tests/test_entities.py \
  tests/test_finetuning.py \
  tests/test_eligibility_vllm.py \
  tests/test_embedder_contract.py \
  tests/test_variant_recognizer.py \
  tests/test_location_filter.py \
  tests/test_concept_sources.py \
  tests/test_schemas.py \
  tests/test_patient_runtime_loading.py \
  -k 'not catalog_swap_makes_metric_follow and not explicit_metric_still_pins_over_embedder and not concept_table_no_recreate_appends_rows and not search_backend_metric_follows_embedder'
```

Reproduce the audit probes:

```bash
.venv/bin/python docs/audit_2026_09_12_reproduce.py
```

The probe command refreshes `docs/audit_2026_09_12_observations.json`. A zero exit means all probes executed; it does **not** mean the observed application behavior is correct. The probes are deliberately separate from the normal regression suite. Convert each relevant example into a desired-behavior regression test when implementing its fix.

Wheel contents were checked by copying `src/trialmatchai`, `pyproject.toml`, `README.md`, and `LICENSE` into a new temporary directory and invoking `setuptools.build_meta.build_wheel` there, without dependency installation. Inspection of the resulting ZIP returned:

```json
{
  "setuptools": "81.0.0",
  "wheel": "trialmatchai-0.8.2-py3-none-any.whl",
  "catalog_present": false,
  "config_present": true
}
```

The review's evidence labels distinguish executable observations, source-level findings, and proposed experiments. No predicted retrieval/clinical improvement is presented as a measured result.
