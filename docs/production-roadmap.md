**TrialMatchAI production roadmap**

Owner: TrialMatchAI maintainers. Started 12 September 2026. Baseline: `508db33` / 0.8.2. Initial implementation branch: `production/reliability-foundation`.

This plan incorporates all 38 findings in [the codebase review](codebase-review-2026-09-12.md), plus production checksums, automated checks, CI/CD, executable end-to-end workflows, and a CLI suitable for both people and automation. The objective is balanced clinical matching quality and practical usability. Completion of the first PR does not imply that the entire system is production-qualified.

**Delivery model**

Use small, reviewed pull requests with one operational objective and an explicit acceptance gate. Keep `main` releasable, use short-lived feature branches, and ship immutable versioned artifacts. A PR must identify the behavior changed, migration implications, tests, remaining limits, and linked review findings. Start with a reproducible fixed pipeline, then add bounded agent actions against that baseline.

The first PR establishes the verification/release foundation, a runnable synthetic CLI example, checksum tooling, installed-package validation, and the immediately reproducible patient-attribution and packaging defects. Later work packages remain separately reviewable. The historical audit/probes remain baseline evidence; do not reinterpret their old observations as the behavior of a fixed release.

**Work packages, dependencies, and acceptance gates**

| Package | Scope and review findings | Depends on | Acceptance gate |
|---|---|---|---|
| P01 — Release and CLI foundation | Reusable required verification; installed-wheel CLI e2e; artifact checksum creation/verification; strict bootstrap integrity mode; CLI version/demo; package catalog; Make targets; FHIR subject attribution. Findings 1, 33, 37; checksum/CI/CLI requests | None | A clean installed wheel runs the synthetic example, verifies artifacts, rejects corruption and contradictory patient references, and produces a report; publishing consumes the exact tested artifacts and waits for verification |
| P02 — Evidence identity and result lifecycle | Canonical ID validation/containment; source namespaces; patient/trial/config/model/prompt fingerprints; schema-valid completion; force propagation; atomic run publication; immutable report inputs. Findings 2, 22, 30, 32 | P01 | Changed patient, trial, adapter, prompt, or settings cannot reuse incompatible output; crash/resume and concurrent jobs cannot publish mixed results; partial failure has a nonzero automation outcome |
| P03 — Shared decision semantics | Separate relevance, eligibility state, completeness, and operational failure; unknown vs absent; pending biomarkers; inclusion/exclusion policy; consistent report labels. Findings 3, 6, 11 | P02 | Contract tests cover required inclusion failure, exclusion violation, unknown evidence, pending tests, malformed outputs, and identical decisions across ranking/reporting |
| P04 — Clinical fact/criterion model | Negation, experiencer, dates, typed quantities/units, assertions, logical criterion groups, source spans, relation-aware biomarkers; importer/exporter validation and conversion-loss reports. Findings 7–10 | P02, P03 | Clinically meaningful FHIR/OMOP/Phenopacket/text fixtures survive import → evidence → assessment; unsupported semantics abstain; supported FHIR resources pass standard validation |
| P05 — Corpus/model integrity | Shared embedder fingerprint; per-artifact model identity; strict dimension/finite-value checks; versioned trial snapshots; registry status reconciliation; fetched/prepared/indexed states; atomic snapshot publication; linked-concept provenance. Findings 13, 14, 23–25 | P02 | Model swaps re-embed correctly; changed/closed trials reach all stages; rebuilds cannot relabel old vectors; corrupt/mixed snapshots are rejected |
| P06 — Honest evaluation and experiment records | All-topic denominators; standard/condensed/conditional metric names; reference evaluator parity; frozen corpora; per-topic run files/configs; figure reproduction and confidence intervals. Findings 27–29 | P01; P02 for artifact lineage | Hand-computed miniature runs match the reference; failed/empty topics count; each published number has immutable inputs and a repeatable command |
| P07 — First retrieval quality | Native pooling/metrics, field channels, accepted-only concepts, filter pushdown/refill, status/location preferences, bounded expansion kept separate from source facts. Findings 4, 15–17, 20 | P04–P06 | Held-out recall improves or is preserved at a declared cost; filter losses/channel failures are visible; no generated query becomes clinical evidence |
| P08 — Second retrieval and assessment | Real patient evidence packets; supportive/contradictory criteria; per-trial coverage; deterministic ties; empty-scope behavior; grouped/chunked reasoning; complete criterion IDs; bounded schema repair. Findings 12, 18, 19, 21 | P03, P04, P06, P07 | Eligible-candidate retention and criterion coverage are measured; long inputs cannot silently lose criteria; malformed or incomplete assessment cannot become complete |
| P09 — Model/training qualification | One model reference contract for Hub/local/merged/adapters; pinned revisions; device normalization; adapter round trips; label validation; train/dev/test isolation; calibration and error mining. Findings 5, 26 | P01, P03, P06 | Every supported backend loads its declared adapter and completes tiny train/save/reload/infer tests; GPU stack passes actual inference and memory qualification before release |
| P10 — CLI and operational interface | Structured errors/exit codes; `init`, `config validate/show`, `doctor`, plan/dry-run/status/resume/cancel; patient/run selection; JSON stdout and logs on stderr; deterministic noninteractive behavior | P02, P05 | Scripted and interactive onboarding require no source edits; failures have actionable messages; machine output parses; Ctrl-C cancels safely and resumes from valid state |
| P11 — Clinical workspace | Fact review/correction; evidence citations; questions; persistent shortlists/comparison; reviewer overrides with rationale; current site/contact/distance data; referral preparation; accessibility. Finding 31 | P03, P04, P08, P10 | A clinician completes representative screening tasks without editing files; time, errors, questions, and correction burden are recorded; results identify their source snapshot |
| P12 — Bounded agent workflow | Typed retrieval/evidence/validation tools; explicit state graph; bounded refinement; useful missing-information questions; selective disagreement review; tool traces/budgets and safe data boundaries | P06–P11 | Agentic mode improves predefined quality/workflow measures against the fixed pipeline; no unbounded retries; unanswered questions remain pending; external actions require the appropriate explicit user action |
| P13 — Simplification and scale | Consolidate stage executor/config/atomic writers/expansion APIs; deprecate dead exports; wire effective knobs; batch index maintenance and GPU work; model lifetime budgets. Findings 34–36, 38 | Incremental throughout | One source of truth per behavior; no exposed inert knobs; compatibility migrations documented; throughput and p95 latency measured on a representative corpus |
| P14 — Deployment qualification | Immutable container/package/model/data versions; staging promotion, migrations, canary, rollback, backup/restore, observability, access control, retention, disaster recovery | P01–P13 as applicable to supported deployment | Staging e2e and recovery drills pass; SLOs and support boundaries published; production promotion/rollback uses previously qualified immutable artifacts |

Every review finding maps to at least one package. Package owners should convert each finding into an issue and a desired-behavior regression test when implementing its fix. Do not merge broad scoring/model changes into the release-infrastructure PR merely to make the roadmap look complete.

**Checksums and artifact trust**

Use SHA-256 for transport/content integrity and content-addressed artifact identities. A checksum alone does not authenticate a publisher: its expected value must come from a trusted manifest or release provenance. Release artifacts should have a checksum manifest, source commit, package/version metadata, dependency inventory/SBOM, and build provenance. Verify checksums after artifact transfer and before publication/deployment. Promote exactly the artifact that passed installed-package tests; do not rebuild it in the publish job.

For downloaded trial/model/training artifacts, validate expected digest syntax and require all requested checksums before starting a strict bootstrap. Hash cached downloads too, reject mismatches before extraction, use atomic downloads/extraction publication, and retain verification provenance. Legacy archives without published SHA-256 values need a documented one-time acquisition/verification process; do not invent trusted digests or silently claim that old caches are verified. Later corpus manifests must record individual file hashes, source versions, schema/model fingerprints, row counts, and completeness. Legacy extraction sentinels cannot establish verified provenance.

**CI/CD contract**

| Trigger | Required checks and behavior |
|---|---|
| Every PR / main push | Frozen lockfile; Ruff; complete CPU unit/integration suite; bounded job timeouts; JUnit artifacts; package build; checksum verification; installed-wheel CLI e2e from outside the checkout; secret/dependency checks |
| Merge queue, if enabled | The same reusable verification against the merge-group commit; one stable aggregate check for branch protection |
| Scheduled / model-stack change | Supported model/backend adapter load/infer round trips; GPU e2e on trusted workers; model revision availability; representative memory/latency checks; vulnerability exception review |
| Benchmark change | Frozen-data smoke evaluator parity on PRs; full held-out experiments on trusted scheduled/manual jobs; archive exact configs, per-topic metrics, run files, and provenance |
| Release | Tag/version consistency; same reusable verification at the release commit; install/test exact wheel; checksums/provenance; OIDC publishing; deployment uses the validated artifact |
| Staging / production | Environment-scoped identity, qualified artifact digest, readiness/liveness checks, synthetic e2e, migration compatibility, canary SLO comparison, automated rollback criteria |

Pin third-party Actions to reviewed commit SHAs and update them with Dependabot. Keep `contents: read` as the default, and grant publishing/attestation permissions only to the relevant jobs. Never execute untrusted PR code on a privileged self-hosted GPU worker or expose deployment secrets to it. Use job timeouts and cancellation for superseded CI runs; releases should not cancel one another mid-publish.

Repository branch protection, required reviewers, protected deployment environments, OIDC trust configuration, runners, registry credentials, and production infrastructure are account-level settings. Record the exact settings in a deployment runbook and apply them only to the intended repository/environments; YAML alone does not activate them. No production environment is assumed or deployed by the first PR.

These choices follow [GitHub reusable-workflow semantics](https://docs.github.com/en/actions/how-tos/reuse-automations/reuse-workflows), [artifact provenance guidance](https://docs.github.com/en/actions/concepts/security/artifact-attestations), [uv's GitHub integration](https://docs.astral.sh/uv/guides/integration/github/), and [PyPI trusted publishing](https://docs.pypi.org/trusted-publishers/).

**End-to-end validation pyramid**

1. Fast deterministic tests exercise clinical/data invariants and failure paths with synthetic records. Assert desired outcomes, not implementation details.
2. Installed-package CPU e2e invokes the public CLI in a clean working directory with a real temporary LanceDB store, synthetic trial/patient files, deterministic embeddings, retrieval, ranking, report generation, and resume. It requires no model download and is explicitly a software smoke test, not clinical/model validation.
3. Tiny-model integration covers tokenizer/prompt budgets, adapters, structured output, save/reload, and backend parity. Use small pinned fixtures and fail supported configurations when these checks fail.
4. GPU e2e covers the actual supported NER/retriever/reranker/reasoner stack, long criteria, malformed generation, cancellation/recovery, model memory, and artifact lineage. Run on isolated trusted workers.
5. Full evaluation uses frozen TREC and clinician-adjudicated holdouts with failed topics included, confidence intervals, and model/data/config provenance. Include temporal/site subgroups and error categories.
6. Staging synthetic transactions validate the deployed endpoint, worker, storage, search snapshot, report, access boundaries, and rollback—not only that a process starts.

Required negative cases include corrupt archives; unsafe archive/manifest paths; missing/wrong checksums; expired or unavailable model revisions; explicit foreign patient references; absent vs negated facts; pending tests; incompatible units; stale measurements; logical OR/AND; criteria beyond context; partial output; disk/write failure; interrupted download/build/run; stale cache after edits; empty query/candidate/result; registry closure; and mixed model/index identity.

**CLI user experience contract**

Provide progressive disclosure: a one-command synthetic demo, a validated starter configuration, then reproducible commands for real data. Show the active workspace/run, artifact locations, progress by stage, skipped work and its reason, and a concrete next command. Common failures should identify the field/path/model at fault without exposing patient notes, tokens, or stack traces by default. Keep verbose diagnostics opt-in.

Support explicit patient/run selection and keep different datasets/organizations isolated. `--json` should emit only a versioned JSON object/event stream on stdout; normal diagnostics go to stderr. Document exit codes for success, invalid input/config, execution failure, and partial completion. Noninteractive jobs must never wait for prompts. Cancellation must save a resumable state and return the appropriate termination code. Dry-run must be genuinely read-only, showing planned work and estimated resources without downloading models or creating completion markers.

An operator should be able to answer: “What will run?”, “Which artifacts/models are used?”, “What changed since the previous run?”, “Why did this trial rank here?”, “What failed?”, and “What exact command resumes it?”

**Agent and clinical quality gates**

Maintain separate states for clinical eligibility, evidence completeness, and execution status. Model explanations should be concise conclusions grounded in source facts and criterion IDs. Generated query text must not replace clinical evidence. Unknown information remains unknown. Let an agent perform bounded retrieval refinement, evidence lookup, question prioritization, and repair; record every tool input/output/version and stop at a declared budget. Additional reasoning is justified by measured improvement and consequential uncertainty, not by a default demand for longer CoT.

Before claiming quality improvements, establish corrected baselines for all-topic recall/nDCG/eligible precision, per-criterion accuracy, false eligibility/exclusion, evidence fidelity, calibration, successful-run fraction, review time, and cost/latency. Thresholds are to be agreed from measured baselines and clinical error tolerance; no arbitrary production score is invented here.

**Initial implementation status**

P01 is implemented on the foundation branch with recorded local validation in
[the implementation checks](production-validation.md). Hosted checks and repository
protection configuration must be verified on the draft PR. No release or deployment
has been performed. P02–P14 remain planned work, including production GPU security
and actual model qualification. The local full model environment's dependency audit
reported 23 advisory rows across six packages on 12 September 2026, after the legacy
exceptions; some rows are repeated aliases. Do not describe that stack as security
qualified. A clean base/dev audit is a separate, narrower check.

**Operational definition of done**

A release is qualified only for its declared deployment profile. The maintainers need a supported OS/Python/GPU/model matrix, reproducible installation, tested data migrations, stable CLI/API contracts, release notes, observability dashboards/alerts, retention and access policy, encrypted storage/transport where applicable, backup/restore and rollback drills, capacity measurements, and a documented incident owner. For a shared clinical application, conduct usability/accessibility and clinical error review using representative users and data.

The first implementation PR supplies a foundation and makes remaining gates visible. Production readiness remains an earned result of the later work packages and recorded acceptance evidence.
