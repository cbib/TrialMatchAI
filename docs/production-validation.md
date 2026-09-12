# Foundation implementation validation

Branch: `production/reliability-foundation`, based on `508db33`. Date: 12 September
2026. This records checks for the implementation, separately from the historical
[baseline review evidence](audit_2026_09_12_validation.md).

- Full local pytest suite after the final Copilot fixes: **474 passed**.
  Includes real CLI e2e with missing/truncated report recovery, plus regression
  coverage for fresh bootstrap replacement across all four stages, retained backups
  and unrelated models, failed extraction/publication, interrupted recovery, writer
  locking, bounded corrupt-cache retries, non-regular markers, explicit unresolved
  FHIR subjects, and explicit/fallback query-expansion model access.
  Hugging Face and Transformers downloads were disabled.
- Initial frozen base/dev check, before review fixes: **440 passed, 1 skipped, 1 e2e deselected**;
  its dependency audit found **no known vulnerabilities without exceptions**.
  The skipped test compares the package version with an adjacent `pyproject.toml`,
  which is absent in this non-editable install. The package job checks version
  consistency from the source checkout separately.
- Final base/dev advisory recheck: **no known vulnerabilities without exceptions**.
- Frozen lockfile validation: passed with uv 0.11.24.
- Wheel and sdist build: passed. Packaged MedCPT and bge-m3 catalogs resolve.
- Isolated installed-wheel CLI smoke from `/tmp`, with checkout imports forbidden:
  passed. Covers synthetic FHIR import, real LanceDB indexing, adult/pediatric
  filtering, ranking, report content, resume preserving ranking output, and
  public checksum verification and missing/truncated report repair. No model downloads or clinical data.
- Ruff, actionlint 1.7.12, and strict MkDocs build: passed.
- CPU imports of torch, Transformers, GLiNER2, embedding, reranking, reasoning,
  and fine-tuning modules: passed in the existing local environment. This is not
  actual GPU inference or an isolated optional-dependency qualification.
- Gitleaks scan: passed; repeated on staged files before committing.
- Existing full model environment dependency audit: **failed** with 23 advisory
  rows across accelerate, mcp, mkdocs-material, setuptools, transformers, and vllm
  after legacy exceptions (13 rows ignored). Some entries are repeated aliases.
  GPU stack migration/requalification and docs/toolchain updates remain follow-up
  work; no additional exceptions were added to hide this result.

LanceDB connections hang in this session's restricted sandbox but succeed outside
it. Full pytest and installed-wheel e2e therefore ran outside that sandbox. The
historical audit's timeout is an environment limitation, not evidence of a LanceDB
product defect.

The synthetic e2e demonstrates software integration, not retrieval benchmark
improvement or clinical validity. Published ranking metrics have not been rerun.
Hosted CI and release execution are recorded on [PR #31](https://github.com/cbib/TrialMatchAI/pull/31)
and the [0.9.0 release](https://github.com/cbib/TrialMatchAI/releases/tag/v0.9.0).
A package release is separate from model migration and application deployment;
neither of those is included.
