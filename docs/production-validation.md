# Foundation implementation validation

Branch: `production/reliability-foundation`, based on `508db33`. Date: 12 September
2026. This records checks for the implementation, separately from the historical
[baseline review evidence](audit_2026_09_12_validation.md).

- Full local pytest suite: **442 passed**, 41.56 seconds. Includes real CLI e2e.
  Hugging Face and Transformers downloads were disabled.
- Fresh frozen base/dev environment: **440 passed, 1 skipped, 1 e2e deselected**;
  its dependency audit found **no known vulnerabilities without exceptions**.
  The skipped test requires an optional dependency covered by the separate import job.
- Frozen lockfile validation: passed with uv 0.11.24.
- Wheel and sdist build: passed. Packaged MedCPT and bge-m3 catalogs resolve.
- Isolated installed-wheel CLI smoke from `/tmp`, with checkout imports forbidden:
  passed. Covers synthetic FHIR import, real LanceDB indexing, adult/pediatric
  filtering, ranking, report content, resume preserving ranking output, and
  public checksum verification. No model downloads or clinical data.
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
Hosted CI and release execution are separate from these local checks; no production
release, model migration, or application deployment was performed.
