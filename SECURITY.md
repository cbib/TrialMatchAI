# Security Policy

## Reporting a vulnerability

Please report security issues privately via GitHub's **Security advisories**
(*Security → Report a vulnerability*) on this repository, or by email to
**abdallahmajd7@gmail.com**. Do not open a public issue for security reports.
We aim to acknowledge within a few business days.

## Supported versions

The latest released `0.x` line is supported. TrialMatchAI is research software,
not a medical device — see the disclaimer in the README.

## Dependency security posture

The **core** install (`pip install trialmatchai`) depends only on a small, current
set of libraries (requests, urllib3, certifi, numpy, pandas, pyarrow, pydantic,
lancedb, PyYAML). These are kept patched and are covered by the CI `pip-audit` gate.

The heavy machine-learning runtime — **vLLM, PyTorch, and their transitive
dependencies** — ships only in the **optional extras** (`gpu`, `llm`, `entity`,
`finetune`). These are **pinned** (`vllm==0.23.0`, `torch==2.11.0`) because the
inference stack (LoRA-served adapters, CoT/reranker engines) requires a specific,
mutually-compatible combination.

GitHub Dependabot reports advisories against these pinned ML packages. As of this
writing they have **no patched release available** (e.g. the `vllm 0.23.0` and
`torch 2.11.0` advisories list no fixed version), so they cannot be resolved by a
version bump. They affect only deployments that install the GPU/LLM extras, never
the base install. We therefore **accept and track** them, and revisit the vLLM/Torch
pin as a deliberate, GPU-validated upgrade when a compatible patched stack exists.

If you run the full stack in a sensitive environment, isolate the GPU inference
service (it is designed to run on a single self-hosted GPU host) and keep it off
untrusted networks.
