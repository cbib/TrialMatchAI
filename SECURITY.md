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
`finetune`). These are **pinned** (`vllm==0.34.0`, `torch==2.14.0`,
`transformers==5.22.2`) because the inference stack (LoRA-served adapters,
CoT/reranker engines) requires a specific, mutually-compatible combination.

Because the pins are exact, advisories against the ML stack are resolved by a
deliberate, GPU-validated bump of the vLLM/Torch/Transformers set together (vLLM
pins an exact Torch and a Transformers range). These advisories affect only
deployments that install the GPU/LLM extras, never the base install. Transitive
dependencies are refreshed in `uv.lock` with `uv lock --upgrade-package <name>`.

If you run the full stack in a sensitive environment, isolate the GPU inference
service (it is designed to run on a single self-hosted GPU host) and keep it off
untrusted networks.
