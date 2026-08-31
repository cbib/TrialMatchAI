.PHONY: audit bootstrap build clean healthcheck index lint lock release-check sync sync-model test update-registry

# Advisories with no fixed version, published against the hard-pinned inference stack
# (vllm and its transitive diskcache). Kept in sync with the same list in
# .github/workflows/ci.yml -- if you change one, change the other. Revisit and drop these
# once vllm / diskcache ship patched releases.
PIP_AUDIT_IGNORES := --ignore-vuln CVE-2025-3000 --ignore-vuln CVE-2025-69872 --ignore-vuln CVE-2026-53923 --ignore-vuln CVE-2026-54236 --ignore-vuln CVE-2026-12491 --ignore-vuln CVE-2026-54235 --ignore-vuln CVE-2026-54233

sync:
	uv sync

sync-model:
	uv sync --extra llm --extra gpu --extra entity

lock:
	uv lock --check

lint:
	uv run ruff check .

test:
	uv run pytest

audit:
	uv run pip-audit --progress-spinner off $(PIP_AUDIT_IGNORES)

build:
	uv build

healthcheck:
	uv run trialmatchai-healthcheck --registry

bootstrap:
	uv run trialmatchai-bootstrap-data

update-registry:
	uv run trialmatchai-update-registry

index:
	uv run trialmatchai-index --prepare

release-check:
	uv lock --check
	uv run ruff check .
	uv run pytest
	uv build
	uv run pre-commit run gitleaks --all-files
	uv run pip-audit --progress-spinner off $(PIP_AUDIT_IGNORES)

clean:
	rm -rf build dist src/*.egg-info .pytest_cache .ruff_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
