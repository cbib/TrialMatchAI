.PHONY: audit bootstrap build clean healthcheck index lint lock release-check sync sync-model test update-registry

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
	uv run pip-audit --progress-spinner off --ignore-vuln CVE-2025-3000

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
	uv run python scripts/scan_secrets.py
	uv run pip-audit --progress-spinner off --ignore-vuln CVE-2025-3000

clean:
	rm -rf build dist src/*.egg-info .pytest_cache .ruff_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
