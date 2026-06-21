.PHONY: venv sync sync-gpu test lock lint audit healthcheck bootstrap index run setup

venv:
	uv venv

sync:
	uv sync

sync-gpu:
	uv sync --extra gpu

test:
	uv run pytest

lock:
	uv lock

lint:
	uv run python -m ruff check .

audit:
	uv run pip-audit --progress-spinner off --ignore-vuln CVE-2025-3000

healthcheck:
	uv run trialmatchai-healthcheck

bootstrap:
	uv run trialmatchai-bootstrap-data

index:
	uv run trialmatchai-index

run:
	uv run trialmatchai-run

setup:
	bash setup.sh
