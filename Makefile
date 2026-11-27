.PHONY: audit bootstrap build clean demo e2e healthcheck index lint lock release-check sync sync-model test update-registry

sync:
	uv sync

sync-model:
	uv sync --extra llm --extra gpu --extra entity

lock:
	uv lock --check

lint:
	uv run --frozen ruff check .

test:
	uv run --frozen pytest

audit:
	uv run --frozen pip-audit --progress-spinner off

build:
	uv build

healthcheck:
	uv run --frozen trialmatchai healthcheck --registry

bootstrap:
	uv run --frozen trialmatchai bootstrap-data

update-registry:
	uv run --frozen trialmatchai update-registry

index:
	uv run --frozen trialmatchai index --prepare

demo:
	uv run --frozen trialmatchai demo

e2e:
	uv run --frozen pytest -m e2e

release-check:
	uv run --frozen pre-commit run actionlint --all-files
	uv lock --check
	uv run --frozen ruff check .
	uv run --frozen pytest
	uv build
	uv run --frozen trialmatchai artifacts manifest dist
	uv run --frozen trialmatchai artifacts verify dist/SHA256SUMS --require-exact --json
	uv run --no-project --isolated --with "$$(realpath dist/trialmatchai-*.whl)" python -I scripts/installed_smoke.py --expect-installed --workspace "$$(mktemp -d /tmp/trialmatchai-wheel-smoke.XXXXXX)"
	uv run --frozen pre-commit run gitleaks --all-files
	uv run --frozen pip-audit --progress-spinner off

clean:
	rm -rf build dist src/*.egg-info .pytest_cache .ruff_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
