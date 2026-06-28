# Release Checklist

TrialMatchAI uses a `src/` package layout and exposes the public package name `trialmatchai`.

## Local Checks

```bash
uv lock --check
uv run ruff check .
uv run pytest
uv build
uv run pre-commit run --all-files
uv run pip-audit --progress-spinner off --ignore-vuln CVE-2025-3000
```

Smoke test console commands:

```bash
uv run trialmatchai --help
uv run trialmatchai-healthcheck --help
uv run trialmatchai-index --help
uv run trialmatchai-build-concepts --help
uv run trialmatchai-update-registry --help
uv run trialmatchai-run --help
```

Wheel install smoke:

```bash
uv build
python -m venv /tmp/trialmatchai-wheel-smoke
/tmp/trialmatchai-wheel-smoke/bin/pip install dist/trialmatchai-*.whl
/tmp/trialmatchai-wheel-smoke/bin/python -c "import trialmatchai; print(trialmatchai.__version__)"
```
