from __future__ import annotations

import importlib.util
import pkgutil

import pytest

import trialmatchai


def test_trialmatchai_imports_with_version():
    assert trialmatchai.__version__ == "0.2.0"


def test_matcher_namespace_is_removed():
    assert importlib.util.find_spec("Matcher") is None


def test_trialmatchai_modules_import_with_core_dependencies():
    failures = []
    for module in pkgutil.walk_packages(
        trialmatchai.__path__,
        prefix=f"{trialmatchai.__name__}.",
    ):
        try:
            __import__(module.name)
        except Exception as exc:  # pragma: no cover - assertion path
            failures.append(f"{module.name}: {type(exc).__name__}: {exc}")

    assert failures == []


def test_command_group_includes_bootstrap_data(monkeypatch, capsys):
    from trialmatchai.cli.main import main

    monkeypatch.setattr("sys.argv", ["trialmatchai", "--help"])
    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "bootstrap-data" in output
    assert "import-patient" in output
