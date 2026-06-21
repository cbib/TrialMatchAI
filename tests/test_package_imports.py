from __future__ import annotations

import warnings

import trialmatchai


def test_trialmatchai_imports_with_version():
    assert trialmatchai.__version__ == "0.2.0"


def test_matcher_config_compatibility_shim():
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        from Matcher.config.config_loader import resolve_config_path

    assert resolve_config_path().name == "config.json"
    assert any("Matcher" in str(warning.message) for warning in captured)
