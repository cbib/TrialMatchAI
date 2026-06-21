from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict, List

from trialmatchai.utils.logging_config import setup_logging

logger = setup_logging(__name__)


def run_preflight_checks(
    config: Dict[str, Any],
    *,
    search_backend: Any | None = None,
    require_patient_inputs: bool = False,
    require_trials_json: bool = False,
    require_models: bool = False,
    require_search_tables: bool = False,
) -> List[str]:
    """Return blocking deployment/runtime issues discovered before heavy startup."""
    issues: List[str] = []
    paths = config.get("paths", {})

    if require_patient_inputs:
        _require_patient_inputs(issues, config)
    else:
        _require_path(
            issues,
            "paths.patients_dir",
            paths.get("patients_dir"),
            required=False,
        )
    _require_path(
        issues,
        "paths.trials_json_folder",
        paths.get("trials_json_folder"),
        required=require_trials_json,
    )
    _require_output_dir(issues, paths.get("output_dir"))

    if require_models:
        entity_cfg = config.get("entity_extraction")
        if entity_cfg:
            _require_path(
                issues,
                "entity_extraction.schema_path",
                entity_cfg.get("schema_path"),
                required=True,
            )
            backend = entity_cfg.get("backend", "gliner2")
            if backend == "gliner2" and importlib.util.find_spec("gliner2") is None:
                issues.append(
                    "entity_extraction.backend=gliner2 requires the entity extra "
                    "(`uv sync --extra entity`)."
                )
            elif backend == "gliner" and importlib.util.find_spec("gliner") is None:
                issues.append(
                    "entity_extraction.backend=gliner requires the GLiNER dependency."
                )

        linker_cfg = config.get("concept_linker")
        if linker_cfg and linker_cfg.get("enabled", True):
            _require_path(
                issues,
                "concept_linker.db_path",
                linker_cfg.get("db_path"),
                required=False,
            )

        model_cfg = config.get("model", {})
        _require_path(
            issues,
            "model.cot_adapter_path",
            model_cfg.get("cot_adapter_path"),
            required=True,
        )
        _require_path(
            issues,
            "model.reranker_adapter_path",
            model_cfg.get("reranker_adapter_path"),
            required=True,
        )
        if config.get("cot_backend") == "vllm":
            if importlib.util.find_spec("vllm") is None:
                issues.append(
                    "cot_backend=vllm requires the GPU extra "
                    "(`uv sync --extra llm --extra gpu`)."
                )
            try:
                import torch
            except Exception:
                issues.append(
                    "cot_backend=vllm requires PyTorch "
                    "(`uv sync --extra llm --extra gpu`)."
                )
            else:
                if not torch.cuda.is_available():
                    issues.append("cot_backend=vllm requires a CUDA-capable runtime.")

    search_cfg = config.get("search_backend", {})
    if search_cfg:
        _require_path(
            issues,
            "search_backend.db_path",
            search_cfg.get("db_path"),
            required=require_search_tables,
        )
    if require_search_tables:
        if search_backend is None:
            try:
                from trialmatchai.search import LanceDBSearchBackend

                search_backend = LanceDBSearchBackend.from_config(config)
            except Exception as exc:
                issues.append(f"Search backend is not available: {exc}")
                search_backend = None
        if search_backend is not None:
            if hasattr(search_backend, "health"):
                issues.extend(search_backend.health(require_tables=True))
            else:
                issues.append("Search backend does not expose a healthcheck.")

    for issue in issues:
        logger.error("Preflight: %s", issue)
    return issues


def _require_path(
    issues: List[str],
    name: str,
    value: str | None,
    *,
    required: bool,
) -> None:
    if not value:
        if required:
            issues.append(f"{name} is not configured.")
        return
    path = Path(value)
    if required and not path.exists():
        issues.append(f"{name} does not exist: {path}")
    elif not path.exists():
        logger.warning("Preflight: optional path does not exist: %s=%s", name, path)


def _require_output_dir(issues: List[str], value: str | None) -> None:
    if not value:
        issues.append("paths.output_dir is not configured.")
        return
    path = Path(value)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        issues.append(f"paths.output_dir is not writable: {path} ({exc})")


def _require_patient_inputs(issues: List[str], config: Dict[str, Any]) -> None:
    paths = config.get("paths", {})
    patient_cfg = config.get("patient_inputs", {})
    profile_dir = Path(patient_cfg.get("profile_dir", ""))
    legacy_dir = Path(paths.get("patients_dir", ""))
    has_profiles = profile_dir.exists()
    has_legacy = legacy_dir.exists()
    if has_profiles or has_legacy:
        return
    if profile_dir:
        issues.append(f"patient_inputs.profile_dir does not exist: {profile_dir}")
    if legacy_dir:
        issues.append(f"paths.patients_dir does not exist: {legacy_dir}")
