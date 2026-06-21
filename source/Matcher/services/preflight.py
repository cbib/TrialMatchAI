from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch

from Matcher.utils.logging_config import setup_logging

logger = setup_logging(__name__)


def run_preflight_checks(
    config: Dict[str, Any],
    *,
    es_client: Any | None = None,
    require_patient_inputs: bool = False,
    require_trials_json: bool = False,
    require_models: bool = False,
    require_indices: bool = False,
) -> List[str]:
    """Return blocking deployment/runtime issues discovered before heavy startup."""
    issues: List[str] = []
    paths = config.get("paths", {})

    _require_path(
        issues,
        "paths.patients_dir",
        paths.get("patients_dir"),
        required=require_patient_inputs,
    )
    _require_path(
        issues,
        "paths.trials_json_folder",
        paths.get("trials_json_folder"),
        required=require_trials_json,
    )
    _require_output_dir(issues, paths.get("output_dir"))

    host = str(config.get("elasticsearch", {}).get("host", ""))
    if host.startswith("https://"):
        _require_path(
            issues,
            "paths.docker_certs",
            paths.get("docker_certs"),
            required=True,
        )

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
                    "(`uv sync --extra gpu`) or the Docker worker image."
                )
            if not torch.cuda.is_available():
                issues.append("cot_backend=vllm requires a CUDA-capable runtime.")

    if es_client is not None:
        if not _ping(es_client):
            issues.append(
                f"Elasticsearch is not reachable at {config['elasticsearch']['host']}."
            )
        elif require_indices:
            missing = _missing_indices(
                es_client,
                [
                    config["elasticsearch"]["index_trials"],
                    config["elasticsearch"]["index_trials_eligibility"],
                ],
            )
            if missing:
                issues.append("Missing Elasticsearch indices: " + ", ".join(missing))

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


def _ping(es_client: Any) -> bool:
    try:
        return bool(es_client.ping())
    except Exception:
        return False


def _missing_indices(es_client: Any, names: Iterable[str]) -> list[str]:
    missing: list[str] = []
    for name in names:
        try:
            if not es_client.indices.exists(index=name):
                missing.append(name)
        except Exception as exc:
            logger.warning("Could not check Elasticsearch index %s: %s", name, exc)
            missing.append(name)
    return missing
