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
        patient_cfg = config.get("patient_inputs", {})
        _require_path(
            issues,
            "patient_inputs.profile_dir",
            patient_cfg.get("profile_dir"),
            required=False,
        )
    _require_path(
        issues,
        "paths.trials_json_folder",
        paths.get("trials_json_folder"),
        required=require_trials_json,
    )
    _require_output_dir(issues, paths.get("output_dir"))

    entity_cfg = config.get("entity_extraction")
    if entity_cfg:
        _require_path(
            issues,
            "entity_extraction.schema_path",
            entity_cfg.get("schema_path"),
            required=True,
        )

    if require_models:
        reranker_enabled = _reranker_enabled(config)
        rag_enabled = _rag_enabled(config)
        reranker_backend = _reranker_backend(config)
        rag_backend = _rag_backend(config)
        if entity_cfg:
            backend = entity_cfg.get("backend", "gliner2")
            if backend == "gliner2" and importlib.util.find_spec("gliner2") is None:
                issues.append(
                    "entity_extraction.backend=gliner2 requires the entity extra "
                    "(`uv sync --extra entity`)."
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
        if rag_enabled and rag_backend == "vllm":
            # Optional: a base CoT model (no LoRA) runs with cot_adapter_path unset/empty.
            _require_path(
                issues,
                "model.cot_adapter_path",
                model_cfg.get("cot_adapter_path"),
                required=False,
            )
        if reranker_enabled and reranker_backend == "vllm":
            _require_path(
                issues,
                "model.reranker_adapter_path",
                model_cfg.get("reranker_adapter_path"),
                required=True,
            )
        needs_vllm = (rag_enabled and rag_backend == "vllm") or (
            reranker_enabled and reranker_backend == "vllm"
        )
        needs_transformers = (rag_enabled and rag_backend == "transformers") or (
            reranker_enabled and reranker_backend == "transformers"
        )
        # vLLM is the production backend; CPU smoke configs use Transformers (no CUDA).
        if needs_vllm:
            vllm_available = importlib.util.find_spec("vllm") is not None
            if not vllm_available:
                issues.append(
                    "vLLM is required (`uv sync --extra llm --extra gpu`)."
                )
            else:
                try:
                    import torch
                except Exception:
                    issues.append(
                        "vLLM requires PyTorch (`uv sync --extra llm --extra gpu`)."
                    )
                else:
                    if not torch.cuda.is_available():
                        issues.append("vLLM requires a CUDA-capable runtime.")
        if needs_transformers:
            if importlib.util.find_spec("torch") is None:
                issues.append(
                    "Transformers CPU backend requires PyTorch (`uv sync --extra llm`)."
                )
            if importlib.util.find_spec("transformers") is None:
                issues.append(
                    "Transformers CPU backend requires transformers (`uv sync --extra llm`)."
                )

        # Check gated base models up front so an HF auth failure surfaces here, not
        # after first-level search.
        issues.extend(
            check_hf_access(
                [model_cfg.get("base_model"), model_cfg.get("reranker_model_path")]
            )
        )

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
                from trialmatchai.search import build_search_backend

                search_backend = build_search_backend(config)
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


def check_cuda(config: Dict[str, Any], *, required: bool) -> List[str]:
    """Verify a CUDA GPU is present when the build/match needs one."""
    issues: List[str] = []
    if importlib.util.find_spec("torch") is None:
        if required:
            issues.append("PyTorch is not installed (`uv sync --extra gpu`).")
        return issues
    try:
        import torch

        available = bool(torch.cuda.is_available())
    except Exception as exc:  # pragma: no cover - torch import edge cases
        if required:
            issues.append(f"Could not initialize PyTorch/CUDA: {exc}")
        return issues
    if not available:
        message = "No CUDA GPU detected; embeddings/LLMs need a GPU host."
        if required:
            issues.append(message)
        else:
            logger.warning("Preflight: %s", message)
    return issues


def check_hf_access(model_ids: List[str | None]) -> List[str]:
    """Fast pre-download check that HuggingFace models are reachable/authorized.

    Metadata API only (no weight download); local paths and non-hub ids are skipped.
    Gated/missing repos error so auth fails in seconds; transient errors only warn.
    """
    issues: List[str] = []
    if importlib.util.find_spec("huggingface_hub") is None:
        return issues  # dependency checks elsewhere cover a missing stack
    import os

    from huggingface_hub import HfApi
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    api = HfApi(token=token)
    for model_id in model_ids:
        if not model_id or Path(model_id).exists() or "/" not in str(model_id):
            continue  # local model dir or not a hub repo id
        try:
            api.model_info(model_id)
        except GatedRepoError:
            issues.append(
                f"HuggingFace model '{model_id}' is gated/unauthorized — accept its "
                "license on huggingface.co and export HF_TOKEN."
            )
        except RepositoryNotFoundError:
            issues.append(
                f"HuggingFace model '{model_id}' was not found (check the id or your token)."
            )
        except Exception as exc:
            logger.warning("Preflight: could not verify HF model '%s': %s", model_id, exc)
    return issues


def run_build_preflight(config: Dict[str, Any]) -> List[str]:
    """Fail-fast checks for the build (setup) half before any heavy work."""
    issues: List[str] = []
    embedder_cfg = config.get("embedder", {})
    issues += check_cuda(config, required=bool(embedder_cfg.get("use_gpu", True)))

    entity_cfg = config.get("entity_extraction", {})
    if (
        entity_cfg.get("backend", "gliner2") == "gliner2"
        and importlib.util.find_spec("gliner2") is None
    ):
        issues.append("entity_extraction.backend=gliner2 requires `uv sync --extra entity`.")

    issues += check_hf_access([embedder_cfg.get("model_name"), entity_cfg.get("model_name")])
    for issue in issues:
        logger.error("Build preflight: %s", issue)
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
    patient_cfg = config.get("patient_inputs", {})
    raw = patient_cfg.get("profile_dir")
    if not raw:
        issues.append("patient_inputs.profile_dir is not configured.")
        return
    profile_dir = Path(raw)
    if not profile_dir.exists():
        issues.append(f"patient_inputs.profile_dir does not exist: {profile_dir}")


def _reranker_enabled(config: Dict[str, Any]) -> bool:
    return bool(config.get("LLM_reranker", {}).get("enabled", True))


def _rag_enabled(config: Dict[str, Any]) -> bool:
    if not bool(config.get("use_cot_reasoning", True)):
        return False
    return bool(config.get("rag", {}).get("enabled", True))


def _reranker_backend(config: Dict[str, Any]) -> str:
    return str(config.get("LLM_reranker", {}).get("backend", "vllm"))


def _rag_backend(config: Dict[str, Any]) -> str:
    return str(config.get("rag", {}).get("backend", "vllm"))
