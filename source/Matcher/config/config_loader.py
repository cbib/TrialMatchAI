from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

from Matcher.config.settings import TrialMatchSettings, apply_env_overrides
from Matcher.utils.logging_config import setup_logging

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None


logger = setup_logging(__name__)


DEFAULT_CONFIG_RELATIVE_PATH = Path("source/Matcher/config/config.json")
LEGACY_CONFIG_RELATIVE_PATH = Path("Matcher/config/config.json")


def load_config(config_path: str | os.PathLike[str] | None = None) -> Dict[str, Any]:
    """Load and validate configuration from a JSON file with env overrides."""
    if load_dotenv:
        load_dotenv(_repo_root() / ".env")
        load_dotenv()

    resolved_config = resolve_config_path(config_path)
    with resolved_config.open("r", encoding="utf-8") as f:
        raw: Dict[str, Any] = json.load(f)
    raw = apply_env_overrides(deepcopy(raw))
    settings = TrialMatchSettings.model_validate(raw)
    cfg = settings.to_dict()
    cfg = normalize_config_paths(cfg, resolved_config)
    if cfg.get("elasticsearch", {}).get("password") in {"", "CHANGE_ME"}:
        logger.warning(
            "Elasticsearch password is not set. Use TRIALMATCHAI_ES_PASSWORD to supply it."
        )
    return cfg


def resolve_config_path(
    config_path: str | os.PathLike[str] | None = None,
) -> Path:
    """Resolve explicit, repo-root, legacy, and packaged config paths."""
    root = _repo_root()
    candidates: list[Path] = []
    if config_path:
        supplied = Path(config_path).expanduser()
        if supplied.is_absolute():
            candidates.append(supplied)
        else:
            candidates.extend(
                [
                    Path.cwd() / supplied,
                    root / supplied,
                    root / "source" / supplied,
                ]
            )
            if supplied == LEGACY_CONFIG_RELATIVE_PATH:
                candidates.append(root / DEFAULT_CONFIG_RELATIVE_PATH)
    else:
        candidates.extend(
            [
                root / DEFAULT_CONFIG_RELATIVE_PATH,
                Path(__file__).resolve().with_name("config.json"),
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    searched = ", ".join(str(path) for path in candidates)
    label = str(config_path) if config_path else str(DEFAULT_CONFIG_RELATIVE_PATH)
    raise FileNotFoundError(f"Configuration file not found: {label}. Searched: {searched}")


def normalize_config_paths(cfg: Dict[str, Any], config_path: Path) -> Dict[str, Any]:
    """Normalize known local paths while leaving remote model IDs untouched."""
    root = _repo_root(config_path)
    for key in ("patients_dir", "output_dir", "trials_json_folder", "docker_certs"):
        value = cfg.get("paths", {}).get(key)
        if value:
            cfg["paths"][key] = str(_resolve_local_path(value, root))

    schema_path = cfg.get("entity_extraction", {}).get("schema_path")
    if schema_path:
        cfg["entity_extraction"]["schema_path"] = str(
            _resolve_local_path(schema_path, root)
        )

    concept_db_path = cfg.get("concept_linker", {}).get("db_path")
    if concept_db_path:
        cfg["concept_linker"]["db_path"] = str(
            _resolve_local_path(concept_db_path, root)
        )

    start_script = cfg.get("elasticsearch", {}).get("start_script")
    if start_script:
        cfg["elasticsearch"]["start_script"] = str(_resolve_local_path(start_script, root))

    for key in ("cot_adapter_path", "reranker_adapter_path"):
        value = cfg.get("model", {}).get(key)
        if value:
            cfg["model"][key] = str(_resolve_local_path(value, root))

    return cfg


def _resolve_local_path(value: str, root: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (root / path).resolve()


def _repo_root(anchor: Path | None = None) -> Path:
    start = (anchor or Path(__file__).resolve()).resolve()
    if start.is_file():
        start = start.parent
    for parent in (start, *start.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd().resolve()
