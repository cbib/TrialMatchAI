from __future__ import annotations

import json
import os
from importlib import resources
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

from trialmatchai.config.settings import TrialMatchSettings, apply_env_overrides
from trialmatchai.utils.logging_config import setup_logging

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None


logger = setup_logging(__name__)


DEFAULT_CONFIG_RELATIVE_PATH = Path("src/trialmatchai/config/config.json")
CONFIG_RELATIVE_PATHS = (
    Path("trialmatchai/config/config.json"),
)


def load_config(config_path: str | os.PathLike[str] | None = None) -> Dict[str, Any]:
    """Load and validate configuration from a JSON file with env overrides."""
    if load_dotenv:
        load_dotenv(_repo_root() / ".env")
        load_dotenv()

    resolved_config = resolve_config_path(config_path)
    with resolved_config.open("r", encoding="utf-8") as f:
        raw: Dict[str, Any] = json.load(f)
    raw = apply_env_overrides(deepcopy(raw))
    raw = _expand_catalog(raw)
    settings = TrialMatchSettings.model_validate(raw)
    # Non-lossy load: overlay the validated dump ONTO raw rather than replacing it, so declared
    # fields take validated values (coercions + defaults win) while undeclared knobs survive to
    # `.get()` consumers instead of being dropped by model_dump (a pure-config knob needs no field).
    cfg = _deep_merge(raw, settings.to_dict())
    _warn_unknown_config_keys(raw, settings)
    cfg = normalize_config_paths(cfg, resolved_config)
    return cfg


_CATALOGS = {"embedder": "embedders.json"}


def _load_catalog(filename: str) -> Dict[str, Any]:
    path = Path(__file__).resolve().parent / "catalog" / filename
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Could not read model catalog %s (ignoring).", path)
        return {}


def _expand_catalog(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Expand a ``model: "<name>"`` shorthand into the full catalog spec (e.g.
    ``embedder: {model: "medcpt"}`` resolves dim/metric/pooling/templates).

    Resolution order: inline > catalog > code defaults. An unknown name falls back to
    ``model_name`` so the raw-path form (``{model_name: "org/model"}``) keeps working. Runs before
    validation and the non-lossy merge, so downstream sees only the expanded section.
    """
    for section, catalog_file in _CATALOGS.items():
        block = raw.get(section)
        if not isinstance(block, dict):
            continue
        name = block.get("model")
        if not name:
            continue
        entry = _load_catalog(catalog_file).get(name)
        if entry is None:
            logger.warning(
                "config[%s].model=%r not in catalog; using it as a raw model_name.", section, name
            )
            block.setdefault("model_name", name)
            block.pop("model", None)
            continue
        merged = dict(entry)
        for key, value in block.items():
            if key != "model":
                merged[key] = value  # inline wins over catalog
        raw[section] = merged
    return raw


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` onto ``base``; ``override`` wins on leaf conflicts.

    Overlays the validated dump onto raw config so validated values win for declared keys while
    undeclared keys survive. Only dict/dict pairs recurse; other types (lists included) are
    replaced wholesale by ``override``'s value.
    """
    merged = dict(base)
    for key, over_val in override.items():
        base_val = merged.get(key)
        if isinstance(base_val, dict) and isinstance(over_val, dict):
            merged[key] = _deep_merge(base_val, over_val)
        else:
            merged[key] = over_val
    return merged


def _warn_unknown_config_keys(raw: Dict[str, Any], settings: TrialMatchSettings) -> None:
    """Log config keys that survive passthrough but aren't in the schema, so a typo is visible.

    Non-lossy loading keeps mistyped keys rather than dropping them; this warning replaces the
    previously silent drop and never removes anything.
    """
    fields = type(settings).model_fields
    # A section is "known" under its field name OR its serialization alias (e.g. ``global_`` is
    # written ``global`` in JSON), so an aliased section isn't mis-flagged as a typo.
    known_sections = set(fields) | {f.alias for f in fields.values() if f.alias}
    for section in raw:
        if section not in known_sections:
            logger.warning(
                "config: unknown top-level section %r (kept but not schema-validated — check for a typo)",
                section,
            )
    for name in fields:
        sub = getattr(settings, name, None)
        extra = getattr(sub, "model_extra", None)
        if extra:
            logger.warning(
                "config[%s]: passthrough keys not in schema (kept): %s",
                name,
                ", ".join(sorted(extra)),
            )


def resolve_config_path(
    config_path: str | os.PathLike[str] | None = None,
) -> Path:
    """Resolve explicit, repo-root, and packaged config paths."""
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
                    root / "src" / supplied,
                ]
            )
            if supplied in CONFIG_RELATIVE_PATHS:
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
    for key in ("output_dir", "trials_json_folder"):
        value = cfg.get("paths", {}).get(key)
        if value:
            cfg["paths"][key] = str(_resolve_local_path(value, root))

    patient_inputs = cfg.get("patient_inputs", {})
    for key in ("raw_dir", "profile_dir", "summary_dir"):
        value = patient_inputs.get(key)
        if value:
            patient_inputs[key] = str(_resolve_local_path(value, root))

    schema_path = cfg.get("entity_extraction", {}).get("schema_path")
    if schema_path:
        cfg["entity_extraction"]["schema_path"] = str(
            _resolve_trialmatchai_resource_or_local_path(schema_path, root)
        )

    concept_db_path = cfg.get("concept_linker", {}).get("db_path")
    if concept_db_path:
        cfg["concept_linker"]["db_path"] = str(
            _resolve_local_path(concept_db_path, root)
        )

    search_db_path = cfg.get("search_backend", {}).get("db_path")
    if search_db_path:
        cfg["search_backend"]["db_path"] = str(
            _resolve_local_path(search_db_path, root)
        )

    registry_cfg = cfg.get("registry", {})
    for key in ("keywords_file", "raw_dir", "manifest_path", "reports_dir"):
        value = registry_cfg.get(key)
        if value:
            registry_cfg[key] = str(_resolve_local_path(value, root))

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


def _resolve_trialmatchai_resource_or_local_path(value: str, root: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()

    local_path = (root / path).resolve()
    if local_path.exists():
        return local_path

    resource_path = _trialmatchai_resource_path(path)
    if resource_path is not None:
        return resource_path

    return local_path


def _trialmatchai_resource_path(path: Path) -> Path | None:
    parts = path.parts
    if parts[:2] == ("src", "trialmatchai"):
        relative = Path(*parts[2:])
    elif parts and parts[0] == "trialmatchai":
        relative = Path(*parts[1:])
    else:
        relative = path

    if not relative.parts or relative.parts[0] not in {
        "config",
        "entity_schemas",
    }:
        return None

    resource = resources.files("trialmatchai").joinpath(*relative.parts)
    if not resource.exists():
        return None
    return Path(str(resource)).resolve()


def _repo_root(anchor: Path | None = None) -> Path:
    start = (anchor or Path(__file__).resolve()).resolve()
    if start.is_file():
        start = start.parent
    for parent in (start, *start.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd().resolve()
