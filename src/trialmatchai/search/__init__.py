from typing import Any, Mapping

from trialmatchai.plugins import resolve
from trialmatchai.search.lancedb_backend import (
    InMemorySearchBackend,
    LanceDBSearchBackend,
    SearchBackendUnavailable,
    build_criteria_record,
    build_trial_record,
)


def build_search_backend(config: Mapping[str, Any]):
    """Construct the search backend selected by ``search_backend.type`` (or legacy ``backend``).

    Single dispatch seam replacing by-name construction of ``LanceDBSearchBackend`` across the
    call sites: a new store registers a ``type`` in the plugin registry and is selectable with
    no edit here. Importing this module imports ``lancedb_backend``, which self-registers
    ``lancedb`` — the default when neither key is set.
    """
    sb = config.get("search_backend", {}) or {}
    name = sb.get("type") or sb.get("backend") or "lancedb"
    return resolve("search_backend", name)(config)


__all__ = [
    "InMemorySearchBackend",
    "LanceDBSearchBackend",
    "SearchBackendUnavailable",
    "build_criteria_record",
    "build_trial_record",
    "build_search_backend",
]
