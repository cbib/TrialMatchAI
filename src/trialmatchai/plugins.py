"""Lightweight component registry: map a ``type`` string to a factory callable.

Adding a new swappable backend (search store, embedder, LLM engine, reranker, …) becomes one
``@register(kind, name)`` decorator on its factory instead of widening a ``Literal`` and editing
every by-name construction site. Selection then flows through ``resolve(kind, name)``.

The registry holds only callables and has no heavy imports, so importing it is cheap and cannot
create a cycle. Concrete backends self-register at import of their defining module; the package
``__init__`` that exposes a ``build_*`` factory is responsible for importing those modules so the
names are populated before ``resolve`` runs.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List

_REGISTRIES: Dict[str, Dict[str, Callable[..., Any]]] = {}


def register(kind: str, name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator: register ``factory`` under ``(kind, name)``. Returns the factory unchanged."""
    table = _REGISTRIES.setdefault(kind, {})

    def _decorator(factory: Callable[..., Any]) -> Callable[..., Any]:
        table[name] = factory
        return factory

    return _decorator


def resolve(kind: str, name: str) -> Callable[..., Any]:
    """Return the factory registered for ``(kind, name)`` or raise a helpful error listing names."""
    table = _REGISTRIES.get(kind, {})
    if name not in table:
        known = ", ".join(sorted(table)) or "(none registered)"
        raise ValueError(f"unknown {kind} type {name!r}; registered: {known}")
    return table[name]


def registered(kind: str) -> List[str]:
    """Sorted names registered for ``kind`` (used by config validators for a helpful message)."""
    return sorted(_REGISTRIES.get(kind, {}))
