"""Shared text helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def flatten_text(value: Any) -> str:
    """Flatten a possibly-nested value (str/mapping/sequence) into whitespace-
    normalized text. Used for both trial indexing and backend search-text building.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, Mapping):
        return " ".join(flatten_text(item) for item in value.values()).strip()
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        return " ".join(flatten_text(item) for item in value).strip()
    return str(value)
