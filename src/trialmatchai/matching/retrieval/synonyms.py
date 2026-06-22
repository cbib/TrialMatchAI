"""Shared disease-synonym extraction used by the first- and second-stage retrievers."""

from __future__ import annotations

from typing import Any, List

from trialmatchai.utils.logging_config import setup_logging

logger = setup_logging(__name__)


def disease_synonyms(entity_annotator: Any, condition: str) -> List[str]:
    """Return linked disease synonyms for a condition via the entity annotator.

    Returns an empty list when the annotator is unavailable or extraction fails.
    """
    if entity_annotator is None:
        logger.info("Entity annotator disabled; skipping synonyms extraction.")
        return []
    try:
        raw_result = entity_annotator.annotate_texts_in_parallel(
            [condition], max_workers=1
        )
        if raw_result and raw_result[0]:
            synonyms: set[str] = set()
            for entity in raw_result[0]:
                if entity.get("entity_group", "").lower() == "disease":
                    synonyms.update(entity.get("synonyms", []))
            return list(synonyms)
        logger.warning("No annotations found for condition: %s", condition)
    except Exception as exc:
        logger.error("Entity synonym extraction failed for '%s': %s", condition, exc)
    return []
