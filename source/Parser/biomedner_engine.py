from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from Matcher.entities import build_entity_annotator


class BioMedNER:
    """Compatibility wrapper over the Python-native TrialMatchAI entity annotator."""

    def __init__(self, *args: Any, config: dict[str, Any] | None = None, **kwargs: Any):
        del args, kwargs
        self._annotator = build_entity_annotator(config or _default_config())

    @staticmethod
    def load_dictionary_file(file_path: str | Path) -> list[str]:
        with Path(file_path).open("r", encoding="utf-8") as handle:
            return handle.read().splitlines()

    def annotate_text(self, text: str) -> list[dict[str, Any]]:
        return self._annotator.annotate_text(text)

    def annotate_texts_in_parallel(
        self,
        texts: Sequence[str],
        max_workers: int = 20,
        retries: int = 1,
        delay: float = 0,
    ) -> list[list[dict[str, Any]]]:
        return self._annotator.annotate_texts_in_parallel(
            texts,
            max_workers=max_workers,
            retries=retries,
            delay=delay,
        )


def _default_config() -> dict[str, Any]:
    return {
        "entity_extraction": {
            "backend": "gliner2",
            "model_name": "fastino/gliner2-base",
            "fallback_model_name": "gliner-community/gliner_large-v2.5",
            "schema_path": str(_repo_root() / "source/Matcher/entity_schemas/trialmatchai.yaml"),
            "batch_size": 8,
            "device": "auto",
            "trust_remote_code": False,
        },
        "concept_linker": {
            "enabled": True,
            "db_path": str(_repo_root() / "data/concepts"),
            "table": "concepts",
            "accept_threshold": 0.8,
            "reject_threshold": 0.3,
            "search_limit": 10,
        },
    }


def _repo_root() -> Path:
    start = Path(__file__).resolve()
    for parent in start.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd().resolve()


def transform_results(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Best-effort adapter for callers that still pass PubTator-style JSON."""
    annotations = data.get("annotations", []) if isinstance(data, dict) else []
    results: list[dict[str, Any]] = []
    for annotation in annotations:
        span = annotation.get("span", {})
        results.append(
            {
                "entity_group": annotation.get("obj", ""),
                "score": annotation.get("prob", 0.0),
                "text": annotation.get("mention", ""),
                "start": span.get("begin", 0),
                "end": span.get("end", 0),
                "normalized_id": annotation.get("id", ["CUI-less"]),
                "synonyms": [],
                "concept_candidates": [],
                "linker_score": None,
                "linker_status": "not_linked",
            }
        )
    return results


def append_synonyms(
    ner_results: list[dict[str, Any]], dict_paths: dict[str, str | Path]
) -> None:
    del dict_paths
    for entity in ner_results:
        entity.setdefault("synonyms", [])


def get_synonyms_from_file(
    file_path: str | Path, entity_ids: Sequence[str]
) -> list[str]:
    del file_path, entity_ids
    return []
