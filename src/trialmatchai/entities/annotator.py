from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence

from trialmatchai.entities.linker import ConceptLinker, LanceDBConceptStore
from trialmatchai.entities.recognizers import EntityRecognizer, build_recognizer
from trialmatchai.entities.schemas import load_entity_schemas
from trialmatchai.entities.types import EntityAnnotation
from trialmatchai.utils.logging_config import setup_logging

logger = setup_logging(__name__)


class SchemaEntityAnnotator:
    def __init__(
        self,
        recognizer: EntityRecognizer,
        schemas: Sequence[Any],
        *,
        linker: ConceptLinker | None = None,
    ):
        self.recognizer = recognizer
        self.schemas = list(schemas)
        self.linker = linker

    def annotate_texts(self, texts: Sequence[str]) -> list[list[EntityAnnotation]]:
        recognized = self.recognizer.recognize(texts, self.schemas)
        if self.linker is None:
            return recognized
        return [self.linker.link_annotations(annotations) for annotations in recognized]

    def annotate_text(self, text: str) -> list[dict[str, Any]]:
        return [annotation.to_dict() for annotation in self.annotate_texts([text])[0]]

    def annotate_texts_in_parallel(
        self,
        texts: Sequence[str],
        max_workers: int = 20,
    ) -> list[list[dict[str, Any]]]:
        if max_workers <= 1 or len(texts) <= 1:
            return [
                [annotation.to_dict() for annotation in annotations]
                for annotations in self.annotate_texts(texts)
            ]

        results: list[list[dict[str, Any]]] = [[] for _ in texts]
        with ThreadPoolExecutor(max_workers=min(max_workers, len(texts))) as executor:
            future_to_index = {
                executor.submit(self.annotate_text, text): index
                for index, text in enumerate(texts)
            }
            for future, index in future_to_index.items():
                try:
                    results[index] = future.result()
                except Exception as exc:
                    logger.exception("Entity annotation failed for text index %s", index)
                    results[index] = [
                        {"error_code": 1, "error_message": str(exc)}
                    ]
        return results


def build_entity_annotator(
    config: dict[str, Any],
    *,
    embedder: Any | None = None,
) -> SchemaEntityAnnotator:
    extraction_cfg = dict(config.get("entity_extraction") or {})
    linker_cfg = dict(config.get("concept_linker") or {})
    schema_path = extraction_cfg.get("schema_path")
    schemas = load_entity_schemas(schema_path)
    recognizer = build_recognizer(extraction_cfg)

    linker: ConceptLinker | None = None
    if linker_cfg.get("enabled", True):
        store = _build_concept_store(linker_cfg, embedder=embedder)
        linker = ConceptLinker(
            store,
            schemas,
            accept_threshold=float(linker_cfg.get("accept_threshold", 0.8)),
            reject_threshold=float(linker_cfg.get("reject_threshold", 0.3)),
            search_limit=int(linker_cfg.get("search_limit", 10)),
        )

    return SchemaEntityAnnotator(recognizer, schemas, linker=linker)


def _build_concept_store(
    linker_cfg: dict[str, Any],
    *,
    embedder: Any | None,
) -> LanceDBConceptStore | None:
    db_path = linker_cfg.get("db_path")
    if not db_path:
        logger.warning("Concept linker enabled but concept_linker.db_path is empty.")
        return None

    path = Path(db_path)
    if not path.exists():
        logger.warning("Concept DB path does not exist; linking will degrade: %s", path)
        return None

    try:
        return LanceDBConceptStore(
            path,
            table_name=linker_cfg.get("table", "concepts"),
            embedder=embedder,
        )
    except Exception as exc:
        logger.warning("Concept DB unavailable; linking will degrade: %s", exc)
        return None
