"""Link already-extracted entities in the prepared criteria to concept IDs.

``prepare`` extracts entities and, when a concept store is available, links them in
the same pass. When the store is built *after* prepare (so prepare ran NER-only,
leaving entities with ``linker_status="concept_store_unavailable"``), this stage
links the persisted entities against the store **without re-running NER**: it
reconstructs each ``EntityAnnotation`` from its stored fields, runs the
``ConceptLinker``, and writes the enriched entities back in place.

Idempotent: only entities whose link decision was never made against a store are
(re)linked; pass ``force`` to redo all of them. Only criteria of trials whose
prepare is complete (the trial JSON marker exists) are touched, so a concurrent
``prepare`` writer is never disturbed.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from trialmatchai.utils.logging_config import setup_logging

logger = setup_logging(__name__)

# Statuses that mean "no link decision was ever made against a store" -> (re)link.
# A linked entity is one of accepted / ambiguous / rejected / not_linkable.
_RELINK_STATUSES = frozenset({"not_linked", "concept_store_unavailable"})


def link_corpus(
    config: dict[str, Any],
    *,
    processed_criteria_folder: str | Path = "data/processed_criteria",
    processed_trials_folder: str | Path = "data/processed_trials",
    force: bool = False,
    log_every: int = 500,
) -> dict[str, int]:
    """Link the entities in the prepared criteria to concept IDs, in place.

    Returns a tally of entity link statuses produced this run.
    """
    linker_cfg = dict(config.get("concept_linker") or {})
    if not linker_cfg.get("enabled", True):
        logger.info("link: concept_linker disabled in config; skipping.")
        return {}

    linker = _build_linker(config, linker_cfg)
    if linker is None or linker.store is None:
        logger.warning(
            "link: concept store unavailable (db_path=%s); nothing linked. "
            "Build it first with `trialmatchai build --concepts`.",
            linker_cfg.get("db_path"),
        )
        return {}

    criteria_root = Path(processed_criteria_folder)
    trials_root = Path(processed_trials_folder)
    if not criteria_root.exists():
        logger.warning("link: %s does not exist; nothing to link.", criteria_root)
        return {}

    trial_dirs = [d for d in sorted(criteria_root.iterdir()) if d.is_dir()]
    tally: dict[str, int] = {}
    cache: dict[tuple[str, str], dict[str, Any]] = {}
    relinked_trials = skipped_trials = 0

    for i, trial_dir in enumerate(trial_dirs, start=1):
        if not (trials_root / f"{trial_dir.name}.json").exists():
            skipped_trials += 1  # prepare not finished for this trial yet
            continue
        if _link_trial_dir(trial_dir, linker, cache, tally, force=force):
            relinked_trials += 1
        if i % log_every == 0:
            logger.info(
                "link progress: %s/%s trials (%s relinked); "
                "accepted=%s ambiguous=%s rejected=%s",
                i,
                len(trial_dirs),
                relinked_trials,
                tally.get("accepted", 0),
                tally.get("ambiguous", 0),
                tally.get("rejected", 0),
            )

    logger.info(
        "link complete: %s trials relinked, %s skipped (prepare in-flight). entities: %s",
        relinked_trials,
        skipped_trials,
        {k: tally[k] for k in sorted(tally)},
    )
    return tally


def _build_linker(config: dict[str, Any], linker_cfg: dict[str, Any]):
    from trialmatchai.entities.linker import (
        ConceptLinker,
        LanceDBConceptStore,
        lexical_reranker,
    )
    from trialmatchai.entities.schemas import load_entity_schemas

    db_path = linker_cfg.get("db_path")
    if not db_path or not Path(db_path).exists():
        return None
    from trialmatchai.models.embedding import build_embedder

    embedder = build_embedder(config)
    try:
        store = LanceDBConceptStore(
            db_path, table_name=linker_cfg.get("table", "concepts"), embedder=embedder
        )
    except Exception as exc:
        logger.warning("link: could not open concept store: %s", exc)
        return None

    schema_path = (config.get("entity_extraction") or {}).get("schema_path")
    use_path = schema_path if schema_path and Path(schema_path).exists() else None
    schemas = load_entity_schemas(use_path)
    rerank_mode = str(linker_cfg.get("rerank", "lexical")).lower()
    return ConceptLinker(
        store,
        schemas,
        accept_threshold=float(linker_cfg.get("accept_threshold", 0.7)),
        reject_threshold=float(linker_cfg.get("reject_threshold", 0.5)),
        margin=float(linker_cfg.get("margin", 0.05)),
        reranker=lexical_reranker if rerank_mode == "lexical" else None,
        search_limit=int(linker_cfg.get("search_limit", 10)),
    )


def _link_trial_dir(trial_dir: Path, linker, cache, tally, *, force: bool) -> bool:
    changed_any = False
    for criterion_file in sorted(trial_dir.glob("*.json")):
        try:
            row = json.loads(criterion_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        entities = row.get("entities")
        if not isinstance(entities, list) or not entities:
            continue
        new_entities, changed = _link_entities(entities, linker, cache, tally, force=force)
        if changed:
            row["entities"] = new_entities
            _atomic_write_json(criterion_file, row)
            changed_any = True
    return changed_any


def _link_entities(entities, linker, cache, tally, *, force: bool):
    from trialmatchai.entities.types import EntityAnnotation

    out: list[Any] = []
    changed = False
    for entity in entities:
        if not isinstance(entity, dict):
            out.append(entity)
            continue
        status = entity.get("linker_status", "not_linked")
        if not force and status not in _RELINK_STATUSES:
            out.append(entity)
            continue
        group = str(entity.get("entity_group") or entity.get("class") or "")
        text = str(entity.get("text") or entity.get("entity") or "")
        if not text:
            out.append(entity)
            continue

        key = (group, text.casefold())
        fields = cache.get(key)
        if fields is None:
            linked = linker.link_annotation(
                EntityAnnotation(
                    entity_group=group,
                    text=text,
                    start=int(entity.get("start") or 0),
                    end=int(entity.get("end") or 0),
                    score=float(entity.get("score") or 0.0),
                )
            )
            fields = {
                "normalized_id": list(linked.normalized_id),
                "synonyms": list(linked.synonyms),
                "concept_candidates": [c.to_dict() for c in linked.concept_candidates],
                "linker_score": linked.linker_score,
                "linker_status": linked.linker_status,
            }
            cache[key] = fields

        merged = dict(entity)
        merged.update(fields)
        merged["entity"] = merged.get("text", text)  # keep the index aliases consistent
        merged["class"] = merged.get("entity_group", group)
        out.append(merged)
        tally[fields["linker_status"]] = tally.get(fields["linker_status"], 0) + 1
        changed = True
    return out, changed


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-link-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(data, indent=2, sort_keys=True))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
