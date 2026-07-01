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
from trialmatchai.utils.pipeline_state import dir_fingerprint

logger = setup_logging(__name__)

# Statuses that mean "no link decision was ever made against a store" -> (re)link.
# A linked entity is one of accepted / ambiguous / rejected / not_linkable.
_RELINK_STATUSES = frozenset({"not_linked", "concept_store_unavailable"})

# Per-trial resume journal: an append-only write-ahead log of trial IDs already linked, so a
# crashed/partial run resumes by SKIPPING those trials by id (no re-reading their criteria)
# instead of re-walking the whole corpus. This is the checkpoint / committed-cursor pattern --
# durable per-record receipts + idempotent re-processing. A header line binds the log to the
# prepared-corpus fingerprint, so it is discarded when the corpus changes.
_LINK_PROGRESS_VERSION = "1"
_LINK_JOURNAL_NAME = ".link_progress.jsonl"


def _load_link_journal(path: Path, corpus_fingerprint: str) -> tuple[bool, set[str]]:
    """Return (header_valid, completed_trial_ids).

    header_valid is False when the journal is missing, a different version, or was recorded
    against a different corpus -- the caller then starts a fresh journal. A torn final record
    from a crash is skipped, not fatal.
    """
    if not corpus_fingerprint:
        return False, set()
    try:
        with path.open("r", encoding="utf-8") as handle:
            header = handle.readline()
            meta = json.loads(header) if header.strip() else {}
            if (
                meta.get("v") != _LINK_PROGRESS_VERSION
                or meta.get("corpus_fingerprint") != corpus_fingerprint
            ):
                return False, set()
            completed: set[str] = set()
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    completed.add(json.loads(line)["nct"])
                except Exception:
                    continue  # torn final record from a crash -> ignore
            return True, completed
    except FileNotFoundError:
        return False, set()
    except Exception:
        return False, set()


def _write_link_journal_header(path: Path, corpus_fingerprint: str) -> None:
    """(Re)start the journal with a fresh header binding it to the current prepared corpus."""
    path.write_text(
        json.dumps({"v": _LINK_PROGRESS_VERSION, "corpus_fingerprint": corpus_fingerprint}) + "\n",
        encoding="utf-8",
    )


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
    relinked_trials = skipped_trials = resumed_skip = 0

    # Per-trial resume: skip trials already linked in a prior run BY ID (via the append-only
    # journal), without re-reading their criteria. The journal is bound to the prepared-corpus
    # fingerprint, so it is discarded if the corpus changed; ``force`` ignores it.
    journal_path = criteria_root / _LINK_JOURNAL_NAME
    corpus_fp = dir_fingerprint(trials_root)
    header_valid, completed = (
        (False, set()) if force else _load_link_journal(journal_path, corpus_fp)
    )
    if not header_valid:
        _write_link_journal_header(journal_path, corpus_fp)
        completed = set()
    if completed:
        logger.info(
            "link resume: %s of %s trials already linked in a prior run; skipping those by id.",
            len(completed),
            len(trial_dirs),
        )

    # Line-buffered append log: each completed trial is a durable receipt flushed on newline,
    # so a process kill (OOM / timeout) resumes from the last completed trial, not the start.
    journal = journal_path.open("a", buffering=1, encoding="utf-8")
    try:
        for i, trial_dir in enumerate(trial_dirs, start=1):
            name = trial_dir.name
            if not (trials_root / f"{name}.json").exists():
                skipped_trials += 1  # prepare not finished for this trial yet
                continue
            if name in completed:
                resumed_skip += 1  # linked in a prior run -- skip without opening its criteria
                continue
            if _link_trial_dir(trial_dir, linker, cache, tally, force=force):
                relinked_trials += 1
            completed.add(name)
            journal.write(json.dumps({"nct": name}) + "\n")
            if i % log_every == 0:
                logger.info(
                    "link progress: %s/%s trials (%s relinked, %s resumed-skip); "
                    "accepted=%s ambiguous=%s rejected=%s",
                    i,
                    len(trial_dirs),
                    relinked_trials,
                    resumed_skip,
                    tally.get("accepted", 0),
                    tally.get("ambiguous", 0),
                    tally.get("rejected", 0),
                )
    finally:
        try:
            journal.flush()
            os.fsync(journal.fileno())
        except OSError:
            pass
        journal.close()

    logger.info(
        "link complete: %s relinked, %s resumed-skip, %s awaiting prepare. entities: %s",
        relinked_trials,
        resumed_skip,
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
