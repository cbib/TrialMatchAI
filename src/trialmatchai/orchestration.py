"""Idempotent end-to-end orchestration for TrialMatchAI.

Chains the three pipeline stages (ingest patient inputs, build the search index,
run matching) and skips work already done: a patient is skipped once its profile
exists / once it has a non-empty ranked_trials.json, and the index is skipped once
the search tables exist. The ``e2e`` command and the TREC preset are thin wrappers
over these stages, so idempotency behaves identically everywhere.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Sequence

from trialmatchai.search import build_search_backend
from trialmatchai.utils.file_utils import is_valid_json_file, write_json_file
from trialmatchai.utils.logging_config import setup_logging
from trialmatchai.utils.pipeline_state import (
    atomic_write_json,
    digest,
    dir_fingerprint,
    stage_is_current,
)

logger = setup_logging(__name__)


# --------------------------------------------------------------------------- #
# Ingest stage
# --------------------------------------------------------------------------- #
def ingest_inputs(
    config: Dict[str, Any],
    inputs: Sequence[str | Path],
    *,
    input_format: str = "auto",
    with_entities: bool = True,
    force: bool = False,
) -> int:
    """Import patient inputs (any supported format) into canonical profiles.

    Skips a patient whose profile already exists unless ``force``. Returns the
    number of profiles available afterwards.
    """
    from trialmatchai.interop.exporters import profile_to_matching_summary
    from trialmatchai.interop.importers import import_patient_path

    patient_cfg = config.get("patient_inputs", {})
    profile_dir = Path(patient_cfg.get("profile_dir", "data/patients/profiles"))
    summary_dir = Path(patient_cfg.get("summary_dir", "data/patients/summaries"))
    profile_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    entity_annotator = _maybe_entity_annotator(config) if with_entities else None
    strict = bool(patient_cfg.get("strict_validation", False))

    imported = 0
    for raw in inputs:
        profiles = import_patient_path(
            raw,
            input_format=input_format,
            entity_annotator=entity_annotator,
            strict=strict,
        )
        for profile in profiles:
            profile_path = profile_dir / f"{profile.patient_id}.json"
            if not force and is_valid_json_file(str(profile_path)):
                logger.info("Ingest skipped (exists): %s", profile.patient_id)
                continue
            # Summary first, profile JSON last as the completion marker: a crash between them re-imports rather than orphaning a profile.
            write_json_file(
                profile_to_matching_summary(profile),
                str(summary_dir / f"{profile.patient_id}.json"),
            )
            write_json_file(
                profile.model_dump(mode="json", exclude_none=True),
                str(profile_path),
            )
            imported += 1
            logger.info("Ingested patient %s", profile.patient_id)

    total = len(list(profile_dir.glob("*.json")))
    logger.info("Ingest stage: %s new, %s profiles total", imported, total)
    return total


def _maybe_entity_annotator(config: Dict[str, Any]):
    try:
        from trialmatchai.entities import build_entity_annotator
        from trialmatchai.models.embedding import build_embedder

        embedder = build_embedder(config)
        return build_entity_annotator(config, embedder=embedder)
    except Exception as exc:  # pragma: no cover - optional model stack
        logger.warning("Entity annotation unavailable; ingesting without it: %s", exc)
        return None


# --------------------------------------------------------------------------- #
# Query-expansion stage (runtime CoT, faithful to legacy keywords.json)
# --------------------------------------------------------------------------- #
def expand_queries(config: Dict[str, Any], *, force: bool = False) -> int:
    """Enrich each patient's matching summary via the CoT query expander.

    No-op unless ``query_expansion.enabled``. Loads the model once, enriches
    every summary, then frees it before the match stage loads its own model.
    Idempotent: a summary already marked ``query_expanded`` is skipped.
    """
    from trialmatchai.matching.query_expansion import build_query_expander, enrich_summary

    expander = build_query_expander(config)
    if expander is None:
        logger.info("Query expansion disabled; using deterministic summaries.")
        return 0

    patient_cfg = config.get("patient_inputs", {})
    profile_dir = Path(patient_cfg.get("profile_dir", "data/patients/profiles"))
    summary_dir = Path(patient_cfg.get("summary_dir", "data/patients/summaries"))
    enriched = 0
    for profile_path in sorted(profile_dir.glob("*.json")):
        pid = profile_path.stem
        summary_path = summary_dir / f"{pid}.json"
        if not summary_path.exists():
            continue
        summary = _read_json(summary_path)
        if not force and summary.get("query_expanded"):
            continue
        profile = _read_json(profile_path)
        narrative = [n.get("text", "") for n in profile.get("notes", []) if n.get("text")]
        if not narrative:
            narrative = list(summary.get("patient_narrative", []))
        expansion = expander.expand(narrative)
        merged = enrich_summary(
            summary,
            expansion,
            max_main_conditions=int(expander.settings.get("max_main_conditions", 11)),
            max_other_conditions=int(expander.settings.get("max_other_conditions", 50)),
        )
        merged["query_expanded"] = True
        write_json_file(merged, str(summary_path))
        enriched += 1
        logger.info("Query-expanded %s", pid)

    # Release the wrapper; its CoT engine stays cached and is reused by the match stage (a single load, not two).
    del expander
    logger.info("Query-expansion stage: enriched %s summaries.", enriched)
    return enriched


# --------------------------------------------------------------------------- #
# Index stage
# --------------------------------------------------------------------------- #
def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_trial_docs(folder: Path, nct_filter: set[str] | None) -> Iterator[dict]:
    if not folder.exists():
        return
    for path in sorted(folder.glob("*.json")):
        if nct_filter is not None and path.stem not in nct_filter:
            continue
        yield _read_json(path)


def _iter_criteria_docs(folder: Path, nct_filter: set[str] | None) -> Iterator[dict]:
    if not folder.exists():
        return
    for subdir in sorted(p for p in folder.iterdir() if p.is_dir()):
        if nct_filter is not None and subdir.name not in nct_filter:
            continue
        for path in sorted(subdir.glob("*.json")):
            yield _read_json(path)


# Trial text fields whose vectors the backend scores on (mirrors registry.preparation).
_TRIAL_VECTOR_FIELDS = (
    ("brief_title", "brief_title_vector"),
    ("brief_summary", "brief_summary_vector"),
    ("condition", "condition_vector"),
    ("eligibility_criteria", "eligibility_criteria_vector"),
)


def _reembed_docs_inplace(
    config: Dict[str, Any],
    trial_docs: list[dict],
    criteria_docs: list[dict],
) -> None:
    """Replace the corpus's pre-computed vectors with the config embedder's (document side).

    Without this, build_index reuses the prepare-time vectors, so an embedder swap never reaches
    retrieval — the query dim-mismatches the index and silently falls back to BM25. Empty text
    keeps an empty vector, matching registry.preparation._embed_texts.
    """
    from trialmatchai.models.embedding import build_embedder

    embedder = build_embedder(config)
    embed_documents = getattr(embedder, "embed_documents", embedder.embed_texts)

    texts: list[str] = []
    slots: list[tuple[dict, str]] = []
    for doc in trial_docs:
        for text_field, vector_field in _TRIAL_VECTOR_FIELDS:
            text = str(doc.get(text_field) or "")
            if text.strip():
                texts.append(text)
                slots.append((doc, vector_field))
            else:
                doc[vector_field] = []
    for doc in criteria_docs:
        text = str(doc.get("criterion") or "")
        if text.strip():
            texts.append(text)
            slots.append((doc, "criterion_vector"))
        else:
            doc["criterion_vector"] = []

    logger.info(
        "Re-embedding %s document texts with the config embedder (%s trials, %s criteria)...",
        len(texts),
        len(trial_docs),
        len(criteria_docs),
    )
    # Chunk so progress is visible and peak memory is bounded.
    chunk = 8192
    done = 0
    for start in range(0, len(texts), chunk):
        batch = texts[start : start + chunk]
        vectors = embed_documents(batch)
        for (doc, vector_field), vector in zip(slots[start : start + chunk], vectors):
            doc[vector_field] = list(vector)
        done += len(batch)
        logger.info("Re-embedded %s/%s texts", done, len(texts))


_EMBEDDER_SIDECAR = "_embedder.json"


def _embedder_identity(config: Dict[str, Any]) -> dict:
    """Model-load-free identity of the embedder's vector space, to detect an embedder swap between
    builds. Excludes ``dim`` (needs a model load); a ``model_name`` change is a sufficient proxy."""
    from trialmatchai.models.embedding.text_embedder import native_metric_from_config

    ec = config.get("embedder", {}) or {}
    return {
        "backend": ec.get("type") or ec.get("backend") or "hf",
        "model_name": ec.get("model_name"),
        "query_model_name": ec.get("query_model_name"),
        "normalize": bool(ec.get("normalize", True)),
        "native_metric": native_metric_from_config(config),
    }


def _read_embedder_sidecar(db_path: str | Path) -> dict | None:
    """Return the embedder provenance recorded next to the index, or None when absent/unreadable.
    Absent (e.g. a pre-provenance index like data/search_medcpt_*) means 'trust stored vectors'."""
    path = Path(db_path) / _EMBEDDER_SIDECAR
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Could not read embedder provenance %s (ignoring).", path)
        return None


def _write_embedder_sidecar(db_path: str | Path, identity: dict, *, reembedded: bool) -> None:
    """Record which embedder the index vectors correspond to, so a later build detects a swap."""
    path = Path(db_path)
    try:
        path.mkdir(parents=True, exist_ok=True)
        (path / _EMBEDDER_SIDECAR).write_text(
            json.dumps({"identity": identity, "reembedded": reembedded, "tmai_schema": 1}, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning("Could not write embedder provenance to %s: %s", path, exc)


def build_index(
    config: Dict[str, Any],
    *,
    processed_trials_folder: str | Path = "data/processed_trials",
    processed_criteria_folder: str | Path = "data/processed_criteria",
    nct_filter: Iterable[str] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Build the LanceDB search tables, optionally restricted to ``nct_filter``.

    Skips when both tables already exist AND the recorded embedder still matches the configured
    one (unless ``force``). If the embedder changed, the index is rebuilt and re-embedded
    automatically so retrieval matches the new embedder instead of silently falling back to BM25.
    """
    backend = build_search_backend(config)
    search_cfg = config.get("search_backend", {})
    trials_table = search_cfg.get("trials_table", "trials")
    criteria_table = search_cfg.get("criteria_table", "criteria")
    identity = _embedder_identity(config)

    auto_reembed = False
    if (
        not force
        and backend.table_exists(trials_table)
        and backend.table_exists(criteria_table)
    ):
        stored = _read_embedder_sidecar(backend.db_path)
        if stored is not None and stored.get("identity") != identity:
            logger.warning(
                "Index at %s was built with embedder %s but config requests %s — rebuilding and "
                "re-embedding so retrieval matches (this was a silent BM25 fallback before).",
                backend.db_path, stored.get("identity"), identity,
            )
            auto_reembed = True  # fall through to rebuild + re-embed
        else:
            provenance = "embedder matches" if stored else "no provenance recorded; trusting stored vectors"
            logger.info("Index stage skipped: tables present at %s (%s).", backend.db_path, provenance)
            return {"skipped": True, "db_path": str(backend.db_path)}

    nct_set = set(nct_filter) if nct_filter is not None else None
    trial_docs = list(_iter_trial_docs(Path(processed_trials_folder), nct_set))
    if not trial_docs:
        raise RuntimeError(
            f"No prepared trial documents found in {processed_trials_folder}"
            + (f" for {len(nct_set)} filtered NCT ids" if nct_set else "")
        )
    # Validate BOTH corpora before writing any table, else an empty-criteria corpus leaves an
    # inconsistent index (trials but no criteria) where `ready_to_match` never becomes true.
    criteria_docs = list(_iter_criteria_docs(Path(processed_criteria_folder), nct_set))
    if not criteria_docs:
        raise RuntimeError(
            f"No criteria documents found in {processed_criteria_folder}"
            + (f" for the {len(nct_set)} filtered NCT ids" if nct_set else "")
            + ". The corpus appears unprepared — run `trialmatchai build` (prepare) first."
        )

    # Re-embed when requested or when an embedder swap was detected (auto_reembed), so retrieval
    # reaches the new embedder instead of the prepare-time vectors (see _reembed_docs_inplace).
    do_reembed = bool(search_cfg.get("reembed_index", False) or auto_reembed)
    if do_reembed:
        _reembed_docs_inplace(config, trial_docs, criteria_docs)

    n_trials = backend.index_trials(trial_docs, recreate=True)
    logger.info("Indexed %s trial documents.", n_trials)

    n_criteria = backend.index_criteria(criteria_docs, recreate=True)
    logger.info("Indexed %s criteria documents.", n_criteria)

    # Record which embedder these vectors correspond to, so a later build auto-detects a swap.
    _write_embedder_sidecar(backend.db_path, identity, reembedded=do_reembed)

    return {
        "skipped": False,
        "db_path": str(backend.db_path),
        "trials": n_trials,
        "criteria": n_criteria,
    }


# --------------------------------------------------------------------------- #
# Match stage
# --------------------------------------------------------------------------- #
def count_pending(config: Dict[str, Any]) -> tuple[int, int]:
    """Return (pending, done) patient counts for the configured dirs."""
    patient_cfg = config.get("patient_inputs", {})
    profile_dir = Path(patient_cfg.get("profile_dir", "data/patients/profiles"))
    output_dir = Path(config["paths"]["output_dir"])
    pending = done = 0
    for profile_path in sorted(profile_dir.glob("*.json")):
        ranked = output_dir / profile_path.stem / "ranked_trials.json"
        if is_valid_json_file(str(ranked)):
            done += 1
        else:
            pending += 1
    return pending, done


_MATCH_STATE_VERSION = "1"


def _match_signature(config: Dict[str, Any]) -> dict:
    """Match-relevant config whose change should invalidate cached patient matches."""
    reranker = config.get("LLM_reranker", {})
    model = config.get("model", {})
    return {
        "reranker_backend": reranker.get("backend"),
        "reranker_enabled": reranker.get("enabled"),
        # Model identity: swapping reranker/CoT weights or adapter must re-rank even on an unchanged corpus.
        "reranker_model": model.get("reranker_model_path"),
        "reranker_adapter": model.get("reranker_adapter_path"),
        "reranker_revision": model.get("reranker_model_revision"),
        "cot_model": model.get("base_model"),
        "use_cot": config.get("use_cot_reasoning"),
        "query_expansion": (config.get("query_expansion") or {}).get("enabled"),
        "candidate_limit": (config.get("search_backend") or {}).get("candidate_limit"),
        "search_mode": (config.get("search") or {}).get("mode"),
    }


def _match_corpus_fingerprint(config: Dict[str, Any]) -> str:
    """Fingerprint of the search index the match queries + match config. Empty on error."""
    try:
        db_path = (config.get("search_backend") or {}).get("db_path", "data/search")
        return digest(
            _MATCH_STATE_VERSION,
            dir_fingerprint(db_path, include_dirs=True),
            _match_signature(config),
        )
    except Exception:
        return ""


def _match_state_path(config: Dict[str, Any]) -> Path | None:
    try:
        return Path(config["paths"]["output_dir"]) / ".match_state.json"
    except Exception:
        return None


def _match_corpus_changed(config: Dict[str, Any], corpus_fp: str) -> bool:
    """True only when a prior match recorded a different corpus fingerprint (stale results);
    absent record or empty fingerprint -> False, leaving the per-patient resume untouched.
    """
    path = _match_state_path(config)
    if path is None or not corpus_fp:
        return False
    try:
        recorded = json.loads(path.read_text()).get("corpus_fingerprint")
    except Exception:
        return False
    return recorded is not None and recorded != corpus_fp


def _record_match_corpus(config: Dict[str, Any], corpus_fp: str) -> None:
    path = _match_state_path(config)
    if path is None or not corpus_fp:
        return
    try:
        atomic_write_json(path, {"corpus_fingerprint": corpus_fp, "completed_at": _now_iso()})
    except Exception as exc:  # pragma: no cover - best-effort bookkeeping
        logger.debug("match state write skipped: %s", exc)


def run_matching(
    config: Dict[str, Any],
    *,
    resume: bool = True,
    force: bool = False,
) -> int:
    """Run the matching pipeline with per-patient resume.

    Resume skips loading the model stack when every patient is done, and is invalidated when
    the search index changed since the matches were produced, so stale results aren't served.
    """
    use_resume = resume and not force
    corpus_fp = _match_corpus_fingerprint(config)
    if use_resume and _match_corpus_changed(config, corpus_fp):
        logger.info(
            "Match resume invalidated: search index changed since last match; re-matching."
        )
        use_resume = False
    if use_resume:
        pending, done = count_pending(config)
        if pending == 0:
            logger.info("Match stage skipped: all %s patient(s) already matched.", done)
            return 0
        logger.info("Match stage: %s pending, %s already done.", pending, done)
    # Lazy import so the index-only / CPU-only path doesn't pull in main.py's heavy model stack.
    from trialmatchai.main import main_pipeline

    result = main_pipeline(config=config, resume=use_resume)
    _record_match_corpus(config, corpus_fp)
    return result


# --------------------------------------------------------------------------- #
# Full e2e
# --------------------------------------------------------------------------- #
def free_models() -> None:
    """Release cached GPU model engines. Call at the end of a top-level run."""
    try:
        from trialmatchai.models.llm.vllm_loader import free_vllm_engines

        free_vllm_engines()
    except Exception as exc:  # pragma: no cover - teardown is best-effort
        logger.debug("free_models: %s", exc)


def run_e2e(
    config: Dict[str, Any],
    inputs: Sequence[str | Path],
    *,
    input_format: str = "auto",
    with_entities: bool = True,
    processed_trials_folder: str | Path = "data/processed_trials",
    processed_criteria_folder: str | Path = "data/processed_criteria",
    nct_filter: Iterable[str] | None = None,
    force_reingest: bool = False,
    force_reindex: bool = False,
    force_rematch: bool = False,
) -> int:
    """Run-half of the unified pipeline (slice index -> ingest -> expand -> match).

    A thin preset over ``run_pipeline`` (index..match): index/ingest/expand/match
    are idempotent so this resumes from any state, and GPU models are freed once.
    """
    from trialmatchai.pipeline import StageContext, run_pipeline

    force: set[str] = set()
    if force_reingest:
        force.add("ingest")
    if force_reindex:
        force.add("index")
    if force_rematch:
        force.update({"expand", "match"})
    ctx = StageContext(
        config=config,
        processed_trials_folder=Path(processed_trials_folder),
        processed_criteria_folder=Path(processed_criteria_folder),
        inputs=list(inputs),
        input_format=input_format,
        with_entities=with_entities,
        nct_filter=set(nct_filter) if nct_filter is not None else None,
        force=force,
    )
    return run_pipeline(ctx, from_stage="index", to_stage="match")


# --------------------------------------------------------------------------- #
# Build (setup half): prepare -> index, resumable, with a build manifest
# --------------------------------------------------------------------------- #
def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _manifest_path(processed_trials_folder: str | Path) -> Path:
    return Path(processed_trials_folder).parent / ".trialmatchai_build.json"


def _load_manifest(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


# Bump a stage's version when a logic change must invalidate a cached completion.
_PREPARE_STATE_VERSION = "1"
_LINK_STATE_VERSION = "1"
_INDEX_STATE_VERSION = "1"


def _prepare_signature(config: dict) -> dict:
    """Config whose change must invalidate the prepare stage (embeddings + entities)."""
    embedder = config.get("embedder", {})
    entity = config.get("entity_extraction", {})
    return {
        "embedder_model": embedder.get("model_name"),
        "embedder_revision": embedder.get("revision"),
        "entity_backend": entity.get("backend"),
        "entity_model": entity.get("model_name"),
        "entity_threshold": entity.get("threshold"),
    }


def _linker_signature(config: dict) -> dict:
    """Config whose change must invalidate the link stage."""
    linker = config.get("concept_linker", {})
    return {
        key: linker.get(key)
        for key in (
            "enabled", "db_path", "table", "accept_threshold",
            "reject_threshold", "margin", "rerank", "search_limit",
        )
    }


def _index_signature(config: dict) -> dict:
    """Config whose change must invalidate the search index."""
    search = config.get("search_backend", {})
    return {
        key: search.get(key)
        for key in ("backend", "db_path", "trials_table", "criteria_table", "candidate_limit")
    }


def _index_tables_present(config: dict) -> bool:
    db_path = Path((config.get("search_backend") or {}).get("db_path", "data/search"))
    return db_path.exists() and any(db_path.iterdir())


def _save_manifest(path: Path, manifest: dict) -> None:
    atomic_write_json(path, manifest)


def _count_json(folder: Path) -> int:
    return len(list(folder.glob("*.json"))) if folder.exists() else 0


def _count_subdirs(folder: Path) -> int:
    return sum(1 for p in folder.iterdir() if p.is_dir()) if folder.exists() else 0


def _trial_needs_prepare(source: Path, processed_trials_folder: Path) -> bool:
    """True if a source trial has no valid prepared output yet, or was edited since (source
    mtime newer than the output's, make-style).

    Existence alone would permanently skip an in-place edit reusing the same NCT id, baking a
    stale embedding/entities into the index.
    """
    output = processed_trials_folder / f"{source.stem}.json"
    if not is_valid_json_file(str(output)):
        return True
    try:
        return source.stat().st_mtime_ns > output.stat().st_mtime_ns
    except OSError:
        return True


def prepare_corpus(
    config: Dict[str, Any],
    *,
    trials_json_folder: str | Path,
    processed_trials_folder: str | Path,
    processed_criteria_folder: str | Path,
    force: bool = False,
    log_every: int = 500,
) -> dict[str, int]:
    """Embed + annotate normalized trial JSONs into processed_*; resumable.

    Streams one trial at a time (bounded memory), skips trials already prepared
    so an interrupted build picks up where it left off, and isolates per-trial
    failures so one bad document cannot abort the whole corpus.
    """
    trials_json_folder = Path(trials_json_folder)
    processed_trials_folder = Path(processed_trials_folder)
    processed_criteria_folder = Path(processed_criteria_folder)

    all_paths = sorted(trials_json_folder.glob("*.json"))
    if not all_paths:
        raise RuntimeError(f"No trial JSON files found to prepare in {trials_json_folder}")

    pending = [
        p
        for p in all_paths
        if force or _trial_needs_prepare(p, processed_trials_folder)
    ]
    skipped = len(all_paths) - len(pending)
    logger.info(
        "Prepare: %s trials total, %s already prepared, %s to process.",
        len(all_paths),
        skipped,
        len(pending),
    )
    if not pending:
        return {"total": len(all_paths), "prepared": 0, "skipped": skipped, "failed": 0}

    from trialmatchai.entities import build_entity_annotator
    from trialmatchai.models.embedding import build_embedder
    from trialmatchai.registry.preparation import (
        prepare_criteria_documents,
        prepare_trial_document,
        write_prepared_criteria,
        write_prepared_trial,
    )

    processed_trials_folder.mkdir(parents=True, exist_ok=True)
    processed_criteria_folder.mkdir(parents=True, exist_ok=True)
    embedder = build_embedder(config)
    entity_annotator = build_entity_annotator(config, embedder=embedder)

    prepared = failed = 0
    for i, path in enumerate(pending, start=1):
        try:
            doc = _read_json(path)
            trial_row = prepare_trial_document(doc, embedder)
            criteria_rows = prepare_criteria_documents(
                doc, embedder, entity_annotator=entity_annotator
            )
            # Criteria first, trial JSON last as the resume completion marker: an interrupted trial is re-processed, not wrongly skipped.
            write_prepared_criteria(criteria_rows, processed_criteria_folder)
            write_prepared_trial(trial_row, processed_trials_folder)
            prepared += 1
        except Exception:
            failed += 1
            logger.exception("Prepare failed for %s (continuing)", path.name)
        if i % log_every == 0:
            logger.info("Prepare progress: %s/%s done, %s failed.", i, len(pending), failed)

    logger.info(
        "Prepare complete: %s prepared, %s skipped, %s failed.", prepared, skipped, failed
    )
    return {"total": len(all_paths), "prepared": prepared, "skipped": skipped, "failed": failed}


def build_state(
    config: Dict[str, Any],
    *,
    processed_trials_folder: str | Path = "data/processed_trials",
    processed_criteria_folder: str | Path = "data/processed_criteria",
) -> dict:
    """Report what the build half has produced — used by `build --status`."""
    pt = Path(processed_trials_folder)
    pc = Path(processed_criteria_folder)
    search_cfg = config.get("search_backend", {})
    backend = build_search_backend(config)
    trials_table = backend.table_exists(search_cfg.get("trials_table", "trials"))
    criteria_table = backend.table_exists(search_cfg.get("criteria_table", "criteria"))
    linker = config.get("concept_linker", {})
    concepts_path = Path(linker.get("db_path", "data/concepts"))
    concepts_present = concepts_path.exists() and any(concepts_path.iterdir())
    return {
        "processed_trials": {"folder": str(pt), "count": _count_json(pt)},
        "processed_criteria": {"folder": str(pc), "count": _count_subdirs(pc)},
        "index": {
            "db_path": str(backend.db_path),
            "trials_table": trials_table,
            "criteria_table": criteria_table,
        },
        "concepts": {"db_path": str(concepts_path), "present": bool(concepts_present)},
        "ready_to_match": bool(trials_table and criteria_table),
    }


def build_system(
    config: Dict[str, Any],
    *,
    trials_json_folder: str | Path | None = None,
    processed_trials_folder: str | Path = "data/processed_trials",
    processed_criteria_folder: str | Path = "data/processed_criteria",
    force_prepare: bool = False,
    force_reindex: bool = False,
    link_concepts: bool = False,
) -> dict:
    """Run the setup half (prepare -> link -> index), idempotent, with a manifest.

    Each stage is resumable and recorded in ``.trialmatchai_build.json`` next to
    the processed data, so a disrupted build can be re-run and continues from the
    last completed work.
    """
    paths = config.get("paths", {})
    trials_json_folder = Path(trials_json_folder or paths.get("trials_json_folder", "data/trials_jsons"))
    pt = Path(processed_trials_folder)
    pc = Path(processed_criteria_folder)
    manifest_path = _manifest_path(pt)
    manifest = _load_manifest(manifest_path)
    manifest["started_at"] = _now_iso()

    # Stage 1 — prepare embeddings/entities (resumable, GPU).
    have_prepared = _count_json(pt) > 0
    have_source = trials_json_folder.exists() and any(trials_json_folder.glob("*.json"))
    logger.info("=== build: prepare stage ===")
    # Skip the whole stage when source corpus + config + code version are unchanged -- no per-trial rescan/model load.
    prepare_fp = digest(
        _PREPARE_STATE_VERSION, dir_fingerprint(trials_json_folder), _prepare_signature(config)
    )
    if not force_prepare and stage_is_current(
        manifest.get("prepare"), fingerprint=prepare_fp, output_present=have_prepared
    ):
        logger.info(
            "Prepare stage skipped: source corpus + config + code unchanged (fingerprint match)."
        )
    elif have_source:
        # prepare_corpus internally skips already-prepared trials, so this safely resumes.
        stats = prepare_corpus(
            config,
            trials_json_folder=trials_json_folder,
            processed_trials_folder=pt,
            processed_criteria_folder=pc,
            force=force_prepare,
        )
        prepare_ok = int(stats.get("failed", 0) or 0) == 0
        manifest["prepare"] = {
            **stats,
            "status": "complete" if prepare_ok else "incomplete",
            "output_fingerprint": dir_fingerprint(pt),
            "completed_at": _now_iso(),
        }
        # Only stamp the skip fingerprint when every trial prepared, so failures force the next build to retry rather than skip into permanent gaps.
        if prepare_ok:
            manifest["prepare"]["fingerprint"] = prepare_fp
    elif have_prepared:
        logger.info("Prepare skipped: %s already populated (no trials_jsons source).", pt)
        manifest["prepare"] = {
            "skipped_existing": True,
            "status": "complete",
            "fingerprint": prepare_fp,
            "output_fingerprint": dir_fingerprint(pt),
            "completed_at": _now_iso(),
        }
    else:
        raise RuntimeError(
            f"Nothing to prepare: {pt} is empty and no trial JSONs at "
            f"{trials_json_folder}. Run `trialmatchai bootstrap-data` or provide "
            "normalized trial JSONs."
        )
    _save_manifest(manifest_path, manifest)

    # Stage 1b — link extracted entities to concept IDs so the index carries them instead of
    # concept_store_unavailable. Idempotent: relinks an already-prepared NER-only corpus.
    if link_concepts:
        logger.info("=== build: link stage ===")
        # Chain off prepare's output fingerprint: skip the whole stage when prepared corpus +
        # linker config + code are unchanged, instead of re-reading every criterion.
        upstream_fp = (manifest.get("prepare") or {}).get("output_fingerprint", "")
        link_fp = digest(_LINK_STATE_VERSION, upstream_fp, _linker_signature(config))
        if not force_prepare and stage_is_current(
            manifest.get("link"), fingerprint=link_fp, output_present=_count_subdirs(pc) > 0
        ):
            logger.info(
                "Link stage skipped: prepared corpus + linker config + code unchanged "
                "(fingerprint match) -- not re-reading the criteria corpus."
            )
        else:
            from trialmatchai.linking import link_corpus

            link_tally = link_corpus(
                config, processed_criteria_folder=pc, processed_trials_folder=pt
            )
            manifest["link"] = {
                **link_tally,
                "status": "complete",
                "fingerprint": link_fp,
                "completed_at": _now_iso(),
            }
        _save_manifest(manifest_path, manifest)

    # Stage 2 — build the LanceDB search index.
    logger.info("=== build: index stage ===")
    # Chain off the upstream this build reflects: freshly-computed link_fp when link ran, else
    # prepare's current output fingerprint. The stored manifest link fingerprint would be stale
    # on a later link_concepts=False rebuild -- the index would skip its rebuild.
    if link_concepts:
        index_upstream = link_fp
    else:
        index_upstream = (manifest.get("prepare") or {}).get("output_fingerprint", "")
    index_fp = digest(_INDEX_STATE_VERSION, index_upstream, _index_signature(config))
    if not force_reindex and stage_is_current(
        manifest.get("index"), fingerprint=index_fp, output_present=_index_tables_present(config)
    ):
        logger.info(
            "Index stage skipped: corpus + index config + code unchanged (fingerprint match)."
        )
        index_info = dict(manifest["index"])
    else:
        # Fingerprint changed (or forced): rebuild so the index reflects the current corpus.
        index_info = build_index(
            config, processed_trials_folder=pt, processed_criteria_folder=pc, force=True
        )
        manifest["index"] = {
            **index_info,
            "status": "complete",
            "fingerprint": index_fp,
            "completed_at": _now_iso(),
        }
    _save_manifest(manifest_path, manifest)

    state = build_state(
        config,
        processed_trials_folder=pt,
        processed_criteria_folder=pc,
    )
    manifest["state"] = state
    manifest["completed_at"] = _now_iso()
    _save_manifest(manifest_path, manifest)
    logger.info(
        "Build complete — ready_to_match=%s. Manifest: %s",
        state["ready_to_match"],
        manifest_path,
    )
    return manifest
