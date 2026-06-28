"""Idempotent end-to-end orchestration for TrialMatchAI.

Chains the three pipeline stages — ingest patient inputs, build the search
index, run matching — and skips work that is already done:

  * ingest: a patient is skipped if its canonical profile already exists.
  * index:  a stage is skipped if the search tables already exist.
  * match:  a patient is skipped if it already has a non-empty ranked_trials.json.

Both the general ``trialmatchai e2e`` command and the TREC preset are thin
wrappers over these stages, so idempotency behaves identically everywhere.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Sequence

from trialmatchai.search import LanceDBSearchBackend
from trialmatchai.utils.file_utils import is_valid_json_file, write_json_file
from trialmatchai.utils.logging_config import setup_logging

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
            # Summary first; the profile JSON is written last (atomically) as the
            # completion marker, so a crash between them re-imports rather than
            # leaving a profile with no summary.
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
        merged = enrich_summary(summary, expansion)
        merged["query_expanded"] = True
        write_json_file(merged, str(summary_path))
        enriched += 1
        logger.info("Query-expanded %s", pid)

    # Release the expander wrapper; its CoT engine stays in the vLLM engine cache
    # and is reused by the match stage (shared — a single load, not two).
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


def build_index(
    config: Dict[str, Any],
    *,
    processed_trials_folder: str | Path = "data/processed_trials",
    processed_criteria_folder: str | Path = "data/processed_criteria",
    nct_filter: Iterable[str] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Build the LanceDB search tables, optionally restricted to ``nct_filter``.

    Skips when both tables already exist unless ``force``. The backend (and thus
    the target db path) comes from ``config['search_backend']``.
    """
    backend = LanceDBSearchBackend.from_config(config)
    search_cfg = config.get("search_backend", {})
    trials_table = search_cfg.get("trials_table", "trials")
    criteria_table = search_cfg.get("criteria_table", "criteria")

    if (
        not force
        and backend.table_exists(trials_table)
        and backend.table_exists(criteria_table)
    ):
        logger.info("Index stage skipped: tables already present at %s", backend.db_path)
        return {"skipped": True, "db_path": str(backend.db_path)}

    nct_set = set(nct_filter) if nct_filter is not None else None
    trial_docs = list(_iter_trial_docs(Path(processed_trials_folder), nct_set))
    if not trial_docs:
        raise RuntimeError(
            f"No prepared trial documents found in {processed_trials_folder}"
            + (f" for {len(nct_set)} filtered NCT ids" if nct_set else "")
        )
    n_trials = backend.index_trials(trial_docs, recreate=True)
    logger.info("Indexed %s trial documents.", n_trials)

    criteria_docs = list(_iter_criteria_docs(Path(processed_criteria_folder), nct_set))
    if not criteria_docs:
        # Refuse to leave an inconsistent index (trials table but no criteria
        # table) where `ready_to_match` can never become true. An empty corpus is
        # a data error — the corpus is unprepared.
        raise RuntimeError(
            f"No criteria documents found in {processed_criteria_folder}"
            + (f" for the {len(nct_set)} filtered NCT ids" if nct_set else "")
            + ". The corpus appears unprepared — run `trialmatchai build` (prepare) first."
        )
    n_criteria = backend.index_criteria(criteria_docs, recreate=True)
    logger.info("Indexed %s criteria documents.", n_criteria)

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


def run_matching(
    config: Dict[str, Any],
    *,
    resume: bool = True,
    force: bool = False,
) -> int:
    """Run the matching pipeline with per-patient resume.

    When resuming, the expensive model stack is not even loaded if every patient
    is already done.
    """
    use_resume = resume and not force
    if use_resume:
        pending, done = count_pending(config)
        if pending == 0:
            logger.info("Match stage skipped: all %s patient(s) already matched.", done)
            return 0
        logger.info("Match stage: %s pending, %s already done.", pending, done)
    # Imported lazily so the convert/index stages (and the CPU-only --index-only
    # path) do not pull in the heavy model stack that main.py imports.
    from trialmatchai.main import main_pipeline

    return main_pipeline(config=config, resume=use_resume)


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
    """Ingest -> index -> match, each stage idempotent. Returns a process code."""
    if inputs:
        ingest_inputs(
            config,
            inputs,
            input_format=input_format,
            with_entities=with_entities,
            force=force_reingest,
        )
    expand_queries(config, force=force_rematch)
    build_index(
        config,
        processed_trials_folder=processed_trials_folder,
        processed_criteria_folder=processed_criteria_folder,
        nct_filter=nct_filter,
        force=force_reindex,
    )
    try:
        return run_matching(config, resume=True, force=force_rematch)
    finally:
        free_models()


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


def _save_manifest(path: Path, manifest: dict) -> None:
    write_json_file(manifest, str(path))


def _count_json(folder: Path) -> int:
    return len(list(folder.glob("*.json"))) if folder.exists() else 0


def _count_subdirs(folder: Path) -> int:
    return sum(1 for p in folder.iterdir() if p.is_dir()) if folder.exists() else 0


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
        if force or not (processed_trials_folder / f"{p.stem}.json").exists()
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
            # Write criteria first; the trial JSON is written last and is the
            # per-trial completion marker the resume check keys on — so an
            # interrupted trial (criteria written, trial not) is re-processed
            # rather than wrongly skipped.
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
    backend = LanceDBSearchBackend.from_config(config)
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
) -> dict:
    """Run the setup half (prepare -> index), idempotent, with a manifest.

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
    if have_source:
        # prepare_corpus internally skips already-prepared trials, so calling it
        # whenever source exists safely resumes without redoing finished work.
        stats = prepare_corpus(
            config,
            trials_json_folder=trials_json_folder,
            processed_trials_folder=pt,
            processed_criteria_folder=pc,
            force=force_prepare,
        )
        manifest["prepare"] = {**stats, "completed_at": _now_iso()}
    elif have_prepared:
        logger.info("Prepare skipped: %s already populated (no trials_jsons source).", pt)
        manifest["prepare"] = {"skipped_existing": True, "completed_at": _now_iso()}
    else:
        raise RuntimeError(
            f"Nothing to prepare: {pt} is empty and no trial JSONs at "
            f"{trials_json_folder}. Run `trialmatchai-bootstrap-data` or provide "
            "normalized trial JSONs."
        )
    _save_manifest(manifest_path, manifest)

    # Stage 2 — build the LanceDB search index (idempotent).
    logger.info("=== build: index stage ===")
    index_info = build_index(
        config,
        processed_trials_folder=pt,
        processed_criteria_folder=pc,
        force=force_reindex,
    )
    manifest["index"] = {**index_info, "completed_at": _now_iso()}
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
