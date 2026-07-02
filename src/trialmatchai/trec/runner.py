"""End-to-end TREC runner — a preset over the core orchestration stages.

For each requested track it converts the TREC patient topics, builds a search
index restricted to the track's NCT collection, and runs matching with
per-patient resume. Every step is idempotent (skips already-done work).
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

from trialmatchai.config.config_loader import load_config
from trialmatchai.orchestration import (
    build_index,
    count_pending,
    expand_queries,
    free_models,
    run_matching,
)
from trialmatchai.trec import qrels as qrels_mod
from trialmatchai.trec.corpus import TrackSpec, resolve_tracks
from trialmatchai.trec.topics import import_topics
from trialmatchai.utils.file_utils import write_json_file
from trialmatchai.utils.logging_config import setup_logging

logger = setup_logging(__name__)


def _track_config(base_config: Dict[str, Any], spec: TrackSpec) -> Dict[str, Any]:
    """Clone the base config with this track's paths swapped in.

    Faithful to the legacy pipeline, the TREC preset enables runtime CoT query
    expansion (which produced the original keywords.json from raw topic text).
    """
    cfg = deepcopy(base_config)
    cfg.setdefault("search_backend", {})["db_path"] = str(spec.db_path)
    cfg.setdefault("patient_inputs", {})["profile_dir"] = str(spec.profile_dir)
    cfg["patient_inputs"]["summary_dir"] = str(spec.summary_dir)
    cfg.setdefault("paths", {})["output_dir"] = str(spec.output_dir)
    cfg.setdefault("query_expansion", {})["enabled"] = True
    # TREC funnel 1000 -> 500 -> 250 (CoT), no second->CoT thinning (keep_divisor=1);
    # deeper than the interactive defaults because TREC scores the whole ranked list.
    search = cfg.setdefault("search", {})
    search["max_trials_first_level"] = 1000
    search["max_trials_second_level"] = 500
    search["second_level_keep_divisor"] = 1
    cfg.setdefault("rag", {})["max_trials_rag"] = 250
    # No per-topic HTML report: the eval consumes the run files, not the reports.
    cfg.setdefault("reporting", {})["emit_html"] = False
    return cfg


def run_tracks(
    track_keys: list[str],
    *,
    config_path: str | None = None,
    data_dir: str | Path = "data",
    results_root: str | Path = ".",
    processed_trials_folder: str | Path | None = None,
    processed_criteria_folder: str | Path | None = None,
    index_only: bool = False,
    evaluate: bool = True,
    force_reindex: bool = False,
    force_rematch: bool = False,
) -> int:
    """Run the TREC e2e for each track. Returns a process exit code.

    ``processed_trials_folder`` / ``processed_criteria_folder`` default to
    ``<data_dir>/processed_*`` but can point elsewhere (e.g. prepared data on a
    separate mount) while the index and run outputs live under ``data_dir``. The
    per-track corpus is derived from the official qrels.
    """
    base_config = load_config(config_path)
    data_dir = Path(data_dir)
    processed_trials = Path(processed_trials_folder or data_dir / "processed_trials")
    processed_criteria = Path(processed_criteria_folder or data_dir / "processed_criteria")

    specs = resolve_tracks(track_keys, data_dir=data_dir, results_root=Path(results_root))
    failures = 0

    for spec in specs:
        logger.info("================ TREC track %s ================", spec.key)
        cfg = _track_config(base_config, spec)

        # 1) Acquire official topics -> canonical profiles + summaries (idempotent).
        try:
            import_topics(
                spec.key,
                trec_dir=spec.trec_dir,
                profile_dir=spec.profile_dir,
                summary_dir=spec.summary_dir,
            )
        except Exception:
            logger.exception("Topic import failed for track %s", spec.key)
            failures += 1
            continue

        # 2) Official qrels -> per-track corpus pool.
        try:
            qrels_path = qrels_mod.download_qrels(spec.key, spec.trec_dir / "qrels")
            qrels = qrels_mod.parse_qrels(qrels_path, spec.id_prefix)
            nct_filter = qrels_mod.corpus_ncts(qrels)
            logger.info("Track %s corpus pool from qrels: %s trials", spec.key, len(nct_filter))
        except Exception:
            logger.exception("Qrels acquisition failed for track %s", spec.key)
            failures += 1
            continue

        # 3) Runtime CoT query expansion -> enriched keywords.json (idempotent).
        if not index_only:
            expand_queries(cfg, force=force_rematch)
            # Free the query-expansion phi-4 before matching: matching loads its own CoT-LoRA
            # phi-4, and two resident instances (~64 GB) OOM the card, silently degrading
            # eligibility reasoning to constraint-only output.
            free_models()

        # 4) Build the per-track index, restricted to the qrels corpus pool.
        try:
            build_index(
                cfg,
                processed_trials_folder=processed_trials,
                processed_criteria_folder=processed_criteria,
                nct_filter=nct_filter,
                force=force_reindex,
            )
        except Exception:
            logger.exception("Indexing failed for track %s", spec.key)
            failures += 1
            continue

        if index_only:
            pending, done = count_pending(cfg)
            logger.info("Track %s indexed (%s patients pending, %s done).", spec.key, pending, done)
            continue

        # 5) Match with per-patient resume.
        rc = run_matching(cfg, resume=True, force=force_rematch)
        if rc != 0:
            logger.error("Matching returned %s for track %s", rc, spec.key)
            failures += 1
            continue

        # 6) Evaluate recall@k against the same qrels.
        if evaluate:
            try:
                metrics = qrels_mod.evaluate(qrels, spec.output_dir)
                metrics_path = Path(spec.output_dir) / "evaluation_metrics.json"
                write_json_file(metrics, str(metrics_path))
                logger.info(
                    "Track %s metrics: %s -> %s",
                    spec.key,
                    metrics["mean"],
                    metrics_path,
                )
            except Exception:
                logger.exception("Evaluation failed for track %s", spec.key)
        logger.info("Track %s done -> %s", spec.key, spec.output_dir)

    # All tracks done — release the shared GPU engines once.
    free_models()
    if failures:
        logger.warning("TREC run completed with %s track failure(s).", failures)
        return 1
    logger.info("All requested TREC tracks complete: %s", " ".join(track_keys))
    return 0
