"""The single TrialMatchAI pipeline: an ordered registry of idempotent stages.

Every command is a *slice* of this one pipeline. Each stage wraps an
already-idempotent orchestration function (it internally skips work that is done),
so the driver only decides **which** stages to run from the user's selection
(`--only` / `--skip` / `--from` / `--to`) and which to **force** (`--force`).

Because each stage is idempotent, running the whole pipeline from any starting
state "just works": finished stages are cheap no-ops, unfinished ones run. That is
the "one e2e workflow, maximally modular, never redo finished work" contract — a
stage is the unit of modularity, and the e2e run is simply "run every stage".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from trialmatchai.utils.logging_config import setup_logging

logger = setup_logging(__name__)


@dataclass
class StageContext:
    """Everything the stages need, resolved once and threaded through the run."""

    config: dict[str, Any]
    trials_json_folder: Path | None = None
    processed_trials_folder: Path = Path("data/processed_trials")
    processed_criteria_folder: Path = Path("data/processed_criteria")
    inputs: list[str] = field(default_factory=list)
    input_format: str = "auto"
    with_entities: bool = True
    nct_filter: set[str] | None = None
    concepts: str | None = None  # "open" -> build the open concept store
    concept_csv: str | None = None
    synonym_csv: str | None = None
    qrels: dict | None = None  # provided by the TREC preset -> enables eval
    results_dir: Path | None = None
    force: set[str] = field(default_factory=set)

    def forced(self, name: str) -> bool:
        return name in self.force or "all" in self.force


# --------------------------------------------------------------------------- #
# Stage run wrappers — each delegates to an existing idempotent function.
# --------------------------------------------------------------------------- #
def _run_prepare(ctx: StageContext) -> None:
    from trialmatchai.orchestration import prepare_corpus

    trials_json = ctx.trials_json_folder or ctx.config["paths"]["trials_json_folder"]
    prepare_corpus(
        ctx.config,
        trials_json_folder=trials_json,
        processed_trials_folder=ctx.processed_trials_folder,
        processed_criteria_folder=ctx.processed_criteria_folder,
        force=ctx.forced("prepare"),
    )


def _run_concepts(ctx: StageContext) -> None:
    if not ctx.concepts and not ctx.concept_csv:
        logger.info(
            "concepts: not requested (pass concepts='open' and/or a concept_csv); skipping."
        )
        return
    from trialmatchai.cli.build_concepts import run_build_concepts

    run_build_concepts(
        ctx.config,
        sources=ctx.concepts,
        concept_csv=ctx.concept_csv,
        synonym_csv=ctx.synonym_csv,
        force=ctx.forced("concepts"),
    )


def _run_link(ctx: StageContext) -> None:
    from trialmatchai.linking import link_corpus

    link_corpus(
        ctx.config,
        processed_criteria_folder=ctx.processed_criteria_folder,
        processed_trials_folder=ctx.processed_trials_folder,
        force=ctx.forced("link"),
    )


def _run_index(ctx: StageContext) -> None:
    from trialmatchai.orchestration import build_index

    build_index(
        ctx.config,
        processed_trials_folder=ctx.processed_trials_folder,
        processed_criteria_folder=ctx.processed_criteria_folder,
        nct_filter=ctx.nct_filter,
        force=ctx.forced("index"),
    )


def _run_ingest(ctx: StageContext) -> None:
    if not ctx.inputs:
        logger.info("ingest: no inputs given; skipping (using already-imported profiles).")
        return
    from trialmatchai.orchestration import ingest_inputs

    ingest_inputs(
        ctx.config,
        ctx.inputs,
        input_format=ctx.input_format,
        with_entities=ctx.with_entities,
        force=ctx.forced("ingest"),
    )


def _run_expand(ctx: StageContext) -> None:
    from trialmatchai.orchestration import expand_queries

    expand_queries(ctx.config, force=ctx.forced("expand"))


def _run_match(ctx: StageContext) -> None:
    from trialmatchai.orchestration import run_matching

    rc = run_matching(ctx.config, resume=True, force=ctx.forced("match"))
    if rc != 0:
        raise RuntimeError(f"match stage returned exit code {rc}")


def _run_eval(ctx: StageContext) -> None:
    if ctx.qrels is None:
        logger.info("eval: no qrels provided; skipping (not a benchmark run).")
        return
    from trialmatchai.trec import qrels as qrels_mod
    from trialmatchai.utils.file_utils import write_json_file

    results_dir = Path(ctx.results_dir or ctx.config["paths"]["output_dir"])
    metrics = qrels_mod.evaluate(ctx.qrels, results_dir)
    write_json_file(metrics, str(results_dir / "evaluation_metrics.json"))
    logger.info("eval: %s", metrics.get("mean"))


@dataclass(frozen=True)
class Stage:
    name: str
    run: Callable[[StageContext], None]
    help: str


# The canonical ordered pipeline. (acquire/fetch live in bootstrap-data /
# update-registry today; they will join this registry as `fetch` is unified.)
STAGES: tuple[Stage, ...] = (
    Stage("prepare", _run_prepare, "embed + entity-annotate the trial corpus"),
    Stage("concepts", _run_concepts, "build the entity-linking concept store"),
    Stage("link", _run_link, "link extracted entities to concept IDs (idempotent)"),
    Stage("index", _run_index, "build the LanceDB search tables"),
    Stage("ingest", _run_ingest, "import patient inputs into canonical profiles"),
    Stage("expand", _run_expand, "CoT query expansion of patient summaries"),
    Stage("match", _run_match, "retrieval + reranking + CoT eligibility + ranking"),
    Stage("eval", _run_eval, "score results against qrels (benchmark runs)"),
)
STAGE_NAMES: tuple[str, ...] = tuple(s.name for s in STAGES)


def _validate(names: Iterable[str]) -> None:
    bad = sorted(set(names) - set(STAGE_NAMES))
    if bad:
        raise ValueError(f"unknown stage(s) {bad}; valid stages: {list(STAGE_NAMES)}")


def select_stages(
    *,
    only: Sequence[str] | None = None,
    skip: Sequence[str] = (),
    from_stage: str | None = None,
    to_stage: str | None = None,
) -> list[Stage]:
    """Resolve the user's selection into an ordered list of stages to run."""
    if only:
        _validate(only)
        chosen = set(only)
        return [s for s in STAGES if s.name in chosen]

    for endpoint in (from_stage, to_stage):
        if endpoint is not None:
            _validate([endpoint])
    _validate(skip)

    start = STAGE_NAMES.index(from_stage) if from_stage else 0
    end = STAGE_NAMES.index(to_stage) + 1 if to_stage else len(STAGES)
    if start > end - 1:
        raise ValueError(f"--from {from_stage} is after --to {to_stage}")
    skipped = set(skip)
    return [s for s in STAGES[start:end] if s.name not in skipped]


def run_pipeline(
    ctx: StageContext,
    *,
    only: Sequence[str] | None = None,
    skip: Sequence[str] = (),
    from_stage: str | None = None,
    to_stage: str | None = None,
) -> int:
    """Run the selected pipeline slice, freeing GPU models once at the end."""
    stages = select_stages(only=only, skip=skip, from_stage=from_stage, to_stage=to_stage)
    if not stages:
        logger.warning("No stages selected; nothing to do.")
        return 0
    logger.info("Pipeline: %s", " -> ".join(s.name for s in stages))
    try:
        for stage in stages:
            logger.info("================ stage: %s ================", stage.name)
            stage.run(ctx)
    finally:
        from trialmatchai.orchestration import free_models

        free_models()
    logger.info("Pipeline complete: %s", " -> ".join(s.name for s in stages))
    return 0
