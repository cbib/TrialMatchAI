from __future__ import annotations

import argparse
import sys
from pathlib import Path

from trialmatchai.config.config_loader import load_config
from trialmatchai.search import LanceDBSearchBackend
from trialmatchai.utils.logging_config import setup_logging

logger = setup_logging(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build TrialMatchAI LanceDB search tables."
    )
    parser.add_argument("--config", default=None, help="Path to TrialMatchAI config JSON")
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Prepare embeddings/entities from normalized trial JSONs before indexing.",
    )
    parser.add_argument(
        "--trials-json-folder",
        default=None,
        help="Folder containing normalized trial JSON files. Defaults to config paths.trials_json_folder.",
    )
    parser.add_argument(
        "--processed-trials-folder",
        default="data/processed_trials",
        help="Folder containing prepared trial JSON files.",
    )
    parser.add_argument(
        "--processed-criteria-folder",
        default="data/processed_criteria",
        help="Folder containing prepared criteria subfolders.",
    )
    parser.add_argument(
        "--skip-trials",
        action="store_true",
        help="Do not build the trial table.",
    )
    parser.add_argument(
        "--skip-criteria",
        action="store_true",
        help="Do not build the criteria table.",
    )
    parser.add_argument(
        "--recreate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overwrite target tables before writing.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    backend = LanceDBSearchBackend.from_config(config)
    root = _repo_root()
    failures = 0
    if args.prepare:
        _prepare_from_trials_jsons(
            config=config,
            trials_json_folder=_resolve_path(
                args.trials_json_folder or config["paths"]["trials_json_folder"],
                root,
            ),
            processed_trials_folder=_resolve_path(args.processed_trials_folder, root),
            processed_criteria_folder=_resolve_path(args.processed_criteria_folder, root),
        )

    if not args.skip_trials:
        trials_folder = _resolve_path(args.processed_trials_folder, root)
        trial_docs = _load_flat_json_folder(trials_folder)
        if not trial_docs:
            logger.error("No prepared trial JSON files found in %s.", trials_folder)
            failures += 1
        else:
            count = backend.index_trials(trial_docs, recreate=args.recreate)
            logger.info("Indexed %s trial documents.", count)

    if not args.skip_criteria:
        criteria_folder = _resolve_path(args.processed_criteria_folder, root)
        criteria_docs = _load_nested_json_folder(criteria_folder)
        if not criteria_docs:
            logger.error("No prepared criteria JSON files found in %s.", criteria_folder)
            failures += 1
        else:
            count = backend.index_criteria(criteria_docs, recreate=args.recreate)
            logger.info("Indexed %s criteria documents.", count)

    if failures:
        return 1
    logger.info("Search tables ready at %s.", backend.db_path)
    return 0


def _load_flat_json_folder(folder: Path) -> list[dict]:
    if not folder.exists():
        return []
    return [
        _read_json(path)
        for path in sorted(folder.glob("*.json"))
        if path.is_file()
    ]


def _load_nested_json_folder(folder: Path) -> list[dict]:
    if not folder.exists():
        return []
    return [
        _read_json(path)
        for path in sorted(folder.glob("*/*.json"))
        if path.is_file()
    ]


def _read_json(path: Path) -> dict:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _prepare_from_trials_jsons(
    *,
    config: dict,
    trials_json_folder: Path,
    processed_trials_folder: Path,
    processed_criteria_folder: Path,
) -> None:
    from trialmatchai.entities import build_entity_annotator
    from trialmatchai.models.embedding.text_embedder import TextEmbedder, TextEmbedderConfig
    from trialmatchai.registry.preparation import (
        prepare_criteria_documents,
        prepare_trial_document,
        write_prepared_criteria,
        write_prepared_trial,
    )

    embedder_cfg = config.get("embedder", {})
    embedder = TextEmbedder(
        TextEmbedderConfig(
            model_name=embedder_cfg.get("model_name", "BAAI/bge-m3"),
            revision=embedder_cfg.get("revision"),
            trust_remote_code=embedder_cfg.get("trust_remote_code", False),
            pooling=embedder_cfg.get("pooling", "mean"),
            max_length=embedder_cfg.get("max_length", 512),
            batch_size=embedder_cfg.get("batch_size", 32),
            use_gpu=embedder_cfg.get("use_gpu", True),
            use_fp16=embedder_cfg.get("use_fp16", False),
            normalize=embedder_cfg.get("normalize", True),
        )
    )
    entity_annotator = build_entity_annotator(config, embedder=embedder)
    trial_docs = _load_flat_json_folder(trials_json_folder)
    for doc in trial_docs:
        trial_row = prepare_trial_document(doc, embedder)
        criteria_rows = prepare_criteria_documents(
            doc,
            embedder,
            entity_annotator=entity_annotator,
        )
        write_prepared_trial(trial_row, processed_trials_folder)
        write_prepared_criteria(criteria_rows, processed_criteria_folder)
    logger.info("Prepared %s trial JSON files from %s.", len(trial_docs), trials_json_folder)


def _resolve_path(value: str, root: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (root / path).resolve()


def _repo_root() -> Path:
    start = Path(__file__).resolve()
    for parent in start.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd().resolve()


if __name__ == "__main__":
    sys.exit(main())
