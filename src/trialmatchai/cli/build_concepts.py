from __future__ import annotations

import argparse
import sys
from pathlib import Path

from trialmatchai.config.config_loader import load_config
from trialmatchai.entities.builder import (
    DEFAULT_OMOP_VOCABULARIES,
    build_dictionary_rows,
    build_omop_concept_rows,
    concept_texts_for_embedding,
    write_lancedb_table,
)
from trialmatchai.models.embedding import build_embedder
from trialmatchai.utils.logging_config import setup_logging

logger = setup_logging(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the TrialMatchAI LanceDB concept table."
    )
    parser.add_argument("--config", default=None, help="Path to TrialMatchAI config JSON")
    parser.add_argument(
        "--concept-csv",
        default=None,
        help="OMOP CONCEPT.csv path (optional; omit to build from --dictionary only)",
    )
    parser.add_argument(
        "--synonym-csv",
        default=None,
        help="OMOP CONCEPT_SYNONYM.csv path",
    )
    parser.add_argument(
        "--dictionary",
        action="append",
        default=[],
        metavar="VOCAB:DOMAIN:PATH",
        help="Import a concept dictionary file, e.g. EntrezGene:Gene:/path/dict_Gene.txt",
    )
    parser.add_argument(
        "--vocabulary",
        action="append",
        default=[],
        help="OMOP vocabulary to include. Defaults to TrialMatchAI deployment set.",
    )
    parser.add_argument(
        "--sources",
        choices=["open"],
        default=None,
        help="Download + convert a bundled source set. 'open' = genes, diseases, "
        "chemicals, cell lines, cell types, phenotypes (no licence required).",
    )
    parser.add_argument(
        "--concept-cache",
        default="data/concept_dicts",
        help="Directory for downloaded sources + converted dictionaries (cached).",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download/re-convert bundled sources even if cached.",
    )
    parser.add_argument("--db-path", default=None, help="Output LanceDB directory")
    parser.add_argument("--table", default=None, help="Output LanceDB table name")
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Create an FTS-only table without model embeddings.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild (re-embed) the concept table even if it already exists "
        "(default: skip if present).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if not args.concept_csv and not args.dictionary and not args.sources:
        parser.error("provide --concept-csv, --sources, and/or at least one --dictionary")
    try:
        return run_build_concepts(
            config,
            db_path=args.db_path,
            table=args.table,
            sources=args.sources,
            dictionary=args.dictionary,
            concept_csv=args.concept_csv,
            synonym_csv=args.synonym_csv,
            vocabulary=args.vocabulary,
            concept_cache=args.concept_cache,
            force_download=args.force_download,
            skip_embeddings=args.skip_embeddings,
            force=args.force,
        )
    except ValueError as exc:
        parser.error(str(exc))
        return 2  # pragma: no cover - parser.error exits


def run_build_concepts(
    config: dict,
    *,
    db_path: str | None = None,
    table: str | None = None,
    sources: str | None = None,
    dictionary: list[str] | tuple[str, ...] = (),
    concept_csv: str | None = None,
    synonym_csv: str | None = None,
    vocabulary: list[str] | tuple[str, ...] = (),
    concept_cache: str = "data/concept_dicts",
    force_download: bool = False,
    skip_embeddings: bool = False,
    force: bool = False,
) -> int:
    """Build the LanceDB concept table (the entity-linking store). Idempotent.

    Importable so the unified pipeline can invoke the concepts stage directly,
    rather than re-entering the CLI. Skips the expensive re-embed if the table is
    already present, unless ``force`` (or a new OMOP vocab via ``concept_csv``).
    """
    linker_cfg = config.get("concept_linker", {})
    db_path = db_path or linker_cfg.get("db_path") or "data/concepts"
    table_name = table or linker_cfg.get("table") or "concepts"

    if not concept_csv and not dictionary and not sources:
        raise ValueError("provide concept_csv, sources, and/or at least one dictionary")

    if not force and not concept_csv:
        ready, rows_present = _concept_table_ready(db_path, table_name)
        if ready:
            logger.info(
                "Concept store already present at %s/%s (%s concepts); skipping. "
                "Pass force=True to rebuild.",
                db_path,
                table_name,
                rows_present,
            )
            return 0

    dictionary_specs: list[tuple[str, str, str]] = [
        _parse_dictionary_spec(spec) for spec in dictionary
    ]
    if sources == "open":
        from trialmatchai.entities.concept_sources import build_open_dictionaries

        for source, dict_path in build_open_dictionaries(
            Path(concept_cache), force=force_download
        ):
            dictionary_specs.append(
                (source.vocabulary_id, source.domain_id, str(dict_path))
            )
        if not dictionary_specs and not concept_csv:
            raise ValueError("no concept sources were built (all downloads failed?)")

    vocabularies = tuple(vocabulary or DEFAULT_OMOP_VOCABULARIES)
    rows: list = []
    if concept_csv:
        rows = build_omop_concept_rows(concept_csv, synonym_csv, vocabularies=vocabularies)
    for vocab, domain, path in dictionary_specs:
        rows.extend(build_dictionary_rows(path, vocabulary_id=vocab, domain_id=domain))

    embeddings = None
    if not skip_embeddings:
        embedder = build_embedder(config)
        embeddings = embedder.embed_texts(concept_texts_for_embedding(rows))

    Path(db_path).mkdir(parents=True, exist_ok=True)
    write_lancedb_table(
        rows,
        db_path=db_path,
        table_name=table_name,
        embeddings=embeddings,
        recreate=True,
    )
    logger.info("Wrote %s concepts to %s/%s", len(rows), db_path, table_name)
    return 0


def _parse_dictionary_spec(spec: str) -> tuple[str, str, str]:
    parts = spec.split(":", 2)
    if len(parts) != 3:
        raise ValueError(
            "--dictionary must use VOCAB:DOMAIN:PATH, "
            f"received: {spec}"
        )
    return parts[0], parts[1], parts[2]


def _concept_table_ready(db_path: str, table_name: str) -> tuple[bool, int]:
    """Return (table exists and is non-empty, row count) for the concept table."""
    try:
        import lancedb

        db = lancedb.connect(str(db_path))
        if table_name not in db.table_names():
            return False, 0
        count = db.open_table(table_name).count_rows()
        return count > 0, count
    except Exception:
        return False, 0


if __name__ == "__main__":
    sys.exit(main())
