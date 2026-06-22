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
    parser.add_argument("--concept-csv", required=True, help="OMOP CONCEPT.csv path")
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
    parser.add_argument("--db-path", default=None, help="Output LanceDB directory")
    parser.add_argument("--table", default=None, help="Output LanceDB table name")
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Create an FTS-only table without model embeddings.",
    )
    parser.add_argument(
        "--recreate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overwrite the target table if it exists.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    linker_cfg = config.get("concept_linker", {})
    db_path = args.db_path or linker_cfg.get("db_path") or "data/concepts"
    table_name = args.table or linker_cfg.get("table") or "concepts"

    vocabularies = tuple(args.vocabulary or DEFAULT_OMOP_VOCABULARIES)
    rows = build_omop_concept_rows(
        args.concept_csv,
        args.synonym_csv,
        vocabularies=vocabularies,
    )
    for spec in args.dictionary:
        vocab, domain, path = _parse_dictionary_spec(spec)
        rows.extend(
            build_dictionary_rows(
                path,
                vocabulary_id=vocab,
                domain_id=domain,
            )
        )

    embeddings = None
    if not args.skip_embeddings:
        embedder = build_embedder(config)
        embeddings = embedder.embed_texts(concept_texts_for_embedding(rows))

    Path(db_path).mkdir(parents=True, exist_ok=True)
    write_lancedb_table(
        rows,
        db_path=db_path,
        table_name=table_name,
        embeddings=embeddings,
        recreate=args.recreate,
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


if __name__ == "__main__":
    sys.exit(main())
