from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="trialmatchai",
        description="TrialMatchAI command group.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "healthcheck",
        help="Run deployment health checks",
        add_help=False,
    )
    subparsers.add_parser(
        "bootstrap-data",
        help="Download data and model artifacts",
        add_help=False,
    )
    subparsers.add_parser("index", help="Build LanceDB search tables", add_help=False)
    subparsers.add_parser(
        "build-concepts",
        help="Build LanceDB concept table",
        add_help=False,
    )
    subparsers.add_parser(
        "update-registry",
        help="Fetch and upsert registry studies",
        add_help=False,
    )
    subparsers.add_parser(
        "import-patient",
        help="Import patient data profiles",
        add_help=False,
    )
    subparsers.add_parser(
        "build",
        help="Build the system (prepare corpus + search index), resumable",
        add_help=False,
    )
    subparsers.add_parser("run", help="Run the matching pipeline", add_help=False)
    subparsers.add_parser(
        "e2e",
        help="Run the whole pipeline end-to-end (ingest -> index -> match), idempotent",
        add_help=False,
    )
    subparsers.add_parser(
        "trec",
        help="End-to-end TREC CT evaluation (preset over e2e)",
        add_help=False,
    )
    subparsers.add_parser(
        "finetune",
        help="Fine-tune the CoT / reranker / NER models",
        add_help=False,
    )

    args, remainder = parser.parse_known_args()
    if args.command == "healthcheck":
        from trialmatchai.cli.healthcheck import main as command
    elif args.command == "bootstrap-data":
        from trialmatchai.cli.bootstrap_data import main as command
    elif args.command == "index":
        from trialmatchai.cli.index_data import main as command
    elif args.command == "build-concepts":
        from trialmatchai.cli.build_concepts import main as command
    elif args.command == "update-registry":
        from trialmatchai.cli.update_registry import main as command
    elif args.command == "import-patient":
        from trialmatchai.cli.import_patient import main as command
    elif args.command == "build":
        from trialmatchai.cli.build import main as command
    elif args.command == "run":
        from trialmatchai.cli.run import main as command
    elif args.command == "e2e":
        from trialmatchai.cli.e2e import main as command
    elif args.command == "trec":
        from trialmatchai.cli.trec import main as command
    elif args.command == "finetune":
        from trialmatchai.finetuning.cli import main as command
    else:  # pragma: no cover - argparse enforces choices
        parser.error(f"Unknown command: {args.command}")

    sys.argv = [f"trialmatchai {args.command}", *remainder]
    return command()


if __name__ == "__main__":
    sys.exit(main())
