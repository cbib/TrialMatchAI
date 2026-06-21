from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="trialmatchai",
        description="TrialMatchAI command group.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("healthcheck", help="Run deployment health checks")
    subparsers.add_parser("index", help="Build LanceDB search tables")
    subparsers.add_parser("build-concepts", help="Build LanceDB concept table")
    subparsers.add_parser("update-registry", help="Fetch and upsert registry studies")
    subparsers.add_parser("run", help="Run the matching pipeline")

    args, remainder = parser.parse_known_args()
    if args.command == "healthcheck":
        from trialmatchai.cli.healthcheck import main as command
    elif args.command == "index":
        from trialmatchai.cli.index_data import main as command
    elif args.command == "build-concepts":
        from trialmatchai.cli.build_concepts import main as command
    elif args.command == "update-registry":
        from trialmatchai.cli.update_registry import main as command
    elif args.command == "run":
        from trialmatchai.cli.run import main as command
    else:  # pragma: no cover - argparse enforces choices
        parser.error(f"Unknown command: {args.command}")

    sys.argv = [f"trialmatchai {args.command}", *remainder]
    return command()


if __name__ == "__main__":
    sys.exit(main())
