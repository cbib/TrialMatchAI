"""CLI for auditing the result artifact published with the TrialMatchAI paper."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from collections.abc import Sequence
from pathlib import Path

import requests

from trialmatchai.trec.paper_reproduction import (
    TRACKS,
    extract_reproduction_files,
    obtain_results_archive,
    reproduce_paper_results,
)
from trialmatchai.trec.qrels import download_qrels
from trialmatchai.utils.file_utils import write_json_file


def _print_human(report: dict, output: Path) -> None:
    print(f"Paper result audit: {report['status']}")
    print(f"Verified result archive SHA-256: {report['artifact']['sha256']}")
    print(f"Report: {output}")
    pooled = report["all_125_topics_weighted_mean"]["stored_per_topic"]
    print(
        "All 125 topics, stored weighted means: "
        f"nDCG@10={pooled['ndcg@10']:.6f}, graded P@10={pooled['p@10']:.6f}"
    )
    for track in report["tracks"].values():
        checks = track["checks"]
        stored = track["stored_per_topic"]
        recalculated = track["recalculated_from_rankings"]
        current = track["current_evaluator"]
        print(f"\n{track['track']} ({track['topics']} topics)")
        print(
            "  stored per-topic aggregation: "
            + ("matches archived summary" if checks["archived_summary_matches_stored_topics"] else "MISMATCH")
        )
        print(f"  rankings differing from stored metrics: {checks['ranking_mismatch_count']}")
        print(
            "  paper-method nDCG@10 mean: "
            f"stored={stored['ndcg@10']['mean']:.6f}, "
            f"recalculated={recalculated['ndcg@10']['mean']:.6f}"
        )
        print(
            "  current-evaluator nDCG@10 mean: "
            f"{current['ndcg@10']['mean']:.6f}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify and recalculate the published TrialMatchAI TREC 2021/2022 result "
            "artifact. This audits released outputs; it does not rerun model inference."
        )
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("paper-reproduction"),
        help="Cache and report directory (default: paper-reproduction)",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        help="Use an existing official matching_results.zip instead of downloading it",
    )
    parser.add_argument(
        "--qrels-dir",
        type=Path,
        help="Directory containing qrels_21.txt and qrels_22.txt; missing files are downloaded",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Require already-cached result and qrels files (useful for offline CI)",
    )
    parser.add_argument("--json", action="store_true", help="Print the full JSON report")
    args = parser.parse_args(argv)

    workdir = args.workdir.resolve()
    try:
        if args.archive is None and not (workdir / "matching_results.zip").exists():
            print("Downloading and verifying the published result artifact...", file=sys.stderr)
        archive = obtain_results_archive(
            workdir, archive=args.archive, allow_download=not args.no_download
        )
        results = extract_reproduction_files(archive, workdir)
        qrels_dir = (args.qrels_dir or (workdir / "qrels")).resolve()
        qrels_dir.mkdir(parents=True, exist_ok=True)
        for track in TRACKS:
            path = qrels_dir / f"qrels_{track}.txt"
            if not path.exists():
                if args.no_download:
                    raise FileNotFoundError(f"Qrels file not found: {path}")
                download_qrels(track, qrels_dir)
        report = reproduce_paper_results(results, qrels_dir)
        output = workdir / "reproduction-report.json"
        write_json_file(report, str(output))
    except (OSError, ValueError, requests.RequestException, zipfile.BadZipFile) as exc:
        print(f"reproduce-paper failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_human(report, output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
