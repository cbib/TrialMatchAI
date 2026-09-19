"""Re-evaluate completed TREC rankings without retrieval or model inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from trialmatchai.trec.metrics import UnjudgedPolicy
from trialmatchai.trec.qrels import download_qrels, evaluate, parse_qrels
from trialmatchai.utils.file_utils import write_json_file
from trialmatchai.utils.integrity import sha256_file

POLICIES: tuple[UnjudgedPolicy, ...] = ("exclude", "include_as_zero")
PREFIXES = {"21": "trec-2021", "22": "trec-2022", "23": "trec-2023"}


def _evaluation_input_sha256(results_dir: Path, query_ids: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for query_id in sorted(query_ids):
        for filename in ("nct_ids.txt", "ranked_trials.json"):
            path = results_dir / query_id / filename
            relative = path.relative_to(results_dir).as_posix().encode()
            digest.update(len(relative).to_bytes(4, "big"))
            digest.update(relative)
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def evaluate_completed_runs(
    *,
    track: str,
    results_dirs: Sequence[Path],
    qrels_path: Path,
    policies: Sequence[UnjudgedPolicy] = POLICIES,
    include_per_query: bool = False,
) -> dict:
    if track not in PREFIXES:
        raise ValueError(f"Unsupported TREC track: {track}")
    qrels = parse_qrels(qrels_path, PREFIXES[track])
    runs = []
    for supplied_results_dir in results_dirs:
        supplied_results_dir = Path(supplied_results_dir)
        results_dir = supplied_results_dir.resolve()
        policy_results = {}
        for policy in policies:
            result = evaluate(qrels, results_dir, unjudged_policy=policy)
            if not include_per_query:
                result.pop("per_query", None)
            policy_results[policy] = result
        runs.append(
            {
                "name": results_dir.parent.name,
                "results_dir": supplied_results_dir.as_posix(),
                "evaluation_input_sha256": _evaluation_input_sha256(
                    results_dir, tuple(qrels)
                ),
                "policies": policy_results,
            }
        )
    return {
        "schema": 1,
        "track": f"TREC20{track}",
        "qrels": {"path": qrels_path.as_posix(), "sha256": sha256_file(qrels_path)},
        "policies": {
            "exclude": "remove unjudged trials before applying rank cutoffs",
            "include_as_zero": "retain unjudged trials at their ranks with relevance grade zero",
        },
        "runs": runs,
    }


def _print_summary(report: dict, output: Path) -> None:
    print(f"{report['track']} evaluation comparison")
    print(f"Report: {output}")
    for run in report["runs"]:
        print(f"\n{run['name']} — {run['results_dir']}")
        for policy, result in run["policies"].items():
            metrics = result["mean"]
            print(
                f"  {policy:15s} "
                f"nDCG@10={metrics['ndcg@10']:.6f}  "
                f"nDCG_full@10={metrics['ndcg_full@10']:.6f}  "
                f"graded_P@10={metrics['graded_P@10']:.6f}  "
                f"P@10(eligible)={metrics['P@10(eligible)']:.6f}  "
                f"unjudged@10={metrics['unjudged_fraction@10']:.3f}"
            )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Re-evaluate one or more completed TREC result directories under explicit "
            "unjudged-document policies. No GPU, retrieval, reranking, or CoT inference is run."
        )
    )
    parser.add_argument("--track", required=True, choices=tuple(PREFIXES))
    parser.add_argument(
        "--results-dir",
        action="append",
        required=True,
        type=Path,
        help="Completed results_trec<track> directory; repeat to compare runs",
    )
    parser.add_argument("--qrels", type=Path, help="Official qrels file")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=POLICIES,
        default=list(POLICIES),
    )
    parser.add_argument(
        "--include-per-query",
        action="store_true",
        help="Include every topic metric in the JSON report",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("trec-evaluation-comparison.json"),
    )
    parser.add_argument("--json", action="store_true", help="Print the report as JSON")
    args = parser.parse_args(argv)

    try:
        qrels_path = args.qrels
        if qrels_path is None:
            qrels_path = download_qrels(args.track, args.data_dir / "trec" / "qrels")
        missing = [path for path in args.results_dir if not path.is_dir()]
        if missing:
            raise FileNotFoundError(f"Result directories not found: {', '.join(map(str, missing))}")
        report = evaluate_completed_runs(
            track=args.track,
            results_dirs=args.results_dir,
            qrels_path=qrels_path,
            policies=args.policies,
            include_per_query=args.include_per_query,
        )
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json_file(report, str(output))
    except (OSError, ValueError) as exc:
        print(f"trec-evaluate failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_summary(report, output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
