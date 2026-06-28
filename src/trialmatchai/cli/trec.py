"""``trialmatchai trec`` — end-to-end evaluation over the TREC CT tracks.

A preset over the core e2e orchestration: converts TREC patient topics, builds a
per-track search index restricted to each track's NCT collection, and runs
matching with per-patient resume. Idempotent — re-running skips finished work.
"""

from __future__ import annotations

import argparse
import sys

from trialmatchai.trec.corpus import DEFAULT_TRACKS, TRACK_KEYS
from trialmatchai.trec.runner import run_tracks


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run TrialMatchAI end-to-end on the official TREC Clinical Trials tracks."
    )
    parser.add_argument("--config", default=None, help="Path to TrialMatchAI config JSON")
    parser.add_argument(
        "--tracks",
        default=" ".join(DEFAULT_TRACKS),
        help=f"Space-separated track keys to run. Choices: {', '.join(TRACK_KEYS)}. "
        f"Default: {' '.join(DEFAULT_TRACKS)} (sigir has no official direct source).",
    )
    parser.add_argument("--data-dir", default="data", help="Base data directory.")
    parser.add_argument(
        "--results-root", default=".", help="Root under which results_trec<track>/ are written."
    )
    parser.add_argument(
        "--processed-trials-folder",
        default=None,
        help="Prepared trial JSONs for indexing. Defaults to <data-dir>/processed_trials. "
        "Point at /nfs/scratch to build the index from there while data-dir lives on /nfs/home.",
    )
    parser.add_argument(
        "--processed-criteria-folder",
        default=None,
        help="Prepared criteria subfolders for indexing. Defaults to <data-dir>/processed_criteria.",
    )
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="Convert + build indexes only; skip matching (e.g. on a CPU node).",
    )
    parser.add_argument("--reindex", action="store_true", help="Rebuild indexes even if present.")
    parser.add_argument("--rematch", action="store_true", help="Re-match patients even if results exist.")
    parser.add_argument("--no-eval", action="store_true", help="Skip recall@k evaluation against qrels.")
    args = parser.parse_args()

    track_keys = args.tracks.split()
    return run_tracks(
        track_keys,
        config_path=args.config,
        data_dir=args.data_dir,
        results_root=args.results_root,
        processed_trials_folder=args.processed_trials_folder,
        processed_criteria_folder=args.processed_criteria_folder,
        index_only=args.index_only,
        evaluate=not args.no_eval,
        force_reindex=args.reindex,
        force_rematch=args.rematch,
    )


if __name__ == "__main__":
    sys.exit(main())
