"""Backfill a TREC track's judged-trial corpus from the live ClinicalTrials.gov API.

The per-track corpus pool derives from the official qrels; any judged NCT missing
from the normalized trials folder is fetched from the CT.gov v2 API (JSON) and
written as normalized trial JSON, so the standard build stages (prepare -> index)
can pick it up. Network-only — no GPU, safe on a login/CPU node.

    python -m trialmatchai.trec.backfill --tracks "23"

Needed for TREC 2023, whose May-2023 snapshot corpus extends far beyond the
bootstrap TREC corpus; harmless (a no-op) for tracks already fully present.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from trialmatchai.registry.clinicaltrials_gov import ClinicalTrialsGovClient
from trialmatchai.registry.normalization import normalize_study
from trialmatchai.trec import qrels as qrels_mod
from trialmatchai.trec.corpus import resolve_tracks
from trialmatchai.utils.file_utils import write_json_file
from trialmatchai.utils.logging_config import setup_logging

logger = setup_logging(__name__)


def track_corpus_state(
    track: str,
    *,
    data_dir: str | Path = "data",
    trials_json_folder: str | Path | None = None,
) -> tuple[set[str], list[str]]:
    """(qrels corpus pool, sorted NCTs missing from the normalized trials folder)."""
    data_dir = Path(data_dir)
    spec = resolve_tracks([track], data_dir=data_dir, results_root=data_dir)[0]
    qrels_path = qrels_mod.download_qrels(track, spec.trec_dir / "qrels")
    pool = qrels_mod.corpus_ncts(qrels_mod.parse_qrels(qrels_path, spec.id_prefix))
    folder = Path(trials_json_folder or data_dir / "trials_jsons")
    missing = sorted(nct for nct in pool if not (folder / f"{nct}.json").exists())
    return pool, missing


def backfill_track_corpus(
    track: str,
    *,
    data_dir: str | Path = "data",
    trials_json_folder: str | Path | None = None,
    client: ClinicalTrialsGovClient | None = None,
    log_every: int = 1000,
) -> dict[str, int]:
    """Fetch + normalize every judged NCT missing from the trials folder.

    Returns counts: total pool, already present, written, failed (normalize
    errors), and unavailable (judged ids the live registry no longer returns —
    those trials stay out of the index and are dropped by the condensed eval).
    """
    folder = Path(trials_json_folder or Path(data_dir) / "trials_jsons")
    pool, missing = track_corpus_state(
        track, data_dir=data_dir, trials_json_folder=folder
    )
    stats = {
        "pool": len(pool),
        "present": len(pool) - len(missing),
        "written": 0,
        "failed": 0,
        "unavailable": 0,
    }
    logger.info(
        "Track %s corpus: %s judged trials, %s already normalized, %s to backfill.",
        track, stats["pool"], stats["present"], len(missing),
    )
    if not missing:
        return stats

    folder.mkdir(parents=True, exist_ok=True)
    client = client or ClinicalTrialsGovClient(page_size=1000)
    fetched: set[str] = set()
    for study in client.iter_studies_by_ids(missing):
        try:
            doc = normalize_study(study)
        except ValueError:
            stats["failed"] += 1
            logger.exception("Backfill: could not normalize a fetched study (continuing)")
            continue
        fetched.add(doc["nct_id"])
        write_json_file(doc, str(folder / f"{doc['nct_id']}.json"))
        stats["written"] += 1
        if stats["written"] % log_every == 0:
            logger.info("Backfill progress: %s/%s written.", stats["written"], len(missing))

    unavailable = sorted(set(missing) - fetched)
    stats["unavailable"] = len(unavailable)
    if unavailable:
        logger.warning(
            "Backfill: %s judged NCTs are no longer in the live registry (first few: %s). "
            "They cannot be indexed or retrieved.",
            len(unavailable), ", ".join(unavailable[:5]),
        )
    logger.info(
        "Backfill complete for track %s: %s written, %s failed, %s unavailable.",
        track, stats["written"], stats["failed"], stats["unavailable"],
    )
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill missing judged trials for TREC tracks from the live "
        "ClinicalTrials.gov v2 API (network-only; no GPU)."
    )
    parser.add_argument("--tracks", default="23", help="Space-separated track keys.")
    parser.add_argument("--data-dir", default="data", help="Base data directory.")
    parser.add_argument(
        "--trials-json-folder",
        default=None,
        help="Normalized trials folder to backfill into. Defaults to <data-dir>/trials_jsons.",
    )
    args = parser.parse_args()
    failures = 0
    for track in args.tracks.split():
        try:
            backfill_track_corpus(
                track,
                data_dir=args.data_dir,
                trials_json_folder=args.trials_json_folder,
            )
        except Exception:
            logger.exception("Backfill failed for track %s", track)
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
