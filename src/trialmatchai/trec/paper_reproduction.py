"""Audit the immutable result bundle published with the TrialMatchAI paper.

This module deliberately separates three questions that are easy to conflate:

* do the archived per-topic metrics aggregate to the archived summary;
* can those metrics be recalculated from the archived rankings; and
* what does the current evaluator report for the same rankings.

It does not claim to rerun model inference.  A fresh inference reproduction also
needs the historical corpus, Elasticsearch setup, model revisions, adapters and
GPU runtime used by the study.
"""

from __future__ import annotations

import json
import math
import shutil
import stat
import tempfile
import uuid
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from statistics import mean, median

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from trialmatchai.trec.qrels import evaluate, parse_qrels
from trialmatchai.utils.file_utils import write_json_file
from trialmatchai.utils.integrity import sha256_file, verify_sha256

PAPER_RESULTS_URL = (
    "https://zenodo.org/records/15045515/files/matching_results.zip?download=1"
)
PAPER_RESULTS_SHA256 = "dbfd11f19ffbfd78acfeaeee634c3e7e7ca8cb4d72b526ff8d191f9e880312c8"
PAPER_RESULTS_FILENAME = "matching_results.zip"
TRACKS = {
    "21": {"directory": "TREC21", "prefix": "trec-2021", "topics": 75},
    "22": {"directory": "TREC22", "prefix": "trec-2022", "topics": 50},
}
QRELS_SHA256 = {
    "21": "ba7a2cddc90285e75cd76adcd483394a6c9bacf7017113222058ba6537e6d8ac",
    "22": "e569a531489e03f7b1fab03fe169c8ea66f4a59e8180fa9858b1a6e4bdcb0c5c",
}
RANKING_CUTOFFS = (5, 10, 20)
RECALL_CUTOFFS = (10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000)
_REPRODUCTION_FILENAMES = {
    "average_metrics.json",
    "evaluation_metrics.json",
    "ranked_trials.json",
    "nct_ids.txt",
    *(f"nct_ids_{cutoff}.txt" for cutoff in RECALL_CUTOFFS),
}


@retry(
    retry=retry_if_exception_type(requests.RequestException),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    reraise=True,
)
def _download_once(url: str, destination: Path) -> None:
    partial = destination.with_name(f".{destination.name}.part-{uuid.uuid4().hex}")
    try:
        with requests.get(url, stream=True, timeout=(15, 180)) as response:
            response.raise_for_status()
            with partial.open("wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        output.write(chunk)
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def obtain_results_archive(
    workdir: Path,
    *,
    archive: Path | None = None,
    allow_download: bool = True,
) -> Path:
    """Return a checksum-verified copy of the official matching-results archive."""
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    source = Path(archive).resolve() if archive is not None else workdir / PAPER_RESULTS_FILENAME
    if source.is_symlink() or (source.exists() and not source.is_file()):
        raise ValueError(f"Paper result archive must be a regular file: {source}")
    if not source.exists():
        if archive is not None or not allow_download:
            raise FileNotFoundError(f"Paper result archive not found: {source}")
        _download_once(PAPER_RESULTS_URL, source)
    try:
        verify_sha256(source, PAPER_RESULTS_SHA256)
    except ValueError:
        if archive is not None or not allow_download:
            raise
        quarantine = source.with_name(f".{source.name}.corrupt-{uuid.uuid4().hex}")
        source.replace(quarantine)
        _download_once(PAPER_RESULTS_URL, source)
        verify_sha256(source, PAPER_RESULTS_SHA256)
    return source


def _validated_member(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if (
        not name
        or path.is_absolute()
        or "\\" in name
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"Archive contains an unsafe path: {name!r}")
    return path


def extract_reproduction_files(archive: Path, workdir: Path) -> Path:
    """Extract only files needed for metric reproduction, atomically and safely."""
    workdir = Path(workdir).resolve()
    target = workdir / "matching_results"
    marker = target / ".paper-results.json"
    try:
        state = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        state = {}
    cache_complete = state.get("archive_sha256") == PAPER_RESULTS_SHA256 and all(
        (target / str(spec["directory"]) / "average_metrics.json").is_file()
        and len(list((target / str(spec["directory"])).glob("*/evaluation_metrics.json")))
        == int(spec["topics"])
        and len(list((target / str(spec["directory"])).glob("*/ranked_trials.json")))
        == int(spec["topics"])
        for spec in TRACKS.values()
    )
    if cache_complete:
        return target

    workdir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".paper-results-", dir=workdir) as temporary:
        staging = Path(temporary) / "matching_results"
        staging.mkdir()
        extracted = 0
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                member_path = _validated_member(member.filename)
                mode = member.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise ValueError(f"Archive contains a symbolic link: {member.filename}")
                parts = member_path.parts
                if member.is_dir() or len(parts) < 3 or parts[0] != "matching_results":
                    continue
                if parts[1] not in {spec["directory"] for spec in TRACKS.values()}:
                    continue
                if parts[-1] not in _REPRODUCTION_FILENAMES:
                    continue
                relative = Path(*parts[1:])
                destination = staging / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(member) as source, destination.open("wb") as output:
                    shutil.copyfileobj(source, output)
                extracted += 1
        if extracted < sum(spec["topics"] for spec in TRACKS.values()) * 3:
            raise ValueError("Paper archive is missing the expected TREC reproduction files")
        write_json_file(
            {"schema": 1, "archive_sha256": PAPER_RESULTS_SHA256, "files": extracted},
            str(staging / ".paper-results.json"),
        )
        previous = None
        if target.exists() or target.is_symlink():
            if target.is_symlink() or not target.is_dir():
                raise ValueError(f"Extraction target must be a directory: {target}")
            previous = workdir / f".matching-results-previous-{uuid.uuid4().hex}"
            target.replace(previous)
        try:
            staging.replace(target)
        except BaseException:
            if previous is not None and not target.exists():
                previous.replace(target)
            raise
        if previous is not None:
            shutil.rmtree(previous)
    return target


def _ranked_ids(path: Path) -> list[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    items = value.get("RankedTrials", []) if isinstance(value, dict) else value
    if not isinstance(items, list):
        raise ValueError(f"Invalid ranked trial payload: {path}")
    return [str(item["TrialID"]) for item in items if isinstance(item, dict) and item.get("TrialID")]


def _dcg(grades: Sequence[int], cutoff: int) -> float:
    return sum(grade / math.log2(rank + 2) for rank, grade in enumerate(grades[:cutoff]))


def paper_ranking_metrics(
    ranked_ids: Sequence[str], judgments: Mapping[str, int]
) -> dict[str, float]:
    """Recalculate the metric convention used by the published result files.

    Unjudged trials are removed before applying each cutoff.  nDCG uses linear
    grades and preserves the archived order within score ties.  ``p@k`` is the
    normalized graded precision ``sum(grade) / (2*k)``.
    """
    grades = [int(judgments[trial]) for trial in ranked_ids if trial in judgments]
    ideal = sorted((int(grade) for grade in judgments.values()), reverse=True)
    metrics: dict[str, float] = {}
    for cutoff in RANKING_CUTOFFS:
        ideal_dcg = _dcg(ideal, cutoff)
        metrics[f"ndcg@{cutoff}"] = _dcg(grades, cutoff) / ideal_dcg if ideal_dcg else 0.0
        metrics[f"p@{cutoff}"] = sum(grades[:cutoff]) / (2.0 * cutoff)
    return metrics


def _summaries(rows: Sequence[Mapping[str, float]], keys: Sequence[str]) -> dict[str, dict]:
    return {
        key: {"mean": mean(float(row[key]) for row in rows), "median": median(float(row[key]) for row in rows)}
        for key in keys
    }


def _close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)


def audit_track(track: str, results_root: Path, qrels_path: Path) -> dict:
    """Audit one archived TREC track against official qrels."""
    spec = TRACKS[track]
    track_dir = Path(results_root) / str(spec["directory"])
    qrels = parse_qrels(Path(qrels_path), str(spec["prefix"]))
    expected_topics = int(spec["topics"])
    if len(qrels) != expected_topics:
        raise ValueError(
            f"TREC {track} qrels contain {len(qrels)} topics; expected {expected_topics}"
        )

    metric_keys = tuple(
        key for cutoff in RANKING_CUTOFFS for key in (f"ndcg@{cutoff}", f"p@{cutoff}")
    )
    stored_rows: list[dict[str, float]] = []
    recalculated_rows: list[dict[str, float]] = []
    mismatch_topics: list[dict] = []
    recall_rows: list[dict[str, float]] = []

    for query_id, judgments in qrels.items():
        patient_dir = track_dir / query_id
        if not patient_dir.is_dir():
            raise ValueError(f"Paper archive is missing topic directory: {patient_dir}")
        stored = json.loads((patient_dir / "evaluation_metrics.json").read_text(encoding="utf-8"))
        ranked = _ranked_ids(patient_dir / "ranked_trials.json")
        recalculated = paper_ranking_metrics(ranked, judgments)
        stored_rows.append(stored)
        recalculated_rows.append(recalculated)
        differences = {
            key: {"stored": float(stored[key]), "recalculated": recalculated[key]}
            for key in metric_keys
            if not _close(stored[key], recalculated[key])
        }
        if differences:
            mismatch_topics.append({"topic": query_id, "metrics": differences})

        relevant = {trial for trial, grade in judgments.items() if grade >= 1}
        recall_row = {}
        for cutoff in RECALL_CUTOFFS:
            ids_path = patient_dir / f"nct_ids_{cutoff}.txt"
            retrieved = {line.strip() for line in ids_path.read_text().splitlines() if line.strip()}
            recall_row[f"recall@{cutoff}"] = len(retrieved & relevant) / len(relevant)
        recall_rows.append(recall_row)

    archived_summary = json.loads((track_dir / "average_metrics.json").read_text(encoding="utf-8"))
    stored_summary = _summaries(stored_rows, metric_keys)
    recalculated_summary = _summaries(recalculated_rows, metric_keys)
    recall_keys = tuple(f"recall@{cutoff}" for cutoff in RECALL_CUTOFFS)
    recall_summary = _summaries(recall_rows, recall_keys)
    summary_mismatches = {
        key: {"archived": float(archived_summary[key]), "aggregated": stored_summary[key]["mean"]}
        for key in metric_keys
        if not _close(archived_summary[key], stored_summary[key]["mean"])
    }

    current_by_policy = {}
    for policy in ("exclude", "include_as_zero"):
        current = evaluate(
            qrels, track_dir, cutoffs=RECALL_CUTOFFS, unjudged_policy=policy
        )
        current_by_policy[policy] = {
            key: {"mean": value, "median": current["median"][key]}
            for key, value in current["mean"].items()
            if value is not None
        }
    current_summary = current_by_policy["exclude"]

    return {
        "track": f"TREC20{track}",
        "topics": len(qrels),
        "archived_summary": archived_summary,
        "stored_per_topic": stored_summary,
        "recalculated_from_rankings": recalculated_summary,
        "paper_retrieval_recall": recall_summary,
        "current_evaluator": current_summary,
        "current_evaluator_by_unjudged_policy": current_by_policy,
        "checks": {
            "archived_summary_matches_stored_topics": not summary_mismatches,
            "rankings_match_stored_topic_metrics": not mismatch_topics,
            "summary_mismatches": summary_mismatches,
            "ranking_mismatch_count": len(mismatch_topics),
            "ranking_mismatches": mismatch_topics,
        },
    }


def reproduce_paper_results(results_root: Path, qrels_dir: Path) -> dict:
    for track, expected in QRELS_SHA256.items():
        verify_sha256(Path(qrels_dir) / f"qrels_{track}.txt", expected)
    tracks = {
        track: audit_track(track, results_root, Path(qrels_dir) / f"qrels_{track}.txt")
        for track in TRACKS
    }
    total_topics = sum(item["topics"] for item in tracks.values())

    def pooled(section: str, key: str) -> float:
        return sum(
            item[section][key]["mean"] * item["topics"] for item in tracks.values()
        ) / total_topics

    pooled_means = {
        "stored_per_topic": {
            key: pooled("stored_per_topic", key)
            for key in tracks["21"]["stored_per_topic"]
        },
        "recalculated_from_rankings": {
            key: pooled("recalculated_from_rankings", key)
            for key in tracks["21"]["recalculated_from_rankings"]
        },
        "current_evaluator": {
            key: pooled("current_evaluator", key)
            for key in tracks["21"]["current_evaluator"]
            if key in tracks["22"]["current_evaluator"]
        },
        "current_evaluator_by_unjudged_policy": {
            policy: {
                key: sum(
                    item["current_evaluator_by_unjudged_policy"][policy][key]["mean"]
                    * item["topics"]
                    for item in tracks.values()
                )
                / total_topics
                for key in tracks["21"]["current_evaluator_by_unjudged_policy"][policy]
                if key in tracks["22"]["current_evaluator_by_unjudged_policy"][policy]
            }
            for policy in ("exclude", "include_as_zero")
        },
    }
    return {
        "schema": 1,
        "scope": "published-result artifact audit; model inference was not rerun",
        "paper": "https://doi.org/10.1038/s41467-026-70509-w",
        "artifact": {
            "url": PAPER_RESULTS_URL,
            "sha256": PAPER_RESULTS_SHA256,
        },
        "qrels_sha256": {
            track: sha256_file(Path(qrels_dir) / f"qrels_{track}.txt") for track in TRACKS
        },
        "status": (
            "verified_with_ranking_discrepancies"
            if any(not item["checks"]["rankings_match_stored_topic_metrics"] for item in tracks.values())
            else "verified"
        ),
        "all_125_topics_weighted_mean": pooled_means,
        "tracks": tracks,
    }
