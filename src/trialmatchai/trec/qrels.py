"""Official TREC relevance judgments (qrels): download, parse, corpus, metrics.

The per-track NCT corpus pool is derived directly from the qrels (the set of
judged trials) — replacing the previously-checked-in ``Unique_NCT_IDs`` lists.
Evaluation computes recall@k of the retrieval against the same qrels.

TREC Clinical Trials relevance grades: 0 = not relevant, 1 = excluded (the trial
matches the condition but the patient is excluded), 2 = eligible. By default a
trial counts as relevant at grade >= 1 (matching the legacy recall evaluation);
pass ``threshold=2`` to score eligible-only.
"""

from __future__ import annotations

import json
from pathlib import Path

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from trialmatchai.utils.logging_config import setup_logging

logger = setup_logging(__name__)

QRELS_URLS: dict[str, str] = {
    "21": "https://trec.nist.gov/data/trials/qrels2021.txt",
    "22": "https://trec.nist.gov/data/trials/qrels2022.txt",
}

DEFAULT_CUTOFFS = (10, 50, 100, 200, 300, 500, 1000)


@retry(
    retry=retry_if_exception_type(requests.RequestException),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    reraise=True,
)
def _http_get(url: str, timeout: float = 60.0) -> bytes:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.content


def download_qrels(track: str, dest_dir: Path) -> Path:
    """Fetch (and cache) the official qrels file for a track."""
    if track not in QRELS_URLS:
        raise FileNotFoundError(
            f"No official qrels URL for track '{track}'. Provide the qrels file "
            f"manually at {Path(dest_dir) / f'qrels_{track}.txt'}."
        )
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"qrels_{track}.txt"
    if dest.exists() and dest.stat().st_size > 0:
        logger.info("Qrels for track %s already present: %s", track, dest)
        return dest
    logger.info("Downloading official qrels for track %s from %s", track, QRELS_URLS[track])
    dest.write_bytes(_http_get(QRELS_URLS[track]))
    return dest


def parse_qrels(path: Path, id_prefix: str) -> dict[str, dict[str, int]]:
    """Parse a TREC qrels file into {query_id: {nct_id: relevance}}.

    Lines are ``<topic> <iteration> <nct_id> <relevance>`` (whitespace
    separated). The query id is ``f"{id_prefix}{topic}"`` to match the imported
    topic ids and the per-patient results folders.
    """
    qrels: dict[str, dict[str, int]] = {}
    for raw in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = raw.split()
        if len(parts) < 4:
            continue
        topic, _iteration, nct_id, relevance = parts[0], parts[1], parts[2], parts[3]
        try:
            rel = int(relevance)
        except ValueError:
            continue
        query_id = f"{id_prefix}{topic.strip()}"
        qrels.setdefault(query_id, {})[nct_id.strip()] = rel
    if not qrels:
        raise ValueError(f"No judgments parsed from qrels file {path}")
    return qrels


def corpus_ncts(qrels: dict[str, dict[str, int]]) -> set[str]:
    """The judged-trial pool across all queries (used to restrict the index)."""
    pool: set[str] = set()
    for judgments in qrels.values():
        pool.update(judgments)
    return pool


def relevant_ncts(qrels: dict[str, dict[str, int]], *, threshold: int = 1) -> dict[str, set[str]]:
    return {
        query_id: {nct for nct, rel in judgments.items() if rel >= threshold}
        for query_id, judgments in qrels.items()
    }


def _retrieved_for_patient(patient_dir: Path) -> list[str]:
    """Ordered retrieved NCT ids for one patient.

    Prefers the first-level candidate list (nct_ids.txt, up to ~1000) so recall
    at large cutoffs is meaningful; falls back to the final ranked_trials.json.
    """
    nct_ids = patient_dir / "nct_ids.txt"
    if nct_ids.exists():
        return [line.strip() for line in nct_ids.read_text().splitlines() if line.strip()]
    ranked = patient_dir / "ranked_trials.json"
    if ranked.exists():
        data = json.loads(ranked.read_text())
        return [str(item.get("TrialID")) for item in data if item.get("TrialID")]
    return []


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float | None:
    if not relevant:
        return None
    hits = sum(1 for nct in retrieved[:k] if nct in relevant)
    return hits / len(relevant)


def evaluate(
    qrels: dict[str, dict[str, int]],
    results_dir: Path,
    *,
    cutoffs: tuple[int, ...] = DEFAULT_CUTOFFS,
    threshold: int = 1,
) -> dict:
    """Compute per-query and mean recall@k over the patients in ``results_dir``."""
    results_dir = Path(results_dir)
    relevant = relevant_ncts(qrels, threshold=threshold)
    per_query: dict[str, dict] = {}
    sums = {k: 0.0 for k in cutoffs}
    counts = {k: 0 for k in cutoffs}

    for query_id, rel_set in relevant.items():
        patient_dir = results_dir / query_id
        if not patient_dir.is_dir() or not rel_set:
            continue
        retrieved = _retrieved_for_patient(patient_dir)
        row = {"num_relevant": len(rel_set), "num_retrieved": len(retrieved)}
        for k in cutoffs:
            r = recall_at_k(retrieved, rel_set, k)
            row[f"recall@{k}"] = r
            if r is not None:
                sums[k] += r
                counts[k] += 1
        per_query[query_id] = row

    mean = {f"recall@{k}": (sums[k] / counts[k] if counts[k] else None) for k in cutoffs}
    return {
        "threshold": threshold,
        "num_queries_scored": len(per_query),
        "mean_recall": mean,
        "per_query": per_query,
    }
