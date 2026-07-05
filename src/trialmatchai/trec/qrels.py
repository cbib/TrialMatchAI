"""Official TREC relevance judgments (qrels): download, parse, corpus, metrics.

The per-track NCT corpus pool is derived directly from the judged trials, and
recall@k is scored against the same qrels. Relevance grades: 0 = not relevant,
1 = excluded (condition matches but patient excluded), 2 = eligible. Default
threshold counts grade >= 1 as relevant; pass ``threshold=2`` for eligible-only.
"""

from __future__ import annotations

import json
from pathlib import Path

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from trialmatchai.trec.metrics import (
    condensed_ndcg,
    graded_precision_at_k,
    precision_at_k,
)
from trialmatchai.utils.logging_config import setup_logging

logger = setup_logging(__name__)

QRELS_URLS: dict[str, str] = {
    "21": "https://trec.nist.gov/data/trials/qrels2021.txt",
    "22": "https://trec.nist.gov/data/trials/qrels2022.txt",
}

DEFAULT_CUTOFFS = (10, 50, 100, 200, 300, 500, 1000)
NDCG_CUTOFFS = (5, 10, 20)
P_CUTOFF = 10


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
    tmp = dest.with_name(dest.name + ".part")
    tmp.write_bytes(_http_get(QRELS_URLS[track]))
    tmp.replace(dest)  # atomic: a crash mid-download must not leave a truncated cache
    return dest


def parse_qrels(path: Path, id_prefix: str) -> dict[str, dict[str, int]]:
    """Parse a TREC qrels file (``<topic> <iter> <nct_id> <rel>``) into {query_id: {nct_id: rel}}.

    The query id is ``f"{id_prefix}{topic}"`` to match the imported topic ids.
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

    Prefers the first-level candidate list (nct_ids.txt) so recall at large
    cutoffs is meaningful; falls back to the final ranked_trials.json.
    """
    nct_ids = patient_dir / "nct_ids.txt"
    if nct_ids.exists():
        return [line.strip() for line in nct_ids.read_text().splitlines() if line.strip()]
    ranked = patient_dir / "ranked_trials.json"
    if ranked.exists():
        data = json.loads(ranked.read_text())
        items = data.get("RankedTrials", []) if isinstance(data, dict) else data
        return [
            str(item["TrialID"])
            for item in items or []
            if isinstance(item, dict) and item.get("TrialID") is not None
        ]
    return []


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float | None:
    if not relevant:
        return None
    hits = sum(1 for nct in retrieved[:k] if nct in relevant)
    return hits / len(relevant)


def _ranked_with_scores(patient_dir: Path) -> tuple[list[str], dict[str, float]]:
    """Final ranked NCT ids (in order) + their eligibility scores from ranked_trials.json."""
    ranked = patient_dir / "ranked_trials.json"
    if not ranked.exists():
        return [], {}
    try:
        data = json.loads(ranked.read_text())
    except Exception:
        return [], {}
    items = data.get("RankedTrials", []) if isinstance(data, dict) else data
    order: list[str] = []
    score_of: dict[str, float] = {}
    for item in items or []:
        if not isinstance(item, dict) or item.get("TrialID") is None:
            continue
        nid = str(item["TrialID"])
        order.append(nid)
        score_of[nid] = float(item.get("Score", 0.0))
    return order, score_of


def _mean(sums: dict, counts: dict) -> dict:
    return {key: (sums[key] / counts[key] if counts[key] else None) for key in sums}


def evaluate(
    qrels: dict[str, dict[str, int]],
    results_dir: Path,
    *,
    cutoffs: tuple[int, ...] = DEFAULT_CUTOFFS,
    threshold: int = 1,
) -> dict:
    """Per-query and mean metrics over the patients in ``results_dir``.

    Reports recall@k (retrieval, first-level list) and tie-aware nDCG@{5,10,20}
    + P@10 (ranking, condensed to judged trials). P@10 is split into "relevant"
    (grade>=1) and "eligible" (grade==2).
    """
    results_dir = Path(results_dir)
    relevant = relevant_ncts(qrels, threshold=threshold)
    eligible = relevant_ncts(qrels, threshold=2)
    per_query: dict[str, dict] = {}

    rec_sums = {f"recall@{k}": 0.0 for k in cutoffs}
    rec_counts = {f"recall@{k}": 0 for k in cutoffs}
    rank_sums = {f"ndcg@{k}": 0.0 for k in NDCG_CUTOFFS}
    rank_sums.update({f"ndcg_full@{k}": 0.0 for k in NDCG_CUTOFFS})
    rank_sums[f"P@{P_CUTOFF}(rel>=1)"] = 0.0
    rank_sums[f"P@{P_CUTOFF}(eligible)"] = 0.0
    rank_sums[f"graded_P@{P_CUTOFF}"] = 0.0
    rank_counts = {key: 0 for key in rank_sums}

    for query_id, judgments in qrels.items():
        patient_dir = results_dir / query_id
        rel_set = relevant.get(query_id, set())
        if not patient_dir.is_dir() or not rel_set:
            continue
        retrieved = _retrieved_for_patient(patient_dir)
        ranked, score_of = _ranked_with_scores(patient_dir)
        row = {
            "num_relevant": len(rel_set),
            "num_retrieved": len(retrieved),
            "num_ranked": len(ranked),
        }
        for k in cutoffs:
            r = recall_at_k(retrieved, rel_set, k)
            row[f"recall@{k}"] = r
            if r is not None:
                rec_sums[f"recall@{k}"] += r
                rec_counts[f"recall@{k}"] += 1

        if ranked:
            # Report nDCG under both IDCG bases: ndcg@k normalizes by the ideal over
            # judged-AND-ranked trials (recall-independent — pure ordering quality), while
            # ndcg_full@k normalizes by the ideal over the FULL judged pool (recall-aware,
            # trec_eval-style — a relevant trial never ranked lowers the score). The DCG
            # numerator is condensed (ignores unjudged) in both.
            ndcg = condensed_ndcg(ranked, score_of, judgments, NDCG_CUTOFFS)
            ndcg_full = condensed_ndcg(ranked, score_of, judgments, NDCG_CUTOFFS, full_ideal=True)
            for k in NDCG_CUTOFFS:
                row[f"ndcg@{k}"] = ndcg[k]
                row[f"ndcg_full@{k}"] = ndcg_full[k]
                rank_sums[f"ndcg@{k}"] += ndcg[k]
                rank_sums[f"ndcg_full@{k}"] += ndcg_full[k]
                rank_counts[f"ndcg@{k}"] += 1
                rank_counts[f"ndcg_full@{k}"] += 1
            # Condense the ranked list to the judged pool before the precision cutoff, so an
            # unjudged trial the assessors never saw does not count as a miss. This matches the
            # condensed_ndcg above (and the docstring's stated "condensed to judged trials"):
            # a non-pooled system surfaces many unjudged trials, and scoring those as wrong
            # understated P@k and made it inconsistent with nDCG.
            judged_ranked = [nid for nid in ranked if nid in judgments]
            p_rel = precision_at_k(judged_ranked, rel_set, P_CUTOFF)
            p_elig = precision_at_k(judged_ranked, eligible.get(query_id, set()), P_CUTOFF)
            p_graded = graded_precision_at_k(judged_ranked, judgments, P_CUTOFF)
            row[f"P@{P_CUTOFF}(rel>=1)"] = p_rel
            row[f"P@{P_CUTOFF}(eligible)"] = p_elig
            row[f"graded_P@{P_CUTOFF}"] = p_graded
            rank_sums[f"P@{P_CUTOFF}(rel>=1)"] += p_rel
            rank_sums[f"P@{P_CUTOFF}(eligible)"] += p_elig
            rank_sums[f"graded_P@{P_CUTOFF}"] += p_graded
            rank_counts[f"P@{P_CUTOFF}(rel>=1)"] += 1
            rank_counts[f"P@{P_CUTOFF}(eligible)"] += 1
            rank_counts[f"graded_P@{P_CUTOFF}"] += 1
        per_query[query_id] = row

    mean = {**_mean(rec_sums, rec_counts), **_mean(rank_sums, rank_counts)}
    return {
        "recall_relevance_threshold": threshold,
        "num_queries_scored": len(per_query),
        "num_queries_ranked": rank_counts[f"ndcg@{NDCG_CUTOFFS[0]}"],
        "mean": mean,
        "per_query": per_query,
    }
