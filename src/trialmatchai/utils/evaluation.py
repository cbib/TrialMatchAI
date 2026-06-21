# trialmatchai/utils/evaluation.py
from __future__ import annotations

import csv
import json
import math
import os
import re
from typing import Any, Dict, Iterable, List, Tuple, Union

# Strict NCT id pattern (case-insensitive on input; stored uppercase)
_NCT_RE = re.compile(r"^NCT\d{8}$", re.IGNORECASE)


def load_ground_truth(trec_csv_path: str) -> Dict[str, Dict[str, int]]:
    """
    Load TREC ground-truth (query-id, corpus-id, score) from .tsv or .csv.
    Returns: { query_id: { NCT########: int(score) } }

    Notes:
      - Score semantics (TREC 2021 Clinical Trials):
          2 = eligible (relevant), 1 = excluded (ineligible), 0 = not relevant
      - We uppercase trial ids for consistency.
      - Header row is optional; delimiter auto-detected between tab/comma.
    """
    gt: Dict[str, Dict[str, int]] = {}
    if not os.path.exists(trec_csv_path):
        raise FileNotFoundError(f"TREC ground truth file not found: {trec_csv_path}")

    with open(trec_csv_path, "r", encoding="utf-8") as fh:
        sample = fh.read(4096)
        fh.seek(0)
        first_line = sample.splitlines()[0] if sample else ""
        delimiter = "\t" if ("\t" in first_line) else ","
        reader = csv.reader(fh, delimiter=delimiter)

        # Detect header
        try:
            first = next(reader)
        except StopIteration:
            return gt

        def _is_header(row: List[str]) -> bool:
            if not row:
                return False
            low = [c.strip().lower() for c in row[:3]]
            joined = "|".join(low)
            return (
                ("query-id" in joined or "qid" in joined)
                and ("corpus-id" in joined or "docid" in joined or "doc-id" in joined)
                and ("score" in joined or "label" in joined or "relevance" in joined)
            )

        if not _is_header(first):
            fh.seek(0)
            reader = csv.reader(fh, delimiter=delimiter)

        for row in reader:
            if not row or len(row) < 3:
                continue
            qid = row[0].strip()
            nid = row[1].strip().upper()
            # Only store proper NCT ids
            if not _NCT_RE.match(nid):
                continue
            try:
                score = int(float(row[2]))
            except Exception:
                continue
            gt.setdefault(qid, {})[nid] = score
    return gt


def _dcg(rels: Iterable[int], gain_scheme: str = "linear") -> float:
    """
    DCG with log2 discount. Two gain schemes:
      - 'linear': gain = rel (default; typical for TREC ndcg)
      - 'exp2'  : gain = 2^rel - 1
    """
    dcg = 0.0
    for i, rel in enumerate(rels, start=1):
        if gain_scheme == "exp2":
            gain = (2**rel) - 1.0
        else:
            gain = float(rel)
        dcg += gain / math.log2(i + 1)
    return dcg


def _idcg(ground_rels: List[int], k: int, gain_scheme: str = "linear") -> float:
    ideal = sorted(ground_rels, reverse=True)[:k]
    return _dcg(ideal, gain_scheme=gain_scheme)


def ndcg_at_k(
    pred_ids: List[str],
    ground_truth: Dict[str, int],
    k: int,
    gain_scheme: str = "linear",
) -> float:
    """
    nDCG@K using graded labels (2 eligible, 1 excluded, 0 not relevant).
    Assumes pred_ids already filtered to labelled trials (see evaluate_ranking).
    """
    if k <= 0:
        return 0.0
    pred_rels = [int(ground_truth.get(nid, 0)) for nid in pred_ids[:k]]
    dcg = _dcg(pred_rels, gain_scheme=gain_scheme)
    ideal_rels = list(ground_truth.values())
    if not ideal_rels:
        return 0.0
    idcg = _idcg(ideal_rels, k, gain_scheme=gain_scheme)
    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(
    pred_ids: List[str],
    ground_truth: Dict[str, int],
    k: int,
    eligible_threshold: int = 2,
) -> float:
    """
    Precision@K for TREC CT:
      - Count ONLY 'eligible' trials as relevant (label >= eligible_threshold, default 2).
      - Unlabelled trials are assumed removed by the caller (we don't count them).
      - Denominator is the number of labelled predictions among the first K after filtering.
    """
    if k <= 0:
        return 0.0
    topk = pred_ids[:k]
    if not topk:
        return 0.0
    rels = [
        1 if int(ground_truth.get(nid, 0)) >= eligible_threshold else 0 for nid in topk
    ]
    return sum(rels) / float(len(topk))


def _get_case_insensitive(d: Dict[str, Any], *keys: str) -> Any:
    """Fetch any matching key (case-insensitive) from dict."""
    lower_map = {k.lower(): k for k in d.keys()}
    for k in keys:
        real = lower_map.get(k.lower())
        if real is not None:
            return d[real]
    return None


def _extract_top_list(container: Any) -> Any:
    """
    If 'container' is a dict that wraps the ranked list, extract the list.
    Accepts keys: RankedTrials, ranked_trials, ranked, trials (any case).
    Otherwise return container as is.
    """
    if isinstance(container, dict):
        for k in ("RankedTrials", "ranked_trials", "ranked", "trials"):
            v = _get_case_insensitive(container, k)
            if isinstance(v, list):
                return v
    return container


def _normalize_ranked_input(ranked: Any) -> List[str]:
    """
    Accept common ranked formats and return a clean list of NCT IDs (uppercase), preserving order.
      - list[str]
      - list[dict] with id keys: TrialID, trial_id, nct_id, id, nctId (any case)
      - list[tuple/list] like (id, score)
      - dict with top-level list under RankedTrials / ranked_trials / ranked / trials
    Filters out non-NCT-shaped IDs and normalizes to uppercase.
    """
    ranked = _extract_top_list(ranked)

    ids: List[str] = []

    if not ranked:
        return ids

    # list[str]
    if isinstance(ranked, list) and ranked and isinstance(ranked[0], str):
        ids = [s for s in ranked]

    # list[tuple/list]
    elif isinstance(ranked, list) and ranked and isinstance(ranked[0], (list, tuple)):
        ids = [str(item[0]) for item in ranked if item]

    # list[dict]
    elif isinstance(ranked, list) and ranked and isinstance(ranked[0], dict):
        for r in ranked:
            nid = _get_case_insensitive(
                r, "TrialID", "trial_id", "nct_id", "id", "nctId"
            )
            if nid is None:
                continue
            ids.append(str(nid))

    # fallback
    else:
        try:
            ids = [str(x) for x in ranked]  # type: ignore
        except Exception:
            ids = []

    # Normalize + keep only proper NCT IDs
    cleaned: List[str] = []
    for x in ids:
        x_u = x.strip().upper()
        if _NCT_RE.match(x_u):
            cleaned.append(x_u)
    return cleaned


def evaluate_ranking(
    predicted_ranked: Any,
    ground_truth_for_query: Dict[str, int],
    ks: Tuple[int, ...] = (5, 10, 20),
    gain_scheme: str = "linear",
) -> Dict[str, float]:
    """
    Compute metrics for a single patient/query:
      - nDCG@K with graded labels (0/1/2)
      - Precision@K with 'eligible-only' relevance (label >= 2)

    IMPORTANT: Unlabelled trials are REMOVED before computing metrics, per your requirement.
    """
    pred_ids_all = _normalize_ranked_input(predicted_ranked)

    # Keep only labelled trials (ignore unjudged)
    pred_ids = (
        [pid for pid in pred_ids_all if pid in ground_truth_for_query]
        if ground_truth_for_query
        else []
    )

    results: Dict[str, float] = {}
    for k in ks:
        results[f"ndcg@{k}"] = ndcg_at_k(
            pred_ids, ground_truth_for_query, k, gain_scheme=gain_scheme
        )
        results[f"precision@{k}"] = precision_at_k(
            pred_ids, ground_truth_for_query, k, eligible_threshold=2
        )
    return results


def evaluate_and_save_metrics(
    ranked_trials: Any,
    patient_qid: str,
    ground_truth_source: Union[str, Dict[str, Dict[str, int]]],
    output_folder: str,
    ks: Tuple[int, ...] = (5, 10, 20),
    gain_scheme: str = "linear",
) -> Dict[str, float]:
    """
    Evaluate 'ranked_trials' for a patient using either:
      - ground_truth_source: str path to TREC .tsv/.csv, OR
      - a preloaded dict { qid: { nct_id: score } }

    Saves JSON to {output_folder}/evaluation_metrics.json
    """
    # Load ground truth if a file path is provided
    if isinstance(ground_truth_source, str):
        ground_truth_map = load_ground_truth(ground_truth_source)
    else:
        ground_truth_map = ground_truth_source

    gt = ground_truth_map.get(patient_qid)
    if gt is None:
        metrics: Dict[str, float] = {"error": "no_ground_truth_for_patient"}  # type: ignore[assignment]
    else:
        metrics = evaluate_ranking(ranked_trials, gt, ks=ks, gain_scheme=gain_scheme)  # type: ignore[assignment]
        metrics["patient_id"] = patient_qid  # type: ignore[index]

    os.makedirs(output_folder, exist_ok=True)
    out_path = os.path.join(output_folder, "evaluation_metrics.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    return metrics


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser("Evaluate ranked trials against TREC ground truth")
    ap.add_argument("--ranked", required=True, help="Path to ranked_trials.json")
    ap.add_argument(
        "--ground-truth",
        required=True,
        help="Path to TREC TSV/CSV (query-id, corpus-id, score)",
    )
    ap.add_argument(
        "--patient", required=True, help="Query-id in the ground truth to evaluate"
    )
    ap.add_argument(
        "--out-dir", default=None, help="Output dir (default: folder of ranked)"
    )
    ap.add_argument(
        "--gain",
        default="linear",
        choices=["linear", "exp2"],
        help="DCG gain scheme (default: linear)",
    )
    args = ap.parse_args()

    # Lazy import to avoid package cycles
    try:
        from trialmatchai.utils.file_utils import read_json_file  # type: ignore
    except Exception:
        # Fallback path if run standalone
        from trialmatchai.utils.file_utils import read_json_file  # type: ignore

    ranked = read_json_file(args.ranked)
    out_dir = args.out_dir or os.path.dirname(args.ranked)
    res = evaluate_and_save_metrics(
        ranked_trials=ranked,
        patient_qid=args.patient,
        ground_truth_source=args.ground_truth,
        output_folder=out_dir,
        ks=(5, 10, 20),
        gain_scheme=args.gain,
    )
    print(json.dumps(res, indent=2))
