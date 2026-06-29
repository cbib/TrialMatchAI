"""Ranking-quality metrics for TREC evaluation.

These complement recall@k (the retrieval-side metric in ``qrels``). nDCG here is:

  * **tie-aware** (McSherry & Najork, 2008): trials sharing the same ranking
    score form a tie group, and each member is given the AVERAGE positional
    discount over the ranks the group spans (truncated at k). The result is the
    EXPECTED nDCG over all random orderings of the tied trials, so it is
    invariant to arbitrary tie-breaking — it rewards only genuinely ordering a
    more-relevant trial above a less-relevant one.
  * **condensed**: computed over the labeled-and-retrieved trials only, with the
    IDCG normalized to that same set. It measures the quality of the final
    ranking of the trials the model actually evaluated, decoupled from recall.

Gain is linear (gain = relevance grade), matching trec_eval's default and the
legacy evaluation.
"""

from __future__ import annotations

import math
from typing import Dict, Mapping, Sequence, Set


def _discount(rank: int) -> float:
    """Log2 positional discount for a 1-indexed rank: 1 / log2(rank + 1)."""
    return 1.0 / math.log2(rank + 1)


def tie_aware_dcg_at_k(
    ordered_ids: Sequence[str],
    score_of: Mapping[str, float],
    gain_of: Mapping[str, float],
    k: int,
) -> float:
    """Expected DCG@k over random orderings of tied scores (McSherry-Najork).

    ``ordered_ids`` must already be sorted by descending ranking score, so equal
    scores are contiguous. Each tie group spanning 1-indexed ranks [a..b] gives
    every member the mean discount over ranks a..min(b, k).
    """
    # Enforce the by-descending-score precondition so tie groups are contiguous
    # regardless of how the caller ordered the list (the metric is tie-order
    # invariant, so a stable re-sort cannot change a correct result).
    ordered_ids = sorted(ordered_ids, key=lambda d: score_of.get(d, 0.0), reverse=True)
    n = len(ordered_ids)
    total = 0.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and score_of[ordered_ids[j + 1]] == score_of[ordered_ids[i]]:
            j += 1
        a, b = i + 1, j + 1  # 1-indexed ranks spanned by this tie group
        hi = min(b, k)
        if hi >= a:
            avg_discount = sum(_discount(r) for r in range(a, hi + 1)) / (b - a + 1)
            for d in ordered_ids[i : j + 1]:
                total += float(gain_of.get(d, 0.0)) * avg_discount
        i = j + 1
    return total


def idcg_at_k(gains: Sequence[float], k: int) -> float:
    """Ideal DCG@k — gains sorted descending (tie order is irrelevant: equal gains)."""
    ideal = sorted((float(g) for g in gains), reverse=True)[:k]
    return sum(g * _discount(r + 1) for r, g in enumerate(ideal))


def ndcg_at_k(
    ordered_ids: Sequence[str],
    score_of: Mapping[str, float],
    gain_of: Mapping[str, float],
    k: int,
) -> float:
    """Tie-aware nDCG@k. ``ordered_ids`` should be the condensed (labeled) list."""
    if k <= 0 or not ordered_ids:
        return 0.0
    idcg = idcg_at_k([gain_of.get(d, 0.0) for d in ordered_ids], k)
    if idcg <= 0:
        return 0.0
    return tie_aware_dcg_at_k(ordered_ids, score_of, gain_of, k) / idcg


def precision_at_k(ordered_ids: Sequence[str], relevant: Set[str], k: int) -> float:
    """Standard binary P@k over the final ranked list (hard cutoff k)."""
    if k <= 0:
        return 0.0
    topk = ordered_ids[:k]
    if not topk:
        return 0.0
    return sum(1 for d in topk if d in relevant) / float(k)


def condensed_ndcg(
    ranked_ids: Sequence[str],
    score_of: Mapping[str, float],
    grade_of: Mapping[str, int],
    cutoffs: Sequence[int],
) -> Dict[int, float]:
    """Tie-aware nDCG@k for each cutoff, condensed to labeled-and-retrieved trials.

    ``ranked_ids`` is the final ranking order; ``grade_of`` is the qrels grade for
    judged trials. Only trials present in ``grade_of`` are kept (condensed).
    """
    condensed = [nid for nid in ranked_ids if nid in grade_of]
    return {k: ndcg_at_k(condensed, score_of, grade_of, k) for k in cutoffs}
