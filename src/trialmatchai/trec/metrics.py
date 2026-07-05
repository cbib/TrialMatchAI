"""Ranking-quality metrics for TREC evaluation (complementing recall@k in ``qrels``).

nDCG here is tie-aware (McSherry & Najork, 2008): tied scores each get the mean
positional discount over the ranks the tie group spans, i.e. the expected nDCG
over all orderings of the tie, so it is invariant to arbitrary tie-breaking. It
is also condensed: computed over labeled-and-retrieved trials only, decoupling
ranking quality from recall. Gain is linear (gain = relevance grade), matching
trec_eval's default.
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

    Each tie group spanning 1-indexed ranks [a..b] gives every member the mean
    discount over ranks a..min(b, k).
    """
    # Re-sort by descending score so tie groups are contiguous regardless of caller order.
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
    *,
    ideal_gains: Sequence[float] | None = None,
) -> float:
    """Tie-aware nDCG@k. ``ordered_ids`` should be the condensed (labeled) list.

    ``ideal_gains`` chooses the IDCG basis. When ``None`` (default) the ideal is built from
    the gains of ``ordered_ids`` — i.e. the judged-AND-ranked trials, making nDCG
    recall-independent (it only grades how well the ranked judged trials are ordered). Pass
    the gains of the FULL judged pool to get the recall-aware ``trec_eval``-style nDCG, where
    a relevant trial that was never ranked stays in the ideal and lowers the score.
    """
    if k <= 0 or not ordered_ids:
        return 0.0
    gains = ideal_gains if ideal_gains is not None else [gain_of.get(d, 0.0) for d in ordered_ids]
    idcg = idcg_at_k(gains, k)
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


def graded_precision_at_k(
    ordered_ids: Sequence[str], grade_of: Mapping[str, int], k: int, *, g_max: int = 2
) -> float:
    """Graded P@k: each top-k trial contributes ``min(grade, g_max) / g_max``.

    Unlike binary P@k, this rewards ranking genuinely eligible (grade 2) trials
    above merely on-topic excluded (grade 1) ones.
    """
    if k <= 0 or g_max <= 0:
        return 0.0
    topk = ordered_ids[:k]
    if not topk:
        return 0.0
    return sum(min(grade_of.get(d, 0), g_max) for d in topk) / float(k * g_max)


def condensed_ndcg(
    ranked_ids: Sequence[str],
    score_of: Mapping[str, float],
    grade_of: Mapping[str, int],
    cutoffs: Sequence[int],
    *,
    full_ideal: bool = False,
) -> Dict[int, float]:
    """Tie-aware nDCG@k per cutoff. The DCG numerator is always condensed to the judged
    trials (those in ``grade_of``), so unjudged trials never count. ``full_ideal`` selects the
    IDCG basis: ``False`` (default) normalizes by the ideal over judged-AND-ranked trials
    (recall-independent); ``True`` normalizes by the ideal over the FULL judged pool
    (recall-aware, ``trec_eval``-style) so unretrieved relevant trials lower the score.
    """
    condensed = [nid for nid in ranked_ids if nid in grade_of]
    ideal = [float(g) for g in grade_of.values()] if full_ideal else None
    return {k: ndcg_at_k(condensed, score_of, grade_of, k, ideal_gains=ideal) for k in cutoffs}
