"""Tie-aware nDCG + precision (trec/metrics.py)."""

from itertools import permutations

from trialmatchai.trec.metrics import (
    condensed_ndcg,
    idcg_at_k,
    ndcg_at_k,
    precision_at_k,
)


def test_perfect_ranking_scores_one():
    ids = ["a", "b", "c"]
    score = {"a": 3.0, "b": 2.0, "c": 1.0}
    gain = {"a": 2, "b": 1, "c": 0}
    assert ndcg_at_k(ids, score, gain, 10) == 1.0


def test_all_tied_is_order_invariant():
    # Every doc shares one score -> McSherry-Najork averaging must make nDCG
    # identical across all orderings of the tie group.
    gain = {"a": 2, "b": 1, "c": 0}
    score = {k: 1.0 for k in gain}
    values = {round(ndcg_at_k(list(p), score, gain, 10), 9) for p in permutations(gain)}
    assert len(values) == 1


def test_tie_group_straddling_cutoff_is_order_invariant():
    # Two tied docs straddling the k=1 boundary share the averaged discount, so
    # swapping them cannot change nDCG@1.
    gain = {"a": 1, "b": 0}
    score = {"a": 0.5, "b": 0.5}
    assert ndcg_at_k(["a", "b"], score, gain, 1) == ndcg_at_k(["b", "a"], score, gain, 1)


def test_zero_gain_gives_zero_ndcg():
    assert ndcg_at_k(["a", "b"], {"a": 1.0, "b": 0.5}, {"a": 0, "b": 0}, 10) == 0.0


def test_idcg_uses_best_ordering():
    assert idcg_at_k([0, 2, 1], 10) == idcg_at_k([2, 1, 0], 10)


def test_precision_at_k_hard_cutoff():
    ordered = ["a", "x", "c", "y", "z"]
    assert precision_at_k(ordered, {"a", "c"}, 10) == 2 / 10  # denominator is k
    assert precision_at_k(ordered, {"a", "c"}, 2) == 1 / 2  # only 'a' in top-2


def test_condensed_ndcg_drops_unjudged_and_grades_gains():
    ranked = ["a", "u", "b", "c"]  # 'u' is unjudged
    score = {"a": 1.0, "u": 0.9, "b": 0.5, "c": 0.0}
    grade = {"a": 2, "b": 1, "c": 0}
    out = condensed_ndcg(ranked, score, grade, (5, 10))
    assert set(out) == {5, 10}
    assert out[10] == 1.0  # condensed order a,b,c is already ideal by grade
