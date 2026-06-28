"""Regression guard for the TREC reproduction.

Locks the reproduction-critical behavior — the tie-aware nDCG algorithm, the
corpus restriction, and the relevance thresholds — so the pipeline unification
(folding `trec` into a preset over the one e2e pipeline) cannot silently change a
benchmark number. If any of these change, that is a deliberate decision, not an
accident.
"""

import math

from trialmatchai.trec.metrics import ndcg_at_k
from trialmatchai.trec.qrels import corpus_ncts, relevant_ncts


def test_tie_aware_ndcg_matches_mcsherry_najork_closed_form():
    # Three docs tied at one score: each gets the MEAN discount over the ranks the
    # tie group occupies (McSherry-Najork). Recompute the reference independently.
    def d(r):
        return 1.0 / math.log2(r + 1)

    mean_disc = (d(1) + d(2) + d(3)) / 3
    dcg = (2 + 1 + 0) * mean_disc
    idcg = 2 * d(1) + 1 * d(2) + 0 * d(3)
    expected = dcg / idcg

    got = ndcg_at_k(
        ["a", "b", "c"], {"a": 1.0, "b": 1.0, "c": 1.0}, {"a": 2, "b": 1, "c": 0}, 10
    )
    assert abs(got - expected) < 1e-9
    assert round(got, 4) == 0.8100  # pinned: flags any drift in the averaging


def test_corpus_ncts_is_union_of_judged_trials():
    # The per-run index is restricted to this set (the qrels-judged pool).
    qrels = {
        "trec-1": {"NCT1": 2, "NCT2": 0},
        "trec-2": {"NCT2": 1, "NCT3": 2},
    }
    assert corpus_ncts(qrels) == {"NCT1", "NCT2", "NCT3"}


def test_relevant_ncts_threshold():
    qrels = {"q": {"NCT1": 2, "NCT2": 1, "NCT3": 0}}
    assert relevant_ncts(qrels, threshold=1) == {"q": {"NCT1", "NCT2"}}  # relevant
    assert relevant_ncts(qrels, threshold=2) == {"q": {"NCT1"}}  # eligible-only
