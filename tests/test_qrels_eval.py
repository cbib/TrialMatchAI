"""qrels parsing + evaluation (trec/qrels.py)."""

import json

import pytest

from trialmatchai.trec.qrels import evaluate, parse_qrels, recall_at_k


def test_parse_qrels_prefixes_and_skips_malformed(tmp_path):
    p = tmp_path / "q.txt"
    p.write_text("1 0 NCT001 2\n1 0 NCT002 1\nbad line\n2 0 NCT003 0\n")
    assert parse_qrels(p, "trec-") == {
        "trec-1": {"NCT001": 2, "NCT002": 1},
        "trec-2": {"NCT003": 0},
    }


def test_parse_qrels_empty_raises(tmp_path):
    p = tmp_path / "empty.txt"
    p.write_text("\n# nothing parseable\n")
    with pytest.raises(ValueError):
        parse_qrels(p, "trec-")


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], {"a", "c", "z"}, 3) == 2 / 3
    assert recall_at_k(["a"], set(), 3) is None  # no relevant -> undefined


def test_evaluate_end_to_end(tmp_path):
    q = "trec-1"
    pdir = tmp_path / q
    pdir.mkdir()
    (pdir / "ranked_trials.json").write_text(
        json.dumps(
            {
                "RankedTrials": [
                    {"TrialID": "NCT1", "Score": 1.0},
                    {"TrialID": "NCT2", "Score": 0.5},
                    {"TrialID": "NCT3", "Score": 0.0},
                ]
            }
        )
    )
    (pdir / "nct_ids.txt").write_text("NCT1\nNCT2\nNCT3\n")
    qrels = {q: {"NCT1": 2, "NCT2": 1, "NCT3": 0}}
    res = evaluate(qrels, tmp_path, cutoffs=(10,))
    assert res["num_queries_scored"] == 1
    mean = res["mean"]
    assert mean["recall@10"] == 1.0  # both relevant retrieved
    assert mean["ndcg@10"] == 1.0  # ranking is the ideal grade order
    assert mean["P@10(rel>=1)"] == 2 / 10


def test_evaluate_reports_both_ndcg_ideal_bases(tmp_path):
    """ndcg@k uses the ideal over judged-AND-ranked trials (recall-independent);
    ndcg_full@k uses the ideal over the FULL judged pool (recall-aware). Here NCT2 is
    judged-eligible but never ranked, so ndcg_full must fall below the perfect ndcg."""
    import math

    q = "trec-1"
    pdir = tmp_path / q
    pdir.mkdir()
    (pdir / "ranked_trials.json").write_text(
        json.dumps({"RankedTrials": [{"TrialID": "NCT1", "Score": 1.0}]})  # only NCT1 ranked
    )
    (pdir / "nct_ids.txt").write_text("NCT1\nNCT2\n")  # both retrieved
    qrels = {q: {"NCT1": 2, "NCT2": 2}}  # NCT2 judged-eligible but absent from the ranking
    mean = evaluate(qrels, tmp_path, cutoffs=(10,))["mean"]
    # Condensed ideal = just NCT1 -> NCT1 ranked #1 is perfect.
    assert mean["ndcg@10"] == pytest.approx(1.0)
    # Full ideal = [NCT1, NCT2] both eligible; DCG only credits NCT1 at rank 1.
    disc1, disc2 = 1 / math.log2(2), 1 / math.log2(3)
    assert mean["ndcg_full@10"] == pytest.approx(disc1 / (disc1 + disc2))
    assert mean["ndcg_full@10"] < mean["ndcg@10"]  # recall-aware penalizes the missing relevant


def test_evaluate_precision_is_condensed_to_judged_pool(tmp_path):
    """Unjudged trials are dropped before the P@10 cutoff (condensed), matching nDCG. Ten
    unjudged trials sit ahead of the two judged relevant ones: raw P@10 would see only the
    unjudged block (0.0); condensed skips them so both relevant trials count."""
    q = "trec-1"
    pdir = tmp_path / q
    pdir.mkdir()
    ranked = [{"TrialID": f"NCTX{i}", "Score": 2.0 - i * 0.01} for i in range(10)]
    ranked += [{"TrialID": "NCT1", "Score": 0.5}, {"TrialID": "NCT2", "Score": 0.4}]
    (pdir / "ranked_trials.json").write_text(json.dumps({"RankedTrials": ranked}))
    (pdir / "nct_ids.txt").write_text("\n".join(r["TrialID"] for r in ranked) + "\n")
    qrels = {q: {"NCT1": 2, "NCT2": 1}}  # the NCTX* trials are unjudged
    mean = evaluate(qrels, tmp_path, cutoffs=(10,))["mean"]
    # Condensed list is [NCT1, NCT2]; both relevant over the k=10 cutoff -> 2/10.
    assert mean["P@10(rel>=1)"] == pytest.approx(2 / 10)  # raw would be 0/10
    assert mean["P@10(eligible)"] == pytest.approx(1 / 10)  # only NCT1 is grade 2
    assert mean["graded_P@10"] == pytest.approx((2 + 1) / (10 * 2))  # raw would be 0
