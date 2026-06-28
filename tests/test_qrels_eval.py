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
