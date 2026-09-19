from __future__ import annotations

import json

from trialmatchai.cli.trec_evaluate import evaluate_completed_runs


def test_completed_run_comparison_reports_both_unjudged_policies(tmp_path):
    qrels = tmp_path / "qrels.txt"
    qrels.write_text("1 0 NCT1 2\n", encoding="utf-8")
    results = tmp_path / "experiment" / "results_trec21"
    topic = results / "trec-20211"
    topic.mkdir(parents=True)
    ranked = [
        {"TrialID": "UNJUDGED", "Score": 1.0},
        {"TrialID": "NCT1", "Score": 0.5},
    ]
    (topic / "ranked_trials.json").write_text(json.dumps(ranked), encoding="utf-8")
    (topic / "nct_ids.txt").write_text("UNJUDGED\nNCT1\n", encoding="utf-8")

    report = evaluate_completed_runs(
        track="21", results_dirs=[results], qrels_path=qrels
    )

    run = report["runs"][0]
    assert run["name"] == "experiment"
    assert len(run["evaluation_input_sha256"]) == 64
    assert len(report["qrels"]["sha256"]) == 64
    assert run["policies"]["exclude"]["mean"]["ndcg@10"] == 1.0
    assert run["policies"]["include_as_zero"]["mean"]["ndcg@10"] < 1.0
    assert "per_query" not in run["policies"]["exclude"]
