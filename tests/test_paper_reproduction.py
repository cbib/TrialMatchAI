from __future__ import annotations

import json
import math
import subprocess
import sys
import zipfile

import pytest

from trialmatchai.trec import paper_reproduction as reproduction


def test_trec_package_keeps_runner_import_lazy():
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import trialmatchai.trec; "
            "assert 'trialmatchai.trec.runner' not in sys.modules",
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert probe.returncode == 0, probe.stderr


def test_paper_archive_extraction_rejects_traversal(tmp_path):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../outside.json", "{}")

    with pytest.raises(ValueError, match="unsafe path"):
        reproduction.extract_reproduction_files(archive, tmp_path / "work")
    assert not (tmp_path / "outside.json").exists()


def test_paper_archive_extraction_requires_recall_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(
        reproduction,
        "TRACKS",
        {"21": {"directory": "TREC21", "prefix": "trec-2021", "topics": 1}},
    )
    monkeypatch.setattr(reproduction, "RECALL_CUTOFFS", (10,))
    archive = tmp_path / "incomplete.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("matching_results/TREC21/average_metrics.json", "{}")
        bundle.writestr(
            "matching_results/TREC21/trec-20211/evaluation_metrics.json", "{}"
        )
        bundle.writestr("matching_results/TREC21/trec-20211/ranked_trials.json", "[]")
        bundle.writestr("matching_results/TREC21/trec-20211/nct_ids.txt", "NCT1\n")

    with pytest.raises(ValueError, match="nct_ids_10.txt"):
        reproduction.extract_reproduction_files(archive, tmp_path / "work")


def test_incomplete_extraction_cache_is_rebuilt(tmp_path, monkeypatch):
    monkeypatch.setattr(
        reproduction,
        "TRACKS",
        {"21": {"directory": "TREC21", "prefix": "trec-2021", "topics": 1}},
    )
    monkeypatch.setattr(reproduction, "RECALL_CUTOFFS", (10,))
    monkeypatch.setattr(reproduction, "PAPER_RESULTS_SHA256", "test-archive")
    work = tmp_path / "work"
    cached_topic = work / "matching_results" / "TREC21" / "trec-20211"
    cached_topic.mkdir(parents=True)
    (work / "matching_results" / ".paper-results.json").write_text(
        json.dumps({"archive_sha256": "test-archive"}), encoding="utf-8"
    )
    for name in ("evaluation_metrics.json", "ranked_trials.json", "nct_ids.txt"):
        (cached_topic / name).write_text("{}", encoding="utf-8")
    (cached_topic.parent / "average_metrics.json").write_text("{}", encoding="utf-8")

    archive = tmp_path / "complete.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("matching_results/TREC21/average_metrics.json", "{}")
        for name, content in (
            ("evaluation_metrics.json", "{}"),
            ("ranked_trials.json", "[]"),
            ("nct_ids.txt", "NCT1\n"),
            ("nct_ids_10.txt", "NCT1\n"),
        ):
            bundle.writestr(f"matching_results/TREC21/trec-20211/{name}", content)

    extracted = reproduction.extract_reproduction_files(archive, work)

    assert (extracted / "TREC21" / "trec-20211" / "nct_ids_10.txt").is_file()


def test_paper_ranking_metric_drops_unjudged_and_preserves_archived_order():
    metrics = reproduction.paper_ranking_metrics(
        ["UNJUDGED", "NCT_GRADE_1", "NCT_GRADE_2"],
        {"NCT_GRADE_1": 1, "NCT_GRADE_2": 2},
    )

    dcg = 1 / math.log2(2) + 2 / math.log2(3)
    ideal = 2 / math.log2(2) + 1 / math.log2(3)
    assert metrics["ndcg@5"] == pytest.approx(dcg / ideal)
    assert metrics["p@5"] == pytest.approx(3 / (2 * 5))


def test_audit_track_separates_stored_metrics_from_ranking_recalculation(
    tmp_path, monkeypatch
):
    monkeypatch.setitem(
        reproduction.TRACKS,
        "21",
        {"directory": "TREC21", "prefix": "trec-2021", "topics": 1},
    )
    qrels = tmp_path / "qrels_21.txt"
    qrels.write_text("1 0 NCT1 2\n1 0 NCT2 1\n", encoding="utf-8")
    topic = tmp_path / "results" / "TREC21" / "trec-20211"
    topic.mkdir(parents=True)

    stored = reproduction.paper_ranking_metrics(["NCT1", "NCT2"], {"NCT1": 2, "NCT2": 1})
    (topic / "evaluation_metrics.json").write_text(json.dumps(stored), encoding="utf-8")
    (topic.parent / "average_metrics.json").write_text(json.dumps(stored), encoding="utf-8")
    # The archived ranking has the reverse order, so recomputation must expose a mismatch.
    (topic / "ranked_trials.json").write_text(
        json.dumps(
            [
                {"TrialID": "UNJUDGED", "Score": 2.0},
                {"TrialID": "NCT2", "Score": 1.0},
                {"TrialID": "NCT1", "Score": 1.0},
            ]
        ),
        encoding="utf-8",
    )
    (topic / "nct_ids.txt").write_text("UNJUDGED\nNCT2\nNCT1\n", encoding="utf-8")
    for cutoff in reproduction.RECALL_CUTOFFS:
        (topic / f"nct_ids_{cutoff}.txt").write_text("NCT2\nNCT1\n", encoding="utf-8")

    result = reproduction.audit_track("21", tmp_path / "results", qrels)

    assert result["checks"]["archived_summary_matches_stored_topics"] is True
    assert result["checks"]["rankings_match_stored_topic_metrics"] is False
    assert result["checks"]["ranking_mismatch_count"] == 1
    assert result["paper_retrieval_recall"]["recall@10"]["mean"] == 1.0
    # Current evaluation tie-averages the equal scores; the paper method preserves file order.
    assert result["current_evaluator"]["ndcg@5"]["mean"] != pytest.approx(
        result["recalculated_from_rankings"]["ndcg@5"]["mean"]
    )
    assert (
        result["current_evaluator_by_unjudged_policy"]["include_as_zero"]["ndcg@5"][
            "mean"
        ]
        < result["current_evaluator_by_unjudged_policy"]["exclude"]["ndcg@5"]["mean"]
    )
