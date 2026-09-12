"""Synthetic audit probes for revision 508db33; no models, network, or real patients.

Run from the repository root:
    .venv/bin/python docs/audit_2026_09_12_reproduce.py

These observations deliberately describe current defects, rather than asserting
desired behavior. They are review evidence, not additions to the regression suite.
All temporary patient and trial artifacts live under a TemporaryDirectory.
"""

from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RESULTS: dict[str, dict] = {}


def probe(name):
    def decorate(fn):
        try:
            RESULTS[name] = {"status": "observed", **fn()}
        except Exception as exc:
            RESULTS[name] = {"status": "probe_error", "error": repr(exc)}
        print(json.dumps({name: RESULTS[name]}, sort_keys=True), flush=True)
        return fn
    return decorate


@probe("A01_remote_adapter_paths")
def adapters():
    from trialmatchai.config.config_loader import load_config

    config = load_config(str(ROOT / "src/trialmatchai/config/config.json"))
    paths = {key: config["model"][key] for key in ("cot_adapter_path", "reranker_adapter_path")}
    return {"normalized": paths, "all_absolute": all(Path(p).is_absolute() for p in paths.values()),
            "any_exists": any(Path(p).exists() for p in paths.values())}


@probe("A02_exclusion_threshold")
def exclusions():
    from trialmatchai.matching.retrieval.criteria_retrieval import SecondStageRetriever

    class PerfectReranker:
        def rank_pairs(self, pairs):
            return [1.0] * len(pairs)

    retriever = SecondStageRetriever(None, PerfectReranker(), None)
    rows = [{"query": "synthetic cancer", "_source": {
        "criterion": "Excluded if synthetic cancer", "eligibility_type": "exclusion",
        "criteria_id": "c1", "nct_id": "NCT00000001"}}]
    scored = retriever.rerank_criteria(rows)
    return {"raw_score": 1.0, "weighted_score": scored[0]["llm_score"],
            "retained_trials": retriever.aggregate_to_trials(scored)}


@probe("A03_cross_patient_fhir")
def cross_patient():
    from trialmatchai.interop.importers.fhir import import_fhir

    bundle = {"resourceType": "Bundle", "type": "collection", "entry": [
        {"resource": {"resourceType": "Patient", "id": "synthetic-A"}},
        {"resource": {"resourceType": "Condition", "id": "condition-B",
                      "subject": {"reference": "Patient/synthetic-B"},
                      "code": {"text": "Synthetic disease belonging to B"}}}]}
    with tempfile.TemporaryDirectory(prefix="tmai-audit-") as tmp:
        path = Path(tmp) / "bundle.json"
        path.write_text(json.dumps(bundle))
        profiles = import_fhir(path, strict=True)
    return {"patient": profiles[0].patient_id, "strict": True,
            "conditions": [f.label for f in profiles[0].conditions]}


@probe("A04_stale_eligibility_worklist")
def stale_eligibility():
    from trialmatchai.matching.eligibility_base import BaseTrialProcessor

    class Recorder(BaseTrialProcessor):
        def __init__(self):
            self.calls = []

        def _process_batch(self, batch, output_folder):
            self.calls.extend(batch)

    with tempfile.TemporaryDirectory(prefix="tmai-audit-") as tmp:
        root = Path(tmp)
        trials, outputs = root / "trials", root / "outputs"
        trials.mkdir()
        outputs.mkdir()
        (trials / "NCT00000001.json").write_text(json.dumps({"eligibility_criteria": "NEW criteria"}))
        (outputs / "NCT00000001.json").write_text(json.dumps({"Final Decision": "Eligible"}))
        processor = Recorder()
        processor.process_trials(["NCT00000001"], str(trials), str(outputs), ["CHANGED patient facts"])
        return {"new_batches": len(processor.calls), "reused": json.loads((outputs / "NCT00000001.json").read_text())}


@probe("A05_incomplete_match_signature")
def signatures():
    from trialmatchai.orchestration import _match_signature

    base = {"model": {"base_model": "base", "cot_adapter_path": "adapter-A"},
            "search": {"mode": "hybrid", "first_level": {"max_trials": 100}},
            "constraints": {"score_weight": 0.25}}
    changed = copy.deepcopy(base)
    changed["model"]["cot_adapter_path"] = "adapter-B"
    changed["search"]["first_level"]["max_trials"] = 900
    changed["constraints"]["score_weight"] = 0.9
    return {"signature_unchanged": _match_signature(base) == _match_signature(changed)}


@probe("A06_invalid_eligibility_accepted")
def invalid_eligibility():
    from trialmatchai.matching.eligibility_base import BaseTrialProcessor, _is_error_output

    with tempfile.TemporaryDirectory(prefix="tmai-audit-") as tmp:
        BaseTrialProcessor()._save_outputs("NCT00000001", '{"unrelated": true}', tmp)
        path = Path(tmp) / "NCT00000001.json"
        return {"stored": json.loads(path.read_text()), "scheduled_for_retry": _is_error_output(str(path))}


@probe("A07_failed_inclusion_score")
def inclusion_score():
    from trialmatchai.matching.trial_ranker import score_trial

    def score(inclusions, exclusions=()):
        return score_trial({"Inclusion_Criteria_Evaluation": [{"Classification": c} for c in inclusions],
                            "Exclusion_Criteria_Evaluation": [{"Classification": c} for c in exclusions]})
    return {"mandatory_inclusion_failed": score(["Met"] * 9 + ["Not Met"]),
            "insufficient_information": score(["Unclear"]),
            "all_exclusions_unknown": score(["Met"], ["Unclear"] * 10),
            "exclusion_violation": score(["Met"], ["Violated"]),
            "generation_error": score_trial({"error": "processing_failed"})}


@probe("A08_missing_fact_becomes_absence")
def absence():
    from trialmatchai.constraints.evaluation import evaluate_constraint_set
    from trialmatchai.constraints.models import Constraint, ConstraintSet, PatientConstraintContext

    context = PatientConstraintContext(patient_id="synthetic")
    out = {}
    for polarity, comparator in [("inclusion", "absent"), ("exclusion", "present")]:
        criteria = ConstraintSet(nct_id="NCT00000001", criteria_id="c1", polarity=polarity,
                                 source_text="Synthetic criterion", constraints=[Constraint(
                                     kind="condition", label="diabetes", comparator=comparator)])
        result = evaluate_constraint_set(criteria, context)
        out[f"{polarity}_{comparator}"] = {"status": result.evaluations[0].status,
                                           "signal": result.constraint_signal,
                                           "evidence": result.evaluations[0].patient_evidence}
    return out


@probe("A09_gene_mention_becomes_mutation")
def biomarker():
    from trialmatchai.constraints.evaluation import evaluate_constraint_set
    from trialmatchai.constraints.models import Constraint, ConstraintSet, PatientConstraintContext, PatientConstraintFact

    criteria = ConstraintSet(nct_id="NCT00000001", criteria_id="c1", polarity="inclusion",
                             source_text="EGFR mutation required", constraints=[Constraint(
                                 kind="biomarker", label="EGFR", comparator="mutated")])
    context = PatientConstraintContext(patient_id="synthetic", facts=[PatientConstraintFact(
        kind="biomarker", label="EGFR", evidence_text="EGFR testing ordered; result pending")])
    result = evaluate_constraint_set(criteria, context)
    return {"clinical_status": result.evaluations[0].status, "signal": result.constraint_signal}


@probe("A10_empty_candidate_scope")
def empty_scope():
    from trialmatchai.search.lancedb_backend import InMemorySearchBackend

    backend = InMemorySearchBackend(criteria=[{"nct_id": "NCT00000001", "criteria_id": "c1",
                                               "criterion": "synthetic cancer", "eligibility_type": "inclusion"}])
    hits = backend.search_criteria(query="synthetic cancer", nct_ids=[], search_mode="bm25")
    return {"requested_nct_ids": [], "returned_nct_ids": [h["_source"]["nct_id"] for h in hits]}


@probe("A11_vector_dimension_mismatch")
def mismatched_vectors():
    from trialmatchai.search.lancedb_backend import _cosine

    return {"left": [1.0], "right": [1.0, 99.0], "cosine": _cosine([1.0], [1.0, 99.0])}


@probe("A12_rejected_concept_indexed")
def rejected_concepts():
    from trialmatchai.search.lancedb_backend import _flatten_entities

    texts, synonyms = _flatten_entities([{"text": "synthetic mention", "linker_status": "rejected",
                                         "concept_candidates": [{"concept_name": "unrelated disease", "score": 1.0}]}])
    return {"text": texts, "indexed_synonyms": synonyms}


@probe("A13_missing_topics_omitted")
def missing_topics():
    from trialmatchai.trec.qrels import evaluate

    with tempfile.TemporaryDirectory(prefix="tmai-audit-") as tmp:
        root = Path(tmp)
        p = root / "synthetic-A"
        p.mkdir()
        (p / "ranked_trials.json").write_text(json.dumps({"RankedTrials": [{"TrialID": "NCT00000001", "Score": 1}]}))
        (root / "synthetic-C").mkdir()
        qrels = {pid: {"NCT00000001": 2} for pid in ["synthetic-A", "synthetic-B", "synthetic-C"]}
        result = evaluate(qrels, root)
    return {"expected_topics": 3, "scored_topics": result["num_queries_scored"],
            "ranked_topics": result["num_queries_ranked"], "mean_ndcg10": result["mean"]["ndcg@10"]}


@probe("A14_condensed_ndcg_one_hit")
def one_hit():
    from trialmatchai.trec.metrics import condensed_ndcg

    judgments = {f"NCT{i:08d}": 2 for i in range(1, 11)}
    ranked, scores = ["NCT00000001"], {"NCT00000001": 1.0}
    return {"eligible_found": 1, "eligible_total": 10,
            "ndcg10": condensed_ndcg(ranked, scores, judgments, [10])[10],
            "ndcg_full10": condensed_ndcg(ranked, scores, judgments, [10], full_ideal=True)[10]}


@probe("A15_failure_report_state")
def failure_report():
    from trialmatchai.interop.exporters.html_report import build_report_model

    model = build_report_model(patient_summary={"patient_id": "synthetic"},
                               ranked=[{"TrialID": "NCT00000001", "Score": -1}],
                               eligibility_by_id={"NCT00000001": {"error": "processing_failed"}},
                               meta_by_id={}, generated_at="synthetic")
    return {"reasoning_available": model["trials"][0]["reasoning_available"],
            "final_decision": model["trials"][0]["final_decision"]}


@probe("A16_fhir_export_required_fields")
def fhir_export():
    from trialmatchai.interop.exporters.fhir import profile_to_fhir_bundle
    from trialmatchai.interop.models import PatientProfile, Provenance
    from trialmatchai.interop.utils import make_fact

    profile = PatientProfile(patient_id="synthetic", medications=[make_fact(
        category="medication", label="synthetic drug", provenance=Provenance(source_format="text"))])
    bundle = profile_to_fhir_bundle(profile)
    medication = next(e["resource"] for e in bundle["entry"] if e["resource"]["resourceType"] == "MedicationStatement")
    return {"resource_keys": sorted(medication), "has_required_status": "status" in medication,
            "has_required_medication": any(k in medication for k in ["medicationCodeableConcept", "medicationReference"])}


@probe("A17_empty_prepared_criteria_stale")
def empty_criteria():
    from trialmatchai.registry.preparation import write_prepared_criteria

    with tempfile.TemporaryDirectory(prefix="tmai-audit-") as tmp:
        write_prepared_criteria([{"nct_id": "NCT00000001", "criteria_id": "old", "criterion": "Old criterion"}], tmp)
        write_prepared_criteria([], tmp)
        return {"old_criterion_remains": (Path(tmp) / "NCT00000001/old.json").exists()}


@probe("A18_unvalidated_canonical_patient_id")
def patient_id():
    from trialmatchai.interop.models import PatientProfile

    profile = PatientProfile(patient_id="../synthetic-escape")
    with tempfile.TemporaryDirectory(prefix="tmai-audit-") as tmp:
        output = Path(tmp) / "outputs"
        target = (output / profile.patient_id).resolve()
        return {"id_accepted": profile.patient_id, "target_outside_output_root": not target.is_relative_to(output.resolve())}


if __name__ == "__main__":
    output = Path(__file__).with_name("audit_2026_09_12_observations.json")
    output.write_text(json.dumps(RESULTS, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {output}")
    sys.exit(1 if any(v["status"] == "probe_error" for v in RESULTS.values()) else 0)
