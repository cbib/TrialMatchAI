"""Unit tests for the `link` stage: relinking persisted entities in place."""

from __future__ import annotations

import json

from trialmatchai.entities.linker import (
    ConceptLinker,
    InMemoryConceptStore,
    gate_status,
    lexical_reranker,
)
from trialmatchai.entities.schemas import load_entity_schemas
from trialmatchai.entities.types import ConceptCandidate, EntityAnnotation
from trialmatchai.linking import _link_entities, _link_trial_dir


def _linker() -> ConceptLinker:
    store = InMemoryConceptStore(
        [
            {
                "concept_id": "DOID:9351",
                "vocabulary_id": "DOID",
                "concept_code": "DOID:9351",
                "concept_name": "diabetes mellitus",
                "domain_id": "Disease",
                "concept_class_id": "",
                "standard_concept": "",
                "synonyms": ["diabetes"],
            }
        ]
    )
    # Low thresholds so the in-memory lexical score deterministically accepts.
    return ConceptLinker(
        store, load_entity_schemas(None), accept_threshold=0.5, reject_threshold=0.1
    )


def _entity(status: str = "concept_store_unavailable") -> dict:
    return {
        "entity_group": "disease",
        "text": "diabetes",
        "start": 0,
        "end": 8,
        "score": 0.9,
        "normalized_id": ["CUI-less"],
        "synonyms": [],
        "concept_candidates": [],
        "linker_score": None,
        "linker_status": status,
        "entity": "diabetes",
        "class": "disease",
    }


def test_link_entities_populates_unlinked():
    out, changed = _link_entities([_entity()], _linker(), {}, {}, force=False)
    assert changed
    assert out[0]["linker_status"] == "accepted"
    assert out[0]["normalized_id"] != ["CUI-less"]
    assert out[0]["concept_candidates"]  # at least one candidate recorded


def test_link_entities_idempotent_skips_already_linked():
    out, changed = _link_entities([_entity(status="accepted")], _linker(), {}, {}, force=False)
    assert not changed
    assert out[0]["linker_status"] == "accepted"


def test_link_entities_force_relinks():
    _, changed = _link_entities([_entity(status="accepted")], _linker(), {}, {}, force=True)
    assert changed


def test_link_trial_dir_writes_back_and_preserves_other_fields(tmp_path):
    trial_dir = tmp_path / "NCT1"
    trial_dir.mkdir()
    row = {
        "criteria_id": "c1",
        "nct_id": "NCT1",
        "criterion": "has diabetes",
        "entities": [_entity()],
        "criterion_vector": [0.1, 0.2, 0.3],
        "constraints": "{}",
    }
    (trial_dir / "c1.json").write_text(json.dumps(row))

    assert _link_trial_dir(trial_dir, _linker(), {}, {}, force=False)

    written = json.loads((trial_dir / "c1.json").read_text())
    assert written["entities"][0]["linker_status"] == "accepted"
    assert written["criterion_vector"] == [0.1, 0.2, 0.3]  # preserved
    assert written["constraints"] == "{}"  # preserved


# --- H3: accept gate on absolute lexical quality, not the RRF top score ---


def test_gate_status_bands():
    def g(quality, runner_up=0.0):
        return gate_status(
            quality, runner_up=runner_up, accept_threshold=0.7, reject_threshold=0.5, margin=0.05
        )

    assert g(0.95) == "accepted"
    assert g(0.40) == "rejected"
    assert g(0.60) == "ambiguous"
    assert g(0.95, 0.93) == "ambiguous"  # near-tied strong runner-up -> abstain
    assert g(0.95, 0.50) == "accepted"  # clear winner


class _RRFLikeStore:
    """Mimics LanceDBConceptStore: returns a candidate whose RRF-normalized score is 1.0
    regardless of literal match quality -- exactly what made the old gate inert."""

    def __init__(self, candidate: ConceptCandidate):
        self._candidate = candidate

    def search(self, query, **_kwargs):
        from dataclasses import replace

        return [replace(self._candidate, score=1.0)]


def _disease_linker(store) -> ConceptLinker:
    return ConceptLinker(
        store, load_entity_schemas(None), accept_threshold=0.7, reject_threshold=0.5
    )


def test_gate_abstains_when_rrf_top_is_a_poor_literal_match():
    poor = ConceptCandidate(
        concept_id="DOID:1", vocabulary_id="DOID", concept_code="1",
        concept_name="pain management", domain_id="Disease",
    )
    out = _disease_linker(_RRFLikeStore(poor)).link_annotation(
        EntityAnnotation(entity_group="disease", text="chronic pain", start=0, end=12, score=0.9)
    )
    # OLD: RRF score 1.0 >= 0.8 -> ACCEPTED (the H3 bug). NEW: lexical ~0.4 -> abstain.
    assert out.linker_status in ("rejected", "ambiguous")
    assert out.normalized_id == ("CUI-less",)
    assert out.concept_candidates  # candidates still recorded for observability


def test_gate_accepts_strong_literal_match():
    good = ConceptCandidate(
        concept_id="DOID:9351", vocabulary_id="DOID", concept_code="9351",
        concept_name="diabetes mellitus", synonyms=("diabetes",), domain_id="Disease",
    )
    out = _disease_linker(_RRFLikeStore(good)).link_annotation(
        EntityAnnotation(entity_group="disease", text="diabetes", start=0, end=8, score=0.9)
    )
    assert out.linker_status == "accepted"
    assert out.normalized_id == ("DOID:9351",)
    assert out.linker_score == 1.0  # exact synonym match, not the RRF 1.0


def test_lexical_reranker_promotes_best_literal_match():
    cands = [
        ConceptCandidate(concept_id="1", vocabulary_id="DOID", concept_code="1", concept_name="pain disorder"),
        ConceptCandidate(concept_id="2", vocabulary_id="DOID", concept_code="2", concept_name="chronic pain"),
    ]
    assert lexical_reranker("chronic pain", cands)[0].concept_name == "chronic pain"


# --- Phase 2: NIL-aware evaluation + threshold sweep ---


def test_linking_metrics_nil_aware():
    from trialmatchai.entities.linker_eval import linking_metrics

    gold = ["A", "B", "C", None, "D"]
    pred = ["A", "B", "X", None, None]
    m = linking_metrics(gold, pred)
    assert abs(m["link_precision"] - 2 / 3) < 1e-9  # predicted A,B,X; correct A,B
    assert abs(m["link_recall"] - 0.5) < 1e-9  # gold A,B,C,D; correct 2
    assert abs(m["nil_precision"] - 0.5) < 1e-9  # predicted NIL x2; correct 1
    assert abs(m["nil_recall"] - 1.0) < 1e-9  # gold NIL x1; correct 1


def test_threshold_sweep_prefers_abstaining_on_weak_nil():
    from trialmatchai.entities.linker_eval import (
        GateInput,
        best_accept_threshold,
        sweep_thresholds,
    )

    inputs = [
        GateInput("A", (("A", 0.95),)),  # strong correct link
        GateInput("B", (("B", 0.80),)),  # correct link
        GateInput(None, (("Z", 0.55),)),  # NIL gold; weak top should be abstained
    ]
    rows = sweep_thresholds(inputs, [0.5, 0.7, 0.9], reject=0.4, margin=0.05)
    assert len(rows) == 3
    best = best_accept_threshold(inputs, [0.5, 0.7, 0.9], reject=0.4, margin=0.05)
    # accept=0.5 links Z (wrong, gold NIL); accept>=0.7 abstains on Z -> higher macro-F1.
    assert best["accept_threshold"] >= 0.7
    assert best["macro_f1"] > 0.6
