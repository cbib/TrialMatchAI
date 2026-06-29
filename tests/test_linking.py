"""Unit tests for the `link` stage: relinking persisted entities in place."""

from __future__ import annotations

import json

from trialmatchai.entities.linker import ConceptLinker, InMemoryConceptStore
from trialmatchai.entities.schemas import load_entity_schemas
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
