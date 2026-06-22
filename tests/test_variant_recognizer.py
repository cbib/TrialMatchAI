"""Tests for the deterministic genetic-variant recognizer and compositing."""

from __future__ import annotations

from trialmatchai.entities.recognizers import (
    CompositeRecognizer,
    RegexSchemaRecognizer,
    RegexVariantRecognizer,
    _load_variant_patterns,
)
from trialmatchai.entities.schemas import load_entity_schemas


def test_variant_patterns_load():
    patterns = _load_variant_patterns()
    assert len(patterns) > 10  # the curated table has dozens of patterns
    assert all(hasattr(p, "finditer") for _, p in patterns)


def test_recognizes_hgvs_protein_and_dna_variants():
    recognizer = RegexVariantRecognizer()
    text = "BRAF p.V600E and the c.1799T>A variant were detected."
    spans = {ann.text for ann in recognizer.recognize([text], [])[0]}
    assert any("V600E" in s for s in spans)
    assert any("1799T>A" in s for s in spans)


def test_recognizes_gene_fusion():
    recognizer = RegexVariantRecognizer()
    spans = {ann.text.lower() for ann in recognizer.recognize(["EGFR gene fusion"], [])[0]}
    assert any("fusion" in s for s in spans)


def test_composite_merges_model_and_variant_spans():
    schemas = [s for s in load_entity_schemas() if s.id == "disease"]
    composite = CompositeRecognizer(
        RegexSchemaRecognizer(), RegexVariantRecognizer()
    )
    text = "Patient with cancer harboring BRAF p.V600E"
    annotations = composite.recognize([text], schemas)[0]
    groups = {ann.entity_group for ann in annotations}
    # Both the schema-recognized disease and the variant span are present.
    assert "disease" in groups
    assert any(ann.text.endswith("V600E") or "V600E" in ann.text for ann in annotations)


def test_no_zero_width_matches():
    recognizer = RegexVariantRecognizer()
    for ann in recognizer.recognize(["plain text without variants"], [])[0]:
        assert ann.end > ann.start
        assert ann.text.strip()
