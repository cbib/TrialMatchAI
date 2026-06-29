"""P3: the regex variant recognizer must not fire on plain English. Curated cues
like 'loss'/'increased' are kept only when a gene/biomarker token is nearby."""

from __future__ import annotations

from trialmatchai.entities.recognizers import RegexVariantRecognizer


def _labels(rec, text):
    return {a.entity_group for a in rec.recognize([text], [])[0]}


def test_bare_ambiguous_words_need_gene_context():
    rec = RegexVariantRecognizer()
    # plain prose ending in an ambiguous cue -> no variant emitted
    assert "loss" not in _labels(rec, "The patient reports appetite loss")
    assert "amplification" not in _labels(rec, "Symptoms have increased")
    assert "insertion" not in _labels(rec, "The catheter required insertion")


def test_ambiguous_words_kept_with_gene_context():
    rec = RegexVariantRecognizer()
    assert "loss" in _labels(rec, "Tumor shows PTEN loss")
    assert "amplification" in _labels(rec, "MYC increased")


def test_specific_variant_cues_still_detected():
    rec = RegexVariantRecognizer()
    # 'amplification' itself is unambiguous; never gated
    assert "amplification" in _labels(rec, "Observed EGFR amplification")
