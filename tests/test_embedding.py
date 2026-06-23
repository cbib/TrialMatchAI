from __future__ import annotations

from trialmatchai.models.embedding import HashingTextEmbedder, build_embedder


def test_hashing_embedder_is_deterministic_and_normalized():
    embedder = HashingTextEmbedder(dimensions=16)
    first = embedder.embed_text("EGFR lung cancer")
    second = embedder.embed_text("EGFR lung cancer")

    assert first == second
    assert len(first) == 16
    assert round(sum(value * value for value in first), 6) == 1.0


def test_build_embedder_supports_hashing_backend():
    embedder = build_embedder(
        {"embedder": {"backend": "hashing", "hashing_dimensions": 8}}
    )

    assert isinstance(embedder, HashingTextEmbedder)
    assert len(embedder.embed_text("melanoma")) == 8
