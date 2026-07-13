"""Self-describing embedder contract + metric-follows-embedder (Phase 3)."""

from trialmatchai.models.embedding.text_embedder import (
    HashingTextEmbedder,
    build_embedder,
    native_metric_from_config,
)
from trialmatchai.search import build_search_backend


def test_hashing_embedder_self_describes():
    e = build_embedder({"embedder": {"backend": "hashing", "hashing_dimensions": 24, "normalize": False}})
    assert e.dim == 24
    assert e.native_metric == "dot"  # unnormalized -> dot
    assert e.is_asymmetric is False
    assert isinstance(e.fingerprint(), str) and e.fingerprint()


def test_native_metric_derivation():
    assert native_metric_from_config({"embedder": {"normalize": True}}) == "cosine"
    assert native_metric_from_config({"embedder": {"normalize": False}}) == "dot"
    assert native_metric_from_config({"embedder": {"normalize": True, "native_metric": "dot"}}) == "dot"


def test_search_backend_metric_follows_embedder():
    # No explicit vector_metric -> follows the embedder's implied space.
    assert build_search_backend({"embedder": {"normalize": True}, "search_backend": {}}).vector_metric == "cosine"
    assert build_search_backend({"embedder": {"normalize": False}, "search_backend": {}}).vector_metric == "dot"
    # Explicit vector_metric still wins (back-compat).
    b = build_search_backend({"embedder": {"normalize": False}, "search_backend": {"vector_metric": "cosine"}})
    assert b.vector_metric == "cosine"


def test_fingerprint_changes_with_identity():
    assert HashingTextEmbedder(32, False).fingerprint() == HashingTextEmbedder(32, False).fingerprint()
    assert HashingTextEmbedder(32, False).fingerprint() != HashingTextEmbedder(64, False).fingerprint()
    assert HashingTextEmbedder(32, True).fingerprint() != HashingTextEmbedder(32, False).fingerprint()
