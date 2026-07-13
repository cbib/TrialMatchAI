"""Component registry (Phase 2 of the modularity refactor)."""

import pytest

from trialmatchai.plugins import register, resolve, registered
from trialmatchai.search import build_search_backend
from trialmatchai.models.embedding.text_embedder import build_embedder


def test_builtins_registered():
    assert "lancedb" in registered("search_backend")
    assert {"hf", "hashing"} <= set(registered("embedder"))


def test_resolve_unknown_lists_registered():
    with pytest.raises(ValueError, match="registered: lancedb"):
        resolve("search_backend", "nope")


def test_search_backend_type_and_legacy_backend_both_resolve():
    a = build_search_backend({"search_backend": {"type": "lancedb", "db_path": "data/search"}})
    b = build_search_backend({"search_backend": {"backend": "lancedb", "db_path": "data/search"}})
    assert type(a) is type(b)


def test_build_search_backend_bad_type_raises():
    with pytest.raises(ValueError, match="unknown search_backend type"):
        build_search_backend({"search_backend": {"type": "lancdb"}})


def test_embedder_registry_builds_hashing():
    emb = build_embedder({"embedder": {"backend": "hashing", "hashing_dimensions": 16}})
    assert len(emb.embed_texts(["hello"])[0]) == 16


def test_register_new_backend_needs_no_core_edit():
    sentinel = object()
    register("search_backend", "_test_mem")(lambda config: sentinel)
    assert build_search_backend({"search_backend": {"type": "_test_mem"}}) is sentinel
