"""P3: config validation guards — reject nonsensical settings at load time."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from trialmatchai.config.settings import ConceptLinkerSettings, EmbedderSettings


def test_embedder_pooling_enum():
    EmbedderSettings(pooling="mean")
    EmbedderSettings(pooling="cls")
    with pytest.raises(ValidationError):
        EmbedderSettings(pooling="max")


def test_embedder_batch_size_and_max_length_positive():
    EmbedderSettings(batch_size=1, max_length=1)
    with pytest.raises(ValidationError):
        EmbedderSettings(batch_size=0)
    with pytest.raises(ValidationError):
        EmbedderSettings(max_length=0)


def test_concept_linker_reject_not_above_accept():
    ConceptLinkerSettings(accept_threshold=0.8, reject_threshold=0.3)
    with pytest.raises(ValidationError):
        ConceptLinkerSettings(accept_threshold=0.3, reject_threshold=0.8)
