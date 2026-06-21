from __future__ import annotations

from typing import Any

from Matcher.entities.linker import ConceptLinker


class Normalizer:
    """Legacy placeholder for Parser callers.

    Normalization now happens in Matcher.entities.ConceptLinker. This class keeps old
    imports from crashing while avoiding external normalizer daemons.
    """

    NO_ENTITY_ID = "CUI-less"

    def __init__(self, *args: Any, concept_linker: ConceptLinker | None = None, **kwargs: Any):
        del args, kwargs
        self.concept_linker = concept_linker
        self.use_neural_normalizer = False

    def normalize(self, base_name: str, doc_dict_list: list[dict[str, Any]]):
        del base_name
        return doc_dict_list

    def neural_normalize(self, ent_type: str, tagged_docs: list[dict[str, Any]]):
        del ent_type
        return tagged_docs
