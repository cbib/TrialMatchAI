"""Fine-tuning utilities for TrialMatchAI's local models.

Lets users train their own NER (GLiNER2), reranker (LoRA), and CoT eligibility
(LoRA) models and plug them straight back into the pipeline via config:

- NER:      entity_extraction.model_name -> a fine-tuned GLiNER2 checkpoint
- reranker: model.reranker_adapter_path  -> a LoRA adapter
- CoT:      model.cot_adapter_path        -> a LoRA adapter

All heavy training dependencies are imported lazily, so this package imports
cleanly without the ``finetune`` extra installed. Install it with
``uv sync --extra finetune``.
"""

from trialmatchai.finetuning.config import FinetuneConfig

__all__ = ["FinetuneConfig"]
