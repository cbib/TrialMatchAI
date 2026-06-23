"""Fine-tune the chain-of-thought eligibility model (LoRA SFT).

Produces a LoRA adapter that plugs into the pipeline via
``model.cot_adapter_path`` (HuggingFace backend) or vLLM's LoRARequest.
"""

from __future__ import annotations

from trialmatchai.finetuning._sft import run_sft
from trialmatchai.finetuning.config import FinetuneConfig
from trialmatchai.finetuning.data import cot_row_to_messages, read_jsonl


def _load_messages(path: str, max_examples: int | None):
    return [cot_row_to_messages(row) for row in read_jsonl(path, max_examples)]


def finetune_cot(config: FinetuneConfig) -> str:
    message_lists = _load_messages(config.train_data, config.max_examples)
    eval_message_lists = (
        _load_messages(config.eval_data, config.max_examples)
        if config.eval_data is not None
        else None
    )
    return run_sft(config, message_lists, eval_message_lists)
