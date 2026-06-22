"""Fine-tune the chain-of-thought eligibility model (LoRA SFT).

Produces a LoRA adapter that plugs into the pipeline via
``model.cot_adapter_path`` (HuggingFace backend) or vLLM's LoRARequest.
"""

from __future__ import annotations

from trialmatchai.finetuning._sft import run_sft
from trialmatchai.finetuning.config import FinetuneConfig
from trialmatchai.finetuning.data import cot_row_to_messages, read_jsonl


def finetune_cot(config: FinetuneConfig) -> str:
    rows = read_jsonl(config.train_data, config.max_examples)
    message_lists = [cot_row_to_messages(row) for row in rows]
    return run_sft(config, message_lists)
