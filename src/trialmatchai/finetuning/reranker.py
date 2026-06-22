"""Fine-tune the cross-encoder reranker (LoRA SFT, Yes/No target).

The reranker reads the next-token logits for "Yes"/"No" after the prompt, so we
SFT the model to emit the correct token. Produces a LoRA adapter that plugs into
the pipeline via ``model.reranker_adapter_path``.
"""

from __future__ import annotations

from trialmatchai.finetuning._sft import run_sft
from trialmatchai.finetuning.config import FinetuneConfig
from trialmatchai.finetuning.data import read_jsonl, reranker_row_to_messages


def finetune_reranker(config: FinetuneConfig) -> str:
    rows = read_jsonl(config.train_data, config.max_examples)
    message_lists = []
    for row in rows:
        messages, label = reranker_row_to_messages(row)
        message_lists.append([*messages, {"role": "assistant", "content": label}])
    return run_sft(config, message_lists)
