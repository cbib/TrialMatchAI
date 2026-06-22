"""Shared LoRA supervised fine-tuning loop for causal LMs.

Builds (input_ids, labels) from chat messages with the prompt masked to -100 so
loss is computed only on the assistant completion, then trains a LoRA adapter
and saves it. Heavy dependencies are imported lazily.
"""

from __future__ import annotations

from typing import Dict, List

from trialmatchai.finetuning.config import FinetuneConfig
from trialmatchai.utils.logging_config import setup_logging

logger = setup_logging(__name__)


def _load_train_deps():
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            DataCollatorForSeq2Seq,
            Trainer,
        )
    except Exception as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "Fine-tuning requires the optional `finetune` dependencies "
            "(`uv sync --extra finetune`)."
        ) from exc
    return {
        "torch": torch,
        "Dataset": Dataset,
        "LoraConfig": LoraConfig,
        "TaskType": TaskType,
        "get_peft_model": get_peft_model,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "DataCollatorForSeq2Seq": DataCollatorForSeq2Seq,
        "Trainer": Trainer,
    }


def _build_tokenizer(deps, config: FinetuneConfig):
    tokenizer = deps["AutoTokenizer"].from_pretrained(
        config.base_model,
        trust_remote_code=config.trust_remote_code,
        token=config.hf_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _encode_example(
    tokenizer, messages: List[Dict[str, str]], max_seq_length: int
) -> Dict[str, list]:
    """Tokenize a chat example, masking the prompt so only the completion trains."""
    full = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    prompt = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    full_ids = tokenizer(full, truncation=True, max_length=max_seq_length)["input_ids"]
    prompt_ids = tokenizer(
        prompt, truncation=True, max_length=max_seq_length
    )["input_ids"]
    prompt_len = min(len(prompt_ids), len(full_ids))
    labels = [-100] * prompt_len + full_ids[prompt_len:]
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}


def run_sft(config: FinetuneConfig, message_lists: List[List[Dict[str, str]]]) -> str:
    """Run LoRA SFT over a list of chat-message examples; returns the adapter dir."""
    if not message_lists:
        raise ValueError("No training examples provided.")
    deps = _load_train_deps()
    torch = deps["torch"]

    tokenizer = _build_tokenizer(deps, config)

    compute_dtype = torch.bfloat16 if config.bf16 else torch.float16
    quant_config = None
    if config.load_in_4bit:
        quant_config = deps["BitsAndBytesConfig"](
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )

    model = deps["AutoModelForCausalLM"].from_pretrained(
        config.base_model,
        torch_dtype=compute_dtype,
        quantization_config=quant_config,
        trust_remote_code=config.trust_remote_code,
        token=config.hf_token,
    )
    model.config.use_cache = False

    peft_config = deps["LoraConfig"](
        task_type=deps["TaskType"].CAUSAL_LM,
        inference_mode=False,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
    )
    model = deps["get_peft_model"](model, peft_config)
    model.print_trainable_parameters()

    encoded = [
        _encode_example(tokenizer, messages, config.max_seq_length)
        for messages in message_lists
    ]
    dataset = deps["Dataset"].from_list(encoded)

    collator = deps["DataCollatorForSeq2Seq"](
        tokenizer, padding=True, label_pad_token_id=-100
    )
    trainer = deps["Trainer"](
        model=model,
        args=config.to_training_arguments(),
        train_dataset=dataset,
        data_collator=collator,
    )

    logger.info("Starting LoRA SFT on %d examples...", len(encoded))
    trainer.train()
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    logger.info("Saved LoRA adapter to %s", config.output_dir)
    return config.output_dir
