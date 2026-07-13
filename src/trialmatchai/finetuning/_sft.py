"""Shared LoRA supervised fine-tuning loop for causal LMs.

Builds (input_ids, labels) from chat messages with the prompt masked to -100 so
loss is computed only on the assistant completion, then trains a LoRA adapter
and saves it. Heavy dependencies are imported lazily.
"""

from __future__ import annotations

import importlib.util
from typing import Dict, List, Optional

from trialmatchai.finetuning.config import FinetuneConfig
from trialmatchai.utils.logging_config import setup_logging

logger = setup_logging(__name__)


def _load_train_deps():
    try:
        import torch
        from datasets import Dataset
        from peft import (
            LoraConfig,
            TaskType,
            get_peft_model,
            prepare_model_for_kbit_training,
        )
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
        "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
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


def _validate_4bit_runtime(deps) -> None:
    torch = deps["torch"]
    if importlib.util.find_spec("bitsandbytes") is None:
        raise RuntimeError(
            "4-bit QLoRA requires bitsandbytes. Install the finetune extra on a "
            "CUDA-capable Linux/Windows environment, or pass --no-4bit."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "4-bit QLoRA through bitsandbytes requires CUDA. Pass --no-4bit "
            "for full/bfloat16 training on this machine."
        )


def _validate_messages(messages: List[Dict[str, str]]) -> None:
    if len(messages) < 2:
        raise ValueError("Each SFT row must contain at least one prompt and one answer.")
    if messages[-1].get("role") != "assistant":
        raise ValueError("Each SFT row must end with an assistant message.")
    for index, message in enumerate(messages):
        if message.get("role") not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"Unsupported chat role at message {index}: {message!r}")
        if "content" not in message:
            raise ValueError(f"Missing chat content at message {index}: {message!r}")


def _encode_example(
    tokenizer, messages: List[Dict[str, str]], max_seq_length: int
) -> Dict[str, list]:
    """Tokenize a chat example, masking the prompt so only the completion trains."""
    if max_seq_length <= 0:
        raise ValueError("max_seq_length must be positive.")
    _validate_messages(messages)
    if getattr(tokenizer, "chat_template", None) is None:
        raise ValueError(
            "The base model tokenizer does not define a chat template. Use a chat "
            "or instruction-tuned model, or add a tokenizer chat_template before training."
        )

    full_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False
    )
    prompt_ids = tokenizer.apply_chat_template(
        messages[:-1], tokenize=True, add_generation_prompt=True
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            "Tokenizer chat template did not produce a prompt prefix for the full "
            "conversation; refusing to risk training on prompt tokens."
        )

    prompt_len = len(prompt_ids)
    labels = [-100] * prompt_len + full_ids[prompt_len:]
    if len(full_ids) > max_seq_length:
        full_ids = full_ids[-max_seq_length:]
        labels = labels[-max_seq_length:]
    if all(label == -100 for label in labels):
        raise ValueError(
            "Encoded example has no trainable assistant tokens. Increase "
            "max_seq_length or shorten the prompt."
        )
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}


def _encode_dataset(deps, tokenizer, message_lists, max_seq_length):
    encoded = [
        _encode_example(tokenizer, messages, max_seq_length) for messages in message_lists
    ]
    if not encoded:
        raise ValueError("No examples provided.")
    return deps["Dataset"].from_list(encoded), len(encoded)


def run_sft(
    config: FinetuneConfig,
    message_lists: List[List[Dict[str, str]]],
    eval_message_lists: Optional[List[List[Dict[str, str]]]] = None,
) -> str:
    """Run LoRA SFT over a list of chat-message examples; returns the adapter dir."""
    if not message_lists:
        raise ValueError("No training examples provided.")
    deps = _load_train_deps()
    torch = deps["torch"]

    tokenizer = _build_tokenizer(deps, config)

    compute_dtype = torch.bfloat16 if config.bf16 else torch.float16
    quant_config = None
    if config.load_in_4bit:
        _validate_4bit_runtime(deps)
        quant_config = deps["BitsAndBytesConfig"](
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )

    model_kwargs = {
        "torch_dtype": compute_dtype,
        "quantization_config": quant_config,
        "trust_remote_code": config.trust_remote_code,
        "token": config.hf_token,
    }
    if config.load_in_4bit and config.device_map:
        model_kwargs["device_map"] = config.device_map

    model = deps["AutoModelForCausalLM"].from_pretrained(
        config.base_model, **model_kwargs
    )
    model.config.use_cache = False  # required with gradient checkpointing

    if config.load_in_4bit:
        # Freeze base weights, fp32 layer norms, gradient checkpointing: required for stable QLoRA.
        model = deps["prepare_model_for_kbit_training"](
            model, use_gradient_checkpointing=True
        )

    peft_kwargs = {
        "task_type": deps["TaskType"].CAUSAL_LM,
        "inference_mode": False,
        "r": config.lora_rank,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
    }
    if config.target_modules is not None:
        peft_kwargs["target_modules"] = config.target_modules
    peft_config = deps["LoraConfig"](**peft_kwargs)
    model = deps["get_peft_model"](model, peft_config)
    model.print_trainable_parameters()

    dataset, train_count = _encode_dataset(
        deps, tokenizer, message_lists, config.max_seq_length
    )
    eval_dataset = None
    eval_count = 0
    if eval_message_lists is not None:
        eval_dataset, eval_count = _encode_dataset(
            deps, tokenizer, eval_message_lists, config.max_seq_length
        )

    collator = deps["DataCollatorForSeq2Seq"](
        tokenizer, padding=True, label_pad_token_id=-100
    )
    trainer = deps["Trainer"](
        model=model,
        args=config.to_training_arguments(),
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    if eval_dataset is None:
        logger.info("Starting LoRA SFT on %d examples...", train_count)
    else:
        logger.info(
            "Starting LoRA SFT on %d examples with %d eval examples...",
            train_count,
            eval_count,
        )
    trainer.train()
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    logger.info("Saved LoRA adapter to %s", config.output_dir)
    return config.output_dir
