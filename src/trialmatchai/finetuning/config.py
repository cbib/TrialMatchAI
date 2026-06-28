"""Shared configuration for the LoRA fine-tuners (CoT and reranker)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union


@dataclass
class FinetuneConfig:
    """Hyper-parameters for LoRA supervised fine-tuning of a causal LM.

    Sensible defaults are tuned for a single consumer GPU; override per run.
    """

    base_model: str
    train_data: str
    output_dir: str
    eval_data: Optional[str] = None

    # Optimization
    epochs: float = 2.0
    learning_rate: float = 5e-5
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 16
    warmup_ratio: float = 0.1
    max_seq_length: int = 2048
    weight_decay: float = 0.0
    seed: int = 42

    # LoRA
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.1
    target_modules: Optional[Union[List[str], str]] = "all-linear"

    # Runtime
    load_in_4bit: bool = True
    bf16: bool = True
    trust_remote_code: bool = False
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: Optional[int] = None
    save_total_limit: int = 3
    max_examples: Optional[int] = None
    hf_token: Optional[str] = None
    device_map: Optional[str] = "auto"

    def to_training_arguments(self):
        """Build transformers.TrainingArguments (imported lazily)."""
        from transformers import TrainingArguments

        has_eval = self.eval_data is not None
        eval_steps = self.eval_steps or self.save_steps
        if has_eval and self.save_steps % eval_steps != 0:
            raise ValueError(
                "save_steps must be a multiple of eval_steps when eval_data is set "
                "because load_best_model_at_end requires aligned save/eval steps."
            )

        kwargs = dict(
            output_dir=self.output_dir,
            num_train_epochs=self.epochs,
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.per_device_batch_size,
            per_device_eval_batch_size=self.per_device_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            warmup_ratio=self.warmup_ratio,
            weight_decay=self.weight_decay,
            logging_steps=self.logging_steps,
            eval_strategy="steps" if has_eval else "no",
            save_strategy="steps",
            save_steps=self.save_steps,
            save_total_limit=self.save_total_limit,
            bf16=self.bf16,
            fp16=not self.bf16,
            seed=self.seed,
            report_to=[],
            ddp_find_unused_parameters=False,
            # QLoRA memory/throughput best practices: gradient checkpointing and
            # a paged optimizer keep large models on a single GPU.
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            optim="paged_adamw_8bit" if self.load_in_4bit else "adamw_torch",
            lr_scheduler_type="cosine",
        )
        if has_eval:
            kwargs.update(
                eval_steps=eval_steps,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
            )
        return TrainingArguments(**kwargs)
