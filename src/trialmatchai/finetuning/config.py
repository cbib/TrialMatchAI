"""Shared configuration for the LoRA fine-tuners (CoT and reranker)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


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
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    # Runtime
    load_in_4bit: bool = True
    bf16: bool = True
    trust_remote_code: bool = False
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 3
    max_examples: Optional[int] = None
    hf_token: Optional[str] = None

    def to_training_arguments(self):
        """Build transformers.TrainingArguments (imported lazily)."""
        from transformers import TrainingArguments

        return TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.epochs,
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.per_device_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            warmup_ratio=self.warmup_ratio,
            weight_decay=self.weight_decay,
            logging_steps=self.logging_steps,
            save_steps=self.save_steps,
            save_total_limit=self.save_total_limit,
            bf16=self.bf16,
            fp16=not self.bf16,
            seed=self.seed,
            report_to=[],
            ddp_find_unused_parameters=False,
        )
