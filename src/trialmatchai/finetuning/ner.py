"""Fine-tune the biomedical NER model (GLiNER).

Produces a GLiNER checkpoint that plugs into the pipeline via
``entity_extraction.model_name`` (and backend "gliner"/"gliner2").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from trialmatchai.finetuning.data import iter_gliner_examples
from trialmatchai.utils.logging_config import setup_logging

logger = setup_logging(__name__)


@dataclass
class NERFinetuneConfig:
    base_model: str
    train_data: str
    output_dir: str
    eval_data: Optional[str] = None
    epochs: float = 3.0
    learning_rate: float = 5e-6
    batch_size: int = 8
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_examples: Optional[int] = None
    labels: List[str] = field(default_factory=list)
    seed: int = 42


def finetune_ner(config: NERFinetuneConfig) -> str:
    """Fine-tune a GLiNER model on span-annotated data and save it."""
    try:
        from gliner import GLiNER
        from gliner.data_processing.collator import DataCollator
        from gliner.training import Trainer, TrainingArguments
    except Exception as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "GLiNER fine-tuning requires the optional `finetune` dependencies "
            "(`uv sync --extra finetune`). If your installed gliner exposes a "
            "different training API, adapt finetuning/ner.py to it."
        ) from exc

    data = list(iter_gliner_examples(config.train_data, config.max_examples))
    if not data:
        raise ValueError("No training examples provided.")
    eval_data = (
        list(iter_gliner_examples(config.eval_data)) if config.eval_data else None
    )

    model = GLiNER.from_pretrained(config.base_model)
    model.set_sampling_params(
        max_types=25, shuffle_types=True, random_drop=True, max_neg_type_ratio=1
    )

    collator = DataCollator(
        model.config, data_processor=model.data_processor, prepare_labels=True
    )

    args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.epochs,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        seed=config.seed,
        report_to=[],
        evaluation_strategy="epoch" if eval_data else "no",
        save_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=data,
        eval_dataset=eval_data,
        data_collator=collator,
        tokenizer=model.data_processor.transformer_tokenizer,
    )

    logger.info("Starting GLiNER fine-tuning on %d examples...", len(data))
    trainer.train()
    model.save_pretrained(config.output_dir)
    logger.info("Saved fine-tuned GLiNER model to %s", config.output_dir)
    return config.output_dir
