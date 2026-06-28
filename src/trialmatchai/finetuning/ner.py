"""Fine-tune the biomedical NER model (GLiNER2).

Uses the native GLiNER2 training stack (GLiNER2Trainer / TrainingConfig /
InputExample). Produces either a full checkpoint (``<output_dir>/best``) or a
LoRA adapter (``<output_dir>/final``) that plugs into the pipeline via
``entity_extraction.model_name`` with backend "gliner2".

GLiNER2 data can be flat entity extraction, schema-based JSON structures,
classifications, relations, or a mix of those tasks. Character-span NER rows
are converted automatically. ``entity_descriptions`` is back-filled from the
entity schema so flat entity examples use the same label semantics as inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from trialmatchai.finetuning.data import gliner2_row_to_training_record, read_jsonl
from trialmatchai.utils.logging_config import setup_logging

logger = setup_logging(__name__)


@dataclass
class NERFinetuneConfig:
    train_data: str
    output_dir: str
    base_model: str = "fastino/gliner2-base-v1"
    eval_data: Optional[str] = None
    epochs: float = 10.0
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    encoder_lr: float = 1e-5
    task_lr: float = 5e-4
    warmup_ratio: float = 0.1
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    fp16: bool = True
    eval_steps: int = 500
    logging_steps: int = 50
    save_total_limit: int = 3
    max_examples: Optional[int] = None
    schema_path: Optional[str] = None
    seed: int = 42


def _schema_descriptions(schema_path: Optional[str]) -> Dict[str, str]:
    """Map entity-group label -> description from the entity schema, if available."""
    try:
        from trialmatchai.entities.schemas import load_entity_schemas

        schemas = load_entity_schemas(schema_path)
    except Exception:  # pragma: no cover - schema optional during training
        return {}
    descriptions: Dict[str, str] = {}
    for schema in schemas:
        if schema.description:
            descriptions[schema.entity_group] = schema.description
            descriptions[schema.label] = schema.description
    return descriptions


def _backfill_entity_descriptions(
    record: Dict[str, object], descriptions: Dict[str, str]
) -> Dict[str, object]:
    output = dict(record.get("output") or {})
    entities = output.get("entities") or {}
    if not isinstance(entities, dict):
        return record
    descs = dict(output.get("entity_descriptions") or {})
    for label in entities:
        if label not in descs and label in descriptions:
            descs[label] = descriptions[label]
    if descs:
        output["entity_descriptions"] = descs
        record = dict(record)
        record["output"] = output
    return record


def _sanitize_example(example) -> bool:
    if not hasattr(example, "sanitize"):
        return True
    warnings, is_valid = example.sanitize()
    for warning in warnings:
        logger.warning("Sanitized GLiNER2 training example: %s", warning)
    return is_valid


def _validate_dataset(dataset) -> None:
    try:
        dataset.validate(raise_on_error=False)
    except TypeError:  # pragma: no cover - older GLiNER2 compatibility
        dataset.validate(strict=False, raise_on_error=False)


def _build_examples(config: "NERFinetuneConfig", path: str, descriptions: Dict[str, str]):
    from gliner2.training.data import InputExample

    examples = []
    for row in read_jsonl(path, config.max_examples):
        record = gliner2_row_to_training_record(row)
        record["input"] = str(record["input"]).strip()
        if not record["input"]:
            continue
        record = _backfill_entity_descriptions(record, descriptions)
        example = InputExample.from_dict(record)
        if _sanitize_example(example):
            examples.append(example)
    return examples


def finetune_ner(config: NERFinetuneConfig) -> str:
    try:
        from gliner2 import GLiNER2
        from gliner2.training.data import TrainingDataset
        from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig
    except Exception as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "GLiNER2 fine-tuning requires the optional `finetune` dependencies "
            "(`uv sync --extra finetune`)."
        ) from exc

    descriptions = _schema_descriptions(config.schema_path)
    train_examples = _build_examples(config, config.train_data, descriptions)
    if not train_examples:
        raise ValueError("No training examples provided.")

    train_dataset = TrainingDataset(train_examples)
    _validate_dataset(train_dataset)
    train_dataset.print_stats()

    if config.eval_data:
        train_data: object = train_dataset
        eval_examples = _build_examples(config, config.eval_data, descriptions)
        if not eval_examples:
            raise ValueError("No evaluation examples provided.")
        val_data: object | None = TrainingDataset(eval_examples)
    elif len(train_examples) >= 10:
        train_data, val_data, _ = train_dataset.split(
            train_ratio=0.9, val_ratio=0.1, test_ratio=0.0, shuffle=True, seed=config.seed
        )
    else:
        train_data = train_dataset
        val_data = None

    has_eval = val_data is not None

    training_config = TrainingConfig(
        output_dir=config.output_dir,
        experiment_name="trialmatchai_ner",
        num_epochs=config.epochs,
        batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        encoder_lr=config.encoder_lr,
        task_lr=config.task_lr,
        warmup_ratio=config.warmup_ratio,
        scheduler_type="cosine",
        fp16=config.fp16,
        eval_strategy="epoch" if has_eval else "no",
        eval_steps=config.eval_steps,
        save_total_limit=config.save_total_limit,
        save_best=has_eval,
        early_stopping=has_eval,
        early_stopping_patience=3,
        logging_steps=config.logging_steps,
        use_lora=config.use_lora,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        lora_target_modules=["encoder"],
        save_adapter_only=config.use_lora,
    )

    model = GLiNER2.from_pretrained(config.base_model)
    trainer = GLiNER2Trainer(model, training_config)
    logger.info("Starting GLiNER2 fine-tuning on %d examples...", len(train_examples))
    trainer.train(train_data=train_data, val_data=val_data)

    result_dir = (
        f"{config.output_dir.rstrip('/')}/final"
        if config.use_lora or not has_eval
        else f"{config.output_dir.rstrip('/')}/best"
    )
    logger.info(
        "Saved %s to %s. Set entity_extraction.model_name to this path "
        "(LoRA adapters load via GLiNER2.load_adapter).",
        "LoRA adapter" if config.use_lora else "fine-tuned model",
        result_dir,
    )
    return result_dir


__all__ = ["NERFinetuneConfig", "finetune_ner"]
